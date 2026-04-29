"""
hybrid_integration.py
=====================
Implements the three-tier event-handling strategy from the paper:

  Tier 1 — Greedy Insertion  (O(V·L), instant)
  Tier 2 — Partial Re-opt    (OR-Tools on affected vehicles only)
  Tier 3 — Global Re-opt     (full OR-Tools, warm-started by RL)

Entry point:
    HybridRouter.optimize(orders, n_vehicles, capacity)
        → initial routes

    HybridRouter.handle_event(event_type, payload)
        → updated routes
"""

import os
import sys
import time
import logging
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ortools_solver import (solve_vrp, _dist_km, TRAFFIC_FACTOR as _TF,
                            fuel_consumption, co2_kg,
                            BASELINE_FUEL, CO2_PER_LITRE)
from delivery_env import TIME_WINDOW, LATE_PENALTY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SPEED_KMH = 40.0   # realistic urban delivery speed — must match delivery_env.py
GREEDY_THRESHOLD = 5.0   # km — accept greedy insertion if cost delta ≤ this


def _eta(lat1, lon1, lat2, lon2, traffic=1) -> float:
    d = _dist_km(lat1, lon1, lat2, lon2)
    return (d / SPEED_KMH) * 60.0 * _TF.get(traffic, 1.0)


def _finite_float(value, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    if n != n or n in (float("inf"), float("-inf")):
        return default
    return n


def _bounded_int(value, default: int, low: int, high: int) -> int:
    n = int(round(_finite_float(value, default)))
    return max(low, min(high, n))


# ── Greedy Insertion ───────────────────────────────────────────────────────────
def greedy_insert(
    new_order    : dict,
    routes       : list[list[dict]],
    threshold_km : float = GREEDY_THRESHOLD,
) -> tuple[bool, list[list[dict]]]:
    """
    Try to insert new_order at the cheapest position in any existing route.
    Returns (success, updated_routes).
    """
    best_delta   = float("inf")
    best_vehicle = -1
    best_pos     = -1

    traffic = int(new_order.get("Road_traffic_density", 1))

    for v_idx, route in enumerate(routes):
        stops = route  # list of order dicts
        # try inserting before each position (including end)
        for pos in range(0, len(stops) + 1):
            prev = stops[pos - 1] if pos > 0 else None
            nxt  = stops[pos]     if pos < len(stops) else None

            if prev is None and nxt is None:
                delta = (_dist_km(new_order["pickup_lat"], new_order["pickup_lon"],
                                  new_order["drop_lat"],   new_order["drop_lon"]))
            elif nxt is None:
                a_lat, a_lon = prev["drop_lat"], prev["drop_lon"]
                delta = _dist_km(a_lat, a_lon,
                                 new_order["pickup_lat"], new_order["pickup_lon"]) + \
                        _dist_km(new_order["pickup_lat"], new_order["pickup_lon"],
                                 new_order["drop_lat"],   new_order["drop_lon"])
            elif prev is None:
                b_lat, b_lon = nxt["pickup_lat"], nxt["pickup_lon"]
                cost_before = 0
                cost_after  = _dist_km(new_order["pickup_lat"], new_order["pickup_lon"],
                                       new_order["drop_lat"],   new_order["drop_lon"]) + \
                              _dist_km(new_order["drop_lat"], new_order["drop_lon"],
                                       b_lat, b_lon)
                delta = cost_after - cost_before
            else:
                a_lat, a_lon = prev["drop_lat"], prev["drop_lon"]
                b_lat, b_lon = nxt["pickup_lat"], nxt["pickup_lon"]
                cost_before  = _dist_km(a_lat, a_lon, b_lat, b_lon)
                cost_after   = (_dist_km(a_lat, a_lon,
                                         new_order["pickup_lat"], new_order["pickup_lon"]) +
                                _dist_km(new_order["pickup_lat"], new_order["pickup_lon"],
                                         new_order["drop_lat"],   new_order["drop_lon"]) +
                                _dist_km(new_order["drop_lat"],   new_order["drop_lon"],
                                         b_lat, b_lon))
                delta = cost_after - cost_before

            if delta < best_delta:
                best_delta   = delta
                best_vehicle = v_idx
                best_pos     = pos

    if best_delta <= threshold_km and best_vehicle >= 0:
        routes[best_vehicle].insert(best_pos, new_order)
        logger.info(f"[GreedyInsert] Inserted order {new_order.get('order_id')} "
                    f"→ vehicle {best_vehicle} pos {best_pos}  Δ={best_delta:.2f} km")
        return True, routes

    logger.info(f"[GreedyInsert] Failed (best Δ={best_delta:.2f} km > threshold)")
    return False, routes


# ── HybridRouter ───────────────────────────────────────────────────────────────
class HybridRouter:
    """
    Stateful router that holds current routes and handles events.
    """

    def __init__(
        self,
        depot_lat     : float,
        depot_lon     : float,
        n_vehicles    : int   = 3,
        capacity      : int   = 10,
        or_time_limit : int   = 10,
        rl_model      = None,          # optional loaded SB3 PPO model
        vecnorm       = None,          # optional VecNormalize stats
    ):
        self.depot_lat     = depot_lat
        self.depot_lon     = depot_lon
        self.n_vehicles    = n_vehicles
        self.capacity      = capacity
        self.or_time_limit = or_time_limit
        self.rl_model      = rl_model
        self.vecnorm       = vecnorm

        self.routes     : list[list[dict]] = [[] for _ in range(n_vehicles)]
        self.all_orders : list[dict]       = []
        # Multi-objective weights (fuel + CO2 aware)
        self.alpha = 0.50   # distance
        self.beta  = 0.20   # time
        self.gamma = 0.20   # fuel
        self.delta = 0.10   # CO2
        self.rl_max_orders = self._infer_rl_max_orders()
        self.last_warm_start = {
            "used": False,
            "reason": "RL model not loaded.",
            "suggested_orders": 0,
            "max_supported_orders": self.rl_max_orders,
        }

    # ── initial optimisation ───────────────────────────────────────────────────
    def optimize(self, orders: list[dict]) -> dict:
        """
        Full initial optimisation: RL warm-start → OR-Tools.
        """
        self.all_orders = list(orders)

        # Step 1: RL warm-start (permutation suggestion)
        rl_perm = self._rl_suggest(orders)

        # Step 2: OR-Tools VRP
        result = solve_vrp(
            orders       = orders,
            depot_lat    = self.depot_lat,
            depot_lon    = self.depot_lon,
            n_vehicles   = self.n_vehicles,
            capacity     = self.capacity,
            time_limit   = self.or_time_limit,
            warm_start   = rl_perm,
            alpha        = self.alpha,
            beta         = self.beta,
            gamma        = self.gamma,
            delta        = self.delta,
        )

        # Build route dicts from index lists
        self.routes = []
        for idx_list in result["routes"]:
            self.routes.append([orders[i] for i in idx_list])
        # pad to n_vehicles
        while len(self.routes) < self.n_vehicles:
            self.routes.append([])

        logger.info(f"[Optimize] {result['status']}  "
                    f"dist={result['total_dist_km']} km  "
                    f"time={result['total_time_min']} min  "
                    f"OR solve={result['solve_time_s']} s")
        solver_metrics = {
            "solver_total_dist_km": result.get("total_dist_km"),
            "solver_total_time_min": result.get("total_time_min"),
            "solver_total_fuel_L": result.get("total_fuel_L"),
            "solver_total_co2_kg": result.get("total_co2_kg"),
        }
        metrics = self._compute_live_metrics()
        return {
            **result,
            **solver_metrics,
            **metrics,
            "routes_detail": metrics["routes_detail"],
            "warm_start": dict(self.last_warm_start),
        }

    # ── event handling ─────────────────────────────────────────────────────────
    def handle_event(self, event_type: str, payload: dict) -> dict:
        """
        event_type : "NEW_ORDER" | "TRAFFIC_UPDATE" | "DELAY"
        payload    : varies by event type
        """
        t0 = time.perf_counter()

        if event_type == "NEW_ORDER":
            new_order = payload["order"]
            # Tier 1 — greedy
            success, self.routes = greedy_insert(new_order, self.routes)
            if success:
                self.all_orders.append(new_order)
                return self._event_result("GREEDY", t0)

            # Tier 2 — partial re-opt on nearest vehicles
            affected = self._nearest_vehicles(new_order, k=2)
            affected_orders = [o for v in affected for o in self.routes[v]] + [new_order]
            result = solve_vrp(
                orders    = affected_orders,
                depot_lat = self.depot_lat,
                depot_lon = self.depot_lon,
                n_vehicles= len(affected),
                capacity  = self.capacity,
                time_limit= max(3, self.or_time_limit // 2),
            )
            if result["routes"]:
                for i, v in enumerate(affected):
                    self.routes[v] = ([affected_orders[j] for j in result["routes"][i]]
                                      if i < len(result["routes"]) else [])
                self.all_orders.append(new_order)
                logger.info(f"[PartialReopt] Used for NEW_ORDER")
            return self._event_result("PARTIAL_REOPT", t0)

        elif event_type == "TRAFFIC_UPDATE":
            # Tier 3 — global re-opt with updated traffic
            rl_perm = self._rl_suggest(self.all_orders)
            result  = solve_vrp(
                orders    = self.all_orders,
                depot_lat = self.depot_lat,
                depot_lon = self.depot_lon,
                n_vehicles= self.n_vehicles,
                capacity  = self.capacity,
                time_limit= self.or_time_limit,
                warm_start= rl_perm,
                alpha     = self.alpha,
                beta      = self.beta,
                gamma     = self.gamma,
                delta     = self.delta,
            )
            self.routes = []
            for idx_list in result["routes"]:
                self.routes.append([self.all_orders[i] for i in idx_list])
            while len(self.routes) < self.n_vehicles:
                self.routes.append([])
            return self._event_result("GLOBAL_REOPT", t0)

        elif event_type == "DELAY":
            # treat like traffic update
            return self.handle_event("TRAFFIC_UPDATE", payload)

        else:
            raise ValueError(f"Unknown event_type: {event_type}")

    # ── helpers ────────────────────────────────────────────────────────────────
    def _rl_suggest(self, orders: list[dict]) -> Optional[list[int]]:
        """Return RL-suggested permutation of order indices or None."""
        if self.rl_model is None or not orders:
            self.last_warm_start = {
                "used": False,
                "reason": "RL model not loaded.",
                "suggested_orders": 0,
                "max_supported_orders": self.rl_max_orders,
            }
            return None
        if self.rl_max_orders is None:
            self.last_warm_start = {
                "used": False,
                "reason": "Could not infer RL observation size.",
                "suggested_orders": 0,
                "max_supported_orders": None,
            }
            return None
        if len(orders) > self.rl_max_orders:
            self.last_warm_start = {
                "used": False,
                "reason": (f"Current RL checkpoint supports up to "
                           f"{self.rl_max_orders} orders, got {len(orders)}."),
                "suggested_orders": 0,
                "max_supported_orders": self.rl_max_orders,
            }
            logger.info("[RL suggest] %s", self.last_warm_start["reason"])
            return None
        try:
            from delivery_env import DeliveryEnv
            import pandas as pd
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            df  = pd.DataFrame(self._prepare_rl_orders(orders))
            env = DummyVecEnv([lambda: DeliveryEnv(df, max_orders=self.rl_max_orders)])
            if self.vecnorm:
                env = VecNormalize.load(self.vecnorm, env)
                env.training = False
                env.norm_reward = False

            obs    = env.reset()
            perm   = []
            for _ in range(len(orders)):
                remaining_idx = list(env.envs[0].remaining_idx)
                if not remaining_idx:
                    break
                action, _ = self.rl_model.predict(obs, deterministic=True)
                chosen_pos = int(action[0]) % len(remaining_idx)
                perm.append(int(remaining_idx[chosen_pos]))
                obs, _, done, _ = env.step(action)
                if done[0]:
                    break
            self.last_warm_start = {
                "used": bool(perm),
                "reason": ("Applied PPO warm-start." if perm
                           else "RL model returned an empty suggestion."),
                "suggested_orders": len(perm),
                "max_supported_orders": self.rl_max_orders,
            }
            return perm
        except Exception as e:
            self.last_warm_start = {
                "used": False,
                "reason": f"RL warm-start failed: {e}",
                "suggested_orders": 0,
                "max_supported_orders": self.rl_max_orders,
            }
            logger.warning(f"[RL suggest] Failed ({e}), no warm-start")
            return None

    def _prepare_rl_orders(self, orders: list[dict]) -> list[dict]:
        """Return PPO-safe order records with no NaN/None values in env features."""
        prepared = []
        for raw in orders:
            order = dict(raw)
            pickup_lat = _finite_float(order.get("pickup_lat"), self.depot_lat)
            pickup_lon = _finite_float(order.get("pickup_lon"), self.depot_lon)
            drop_lat = _finite_float(order.get("drop_lat"), pickup_lat)
            drop_lon = _finite_float(order.get("drop_lon"), pickup_lon)
            traffic = _bounded_int(order.get("Road_traffic_density"), 1, 0, 3)
            weather = _bounded_int(order.get("Weather_conditions"), 0, 0, 6)
            order_time = _bounded_int(order.get("order_time_min"), 480, 0, 1440)
            pickup_time = _bounded_int(order.get("pickup_time_min"), order_time, 0, 1440)
            distance = _finite_float(
                order.get("distance_km"),
                _dist_km(pickup_lat, pickup_lon, drop_lat, drop_lon),
            )
            eta = _finite_float(
                order.get("est_time_derived"),
                _finite_float(order.get("est_time"), (distance / SPEED_KMH) * 60.0 * _TF.get(traffic, 1.0)),
            )
            wait_time = _finite_float(order.get("wait_time_min"), max(0.0, pickup_time - order_time))

            order.update({
                "pickup_lat": pickup_lat,
                "pickup_lon": pickup_lon,
                "drop_lat": drop_lat,
                "drop_lon": drop_lon,
                "distance_km": distance,
                "est_time": eta,
                "est_time_derived": eta,
                "Road_traffic_density": traffic,
                "Weather_conditions": weather,
                "order_time_min": order_time,
                "pickup_time_min": pickup_time,
                "wait_time_min": wait_time,
                "time_taken_min": _finite_float(order.get("time_taken_min"), eta),
                "fuel_L": _finite_float(order.get("fuel_L"), distance * BASELINE_FUEL),
            })
            prepared.append(order)
        return prepared

    def _infer_rl_max_orders(self) -> Optional[int]:
        """Infer the training-time max_orders from the loaded policy shape."""
        obs_space = getattr(self.rl_model, "observation_space", None)
        shape = getattr(obs_space, "shape", None)
        if not shape:
            return None
        obs_dim = int(shape[0])
        if obs_dim < 4 or (obs_dim - 4) % 10 != 0:
            return None
        return (obs_dim - 4) // 10

    def evaluate_rl_policy(self, orders: list[dict], n_episodes: int = 5) -> dict:
        """
        Run the PPO policy over n_episodes with different random seeds and
        return mean ± std for all key metrics.  A single seed can be an
        anomalously easy or hard batch; averaging gives honest evaluation.
        """
        import math

        t0 = time.perf_counter()
        requested_orders = len(orders)
        base = {
            "model_loaded": self.rl_model is not None,
            "requested_orders": requested_orders,
            "max_supported_orders": self.rl_max_orders,
            "n_episodes": n_episodes,
        }
        if self.rl_model is None:
            return {**base, "status": "unavailable", "reason": "RL model not loaded."}
        if not orders:
            return {**base, "status": "unavailable", "reason": "No orders available for RL evaluation."}
        if self.rl_max_orders is None:
            return {**base, "status": "unavailable", "reason": "Could not infer RL observation size."}

        eval_orders = self._prepare_rl_orders(orders[:self.rl_max_orders])
        skipped_orders = max(0, requested_orders - len(eval_orders))

        # ── per-episode collectors ────────────────────────────────────────────
        ep_rewards, ep_served, ep_on_time_pct = [], [], []
        ep_late, ep_dist, ep_fuel = [], [], []
        first_ep_decisions: list[dict] = []

        for ep_idx in range(n_episodes):
            env = None
            seed = ep_idx * 17 + 42          # deterministic but varied seeds
            try:
                from delivery_env import DeliveryEnv
                import pandas as pd
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                df = pd.DataFrame(eval_orders)
                env = DummyVecEnv(
                    [lambda s=seed: DeliveryEnv(df, max_orders=self.rl_max_orders, seed=s)]
                )
                if self.vecnorm:
                    env = VecNormalize.load(self.vecnorm, env)
                    env.training   = False
                    env.norm_reward = False

                obs        = env.reset()
                ep_reward  = 0.0
                decisions  = []
                final_info = {}

                for step_idx in range(len(eval_orders)):
                    remaining_idx = list(env.envs[0].remaining_idx)
                    if not remaining_idx:
                        break
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                    raw_action   = int(action[0])
                    chosen_pos   = raw_action % len(remaining_idx)
                    selected_idx = int(remaining_idx[chosen_pos])
                    selected_order = env.envs[0].episode_orders[selected_idx]

                    obs, reward, done_arr, info = env.step(action)
                    ep_reward  += float(reward[0])
                    final_info  = info[0] or final_info
                    decisions.append({
                        "step":                step_idx + 1,
                        "raw_action":          raw_action,
                        "selected_order_index": selected_idx,
                        "selected_order_id":   selected_order.get("order_id"),
                    })
                    if bool(done_arr[0]):
                        break

                served     = len(decisions)
                late       = int(final_info.get("late_count", 0))
                on_time_p  = 100.0 * max(0, served - late) / max(served, 1)

                ep_rewards.append(ep_reward)
                ep_served.append(served)
                ep_on_time_pct.append(on_time_p)
                ep_late.append(late)
                ep_dist.append(float(final_info.get("total_dist_km", 0.0)))
                ep_fuel.append(float(final_info.get("total_fuel_L", 0.0)))
                if ep_idx == 0:
                    first_ep_decisions = decisions[:20]

            except Exception as e:
                logger.warning(f"[RL evaluate] Episode {ep_idx} failed: {e}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

        if not ep_rewards:
            return {
                **base,
                "status": "error",
                "reason": "All evaluation episodes failed.",
                "inference_time_s": round(time.perf_counter() - t0, 4),
            }

        # ── aggregate ─────────────────────────────────────────────────────────
        def _mean(v): return sum(v) / len(v)
        def _std(v, m): return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))

        n          = len(ep_rewards)
        m_reward   = _mean(ep_rewards)
        m_served   = _mean(ep_served)
        m_on_time  = _mean(ep_on_time_pct)
        m_late     = _mean(ep_late)
        m_dist     = _mean(ep_dist)
        m_fuel     = _mean(ep_fuel)
        elapsed    = time.perf_counter() - t0

        return {
            **base,
            "status":              "evaluated",
            "reason":              f"PPO policy averaged over {n} episodes (seeds vary per run).",
            "evaluated_orders":    len(eval_orders),
            "skipped_orders":      skipped_orders,
            "coverage_pct":        round(100.0 * len(eval_orders) / max(requested_orders, 1), 1),
            "episodes_run":        n,
            # ── mean metrics ──────────────────────────────────────────────────
            "orders_served":       round(m_served, 1),
            "orders_served_std":   round(_std(ep_served, m_served), 1),
            "on_time_pct":         round(m_on_time, 1),
            "on_time_std":         round(_std(ep_on_time_pct, m_on_time), 1),
            "on_time_count":       round(m_on_time * m_served / 100),
            "late_count":          round(m_late, 1),
            "late_std":            round(_std(ep_late, m_late), 1),
            "reward_score":        round(m_reward, 3),
            "reward_std":          round(_std(ep_rewards, m_reward), 3),
            "mean_reward_per_order": round(m_reward / max(m_served, 1), 3),
            "total_dist_km":       round(m_dist, 3),
            "total_fuel_L":        round(m_fuel, 3),
            "inference_time_s":    round(elapsed, 4),
            "avg_decision_time_ms": round(elapsed / max(sum(ep_served), 1) * 1000.0, 3),
            "action_count":        round(m_served),
            "action_sequence":     first_ep_decisions,
        }

    def _nearest_vehicles(self, order: dict, k: int = 2) -> list[int]:
        """Return indices of the k vehicles whose last stop is nearest to the new order."""
        dists = []
        for v_idx, route in enumerate(self.routes):
            if route:
                last = route[-1]
                d = _dist_km(last["drop_lat"], last["drop_lon"],
                             order["pickup_lat"], order["pickup_lon"])
            else:
                d = _dist_km(self.depot_lat, self.depot_lon,
                             order["pickup_lat"], order["pickup_lon"])
            dists.append((d, v_idx))
        dists.sort()
        return [v for _, v in dists[:k]]

    def compute_metrics_for_routes(self, routes: list[list[dict]]) -> dict:
        """Compute the same live metrics for an arbitrary set of routes."""
        return self._compute_live_metrics(routes=routes)

    def _compute_live_metrics(self, routes: Optional[list[list[dict]]] = None) -> dict:
        """Compute route metrics from the currently assigned stop sequence."""
        active_routes = routes if routes is not None else self.routes
        route_payload = []
        total_dist = 0.0
        total_time = 0.0
        total_fuel = 0.0
        total_co2 = 0.0
        total_orders = 0
        on_time_orders = 0
        late_orders = 0
        total_late_min = 0.0
        reward_score = 0.0

        for v_idx, route in enumerate(active_routes):
            current_lat = self.depot_lat
            current_lon = self.depot_lon
            current_time = 480.0
            route_dist = 0.0
            route_time = 0.0
            route_fuel = 0.0
            route_co2 = 0.0
            stops = []
            n_stops = len(route)

            for stop_idx, order in enumerate(route):
                traffic_level = int(order.get("Road_traffic_density", 1))
                pickup_dist = _dist_km(
                    current_lat, current_lon,
                    order["pickup_lat"], order["pickup_lon"],
                )
                pickup_eta = _eta(
                    current_lat, current_lon,
                    order["pickup_lat"], order["pickup_lon"],
                    traffic=traffic_level,
                )
                service_dist = _dist_km(
                    order["pickup_lat"], order["pickup_lon"],
                    order["drop_lat"], order["drop_lon"],
                )
                service_eta = _eta(
                    order["pickup_lat"], order["pickup_lon"],
                    order["drop_lat"], order["drop_lon"],
                    traffic=traffic_level,
                )
                leg_dist = pickup_dist + service_dist
                leg_time = pickup_eta + service_eta
                load_pct = (n_stops - stop_idx) / max(n_stops, 1)
                fuel_l = fuel_consumption(
                    leg_dist,
                    speed_kmh=SPEED_KMH,
                    load_pct=load_pct,
                    road_type="urban",
                    traffic_level=traffic_level,
                )
                co2_value = co2_kg(fuel_l)
                current_time += leg_time
                deadline = float(order.get("order_time_min", 480)) + TIME_WINDOW
                late_min = max(0.0, current_time - deadline)
                on_time = current_time <= deadline

                total_orders += 1
                on_time_orders += int(on_time)
                late_orders += int(not on_time)
                total_late_min += late_min
                reward_score -= leg_dist + 0.5 * (late_min / TIME_WINDOW) * LATE_PENALTY
                total_dist += leg_dist
                total_time += leg_time
                total_fuel += fuel_l
                total_co2 += co2_value
                route_dist += leg_dist
                route_time += leg_time
                route_fuel += fuel_l
                route_co2 += co2_value

                stops.append({
                    "order_id": order.get("order_id"),
                    "pickup_lat": order.get("pickup_lat"),
                    "pickup_lon": order.get("pickup_lon"),
                    "drop_lat": order.get("drop_lat"),
                    "drop_lon": order.get("drop_lon"),
                    "approach_km": round(pickup_dist, 3),
                    "service_km": round(service_dist, 3),
                    "distance_km": round(leg_dist, 3),
                    "eta_min": round(leg_time, 2),
                    "arrival_time_min": round(current_time, 2),
                    "deadline_min": round(deadline, 2),
                    "on_time": on_time,
                    "traffic": order.get("traffic_label", order.get("Road_traffic_density")),
                    "traffic_level": traffic_level,
                    "weather": order.get("weather_label", order.get("Weather_conditions")),
                    "fuel_L": round(fuel_l, 4),
                    "co2_kg": round(co2_value, 4),
                })

                current_lat = order["drop_lat"]
                current_lon = order["drop_lon"]

            route_payload.append({
                "vehicle_id": v_idx + 1,
                "stops": stops,
                "route_dist_km": round(route_dist, 3),
                "route_time_min": round(route_time, 2),
                "route_fuel_L": round(route_fuel, 4),
                "route_co2_kg": round(route_co2, 4),
            })

        baseline_fuel = total_dist * BASELINE_FUEL
        fuel_saved = baseline_fuel - total_fuel
        return {
            "routes_detail": route_payload,
            "total_dist_km": round(total_dist, 3),
            "total_time_min": round(total_time, 2),
            "total_fuel_L": round(total_fuel, 3),
            "total_co2_kg": round(total_co2, 3),
            "baseline_fuel_L": round(baseline_fuel, 3),
            "fuel_saved_L": round(fuel_saved, 3),
            "co2_saved_kg": round(fuel_saved * CO2_PER_LITRE, 3),
            "orders_served": total_orders,
            "on_time_count": on_time_orders,
            "on_time_pct": round(100.0 * on_time_orders / max(total_orders, 1), 1),
            "late_count": late_orders,
            "total_late_min": round(total_late_min, 2),
            "reward_score": round(reward_score, 3),
            "vehicles_used": len([route for route in route_payload if route["stops"]]),
        }

    def _event_result(self, strategy: str, t0: float) -> dict:
        elapsed = round(time.perf_counter() - t0, 3)
        metrics = self._compute_live_metrics()
        return {
            "strategy": strategy,
            "solve_time_s": elapsed,
            **metrics,
            "warm_start": dict(self.last_warm_start),
        }

    @property
    def summary(self) -> dict:
        return self._compute_live_metrics()["routes_detail"]
