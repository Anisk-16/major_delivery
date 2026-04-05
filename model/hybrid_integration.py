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
import math
import time
import logging
import numpy as np
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ortools_solver import (solve_vrp, _dist_km, TRAFFIC_FACTOR as _TF,
                            fuel_consumption, co2_kg, co2_saved_vs_baseline,
                            BASELINE_FUEL, CO2_PER_LITRE)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SPEED_KMH = 25.0
GREEDY_THRESHOLD = 5.0   # km — accept greedy insertion if cost delta ≤ this


def _eta(lat1, lon1, lat2, lon2, traffic=1) -> float:
    d = _dist_km(lat1, lon1, lat2, lon2)
    return (d / SPEED_KMH) * 60.0 * _TF.get(traffic, 1.0)


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
        for pos in range(1, len(stops) + 1):
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

        return {**result, "routes_detail": self._routes_to_json()}

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
            return None
        try:
            from delivery_env import DeliveryEnv
            import pandas as pd
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            df  = pd.DataFrame(orders)
            env = DummyVecEnv([lambda: DeliveryEnv(df, max_orders=len(orders))])
            if self.vecnorm:
                env = VecNormalize.load(self.vecnorm, env)
                env.training = False

            obs    = env.reset()
            perm   = []
            for _ in range(len(orders)):
                action, _ = self.rl_model.predict(obs, deterministic=True)
                perm.append(int(action[0]))
                obs, _, done, _ = env.step(action)
                if done[0]:
                    break
            return perm
        except Exception as e:
            logger.warning(f"[RL suggest] Failed ({e}), no warm-start")
            return None

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

    def _routes_to_json(self) -> list[list[dict]]:
        """Serialisable route structure."""
        result = []
        for v_idx, route in enumerate(self.routes):
            stops = []
            for o in route:
                d_km     = o.get("distance_km") or 0
                t_lvl    = int(o.get("Road_traffic_density", 1))
                f_L      = fuel_consumption(d_km, speed_kmh=25.0,
                                            load_pct=0.5, road_type="urban",
                                            traffic_level=t_lvl)
                c_kg     = co2_kg(f_L)
                stops.append({
                    "order_id"   : o.get("order_id"),
                    "pickup_lat" : o.get("pickup_lat"),
                    "pickup_lon" : o.get("pickup_lon"),
                    "drop_lat"   : o.get("drop_lat"),
                    "drop_lon"   : o.get("drop_lon"),
                    "distance_km": d_km,
                    "eta_min"    : o.get("est_time_derived", o.get("est_time")),
                    "traffic"    : o.get("traffic_label", o.get("Road_traffic_density")),
                    "weather"    : o.get("weather_label", o.get("Weather_conditions")),
                    "fuel_L"     : round(f_L, 4),
                    "co2_kg"     : round(c_kg, 4),
                })
            result.append({"vehicle_id": v_idx + 1, "stops": stops})
        return result

    def _event_result(self, strategy: str, t0: float) -> dict:
        elapsed = round(time.perf_counter() - t0, 3)
        total_dist = 0.0
        total_fuel = 0.0
        total_co2  = 0.0
        for r in self.routes:
            for o in r:
                d    = o.get("distance_km") or 0
                t_l  = int(o.get("Road_traffic_density", 1))
                f    = fuel_consumption(d, speed_kmh=25.0, load_pct=0.5,
                                        road_type="urban", traffic_level=t_l)
                total_dist += d
                total_fuel += f
                total_co2  += co2_kg(f)
        saved_co2 = co2_saved_vs_baseline(total_fuel, total_dist)
        return {
            "strategy"       : strategy,
            "solve_time_s"   : elapsed,
            "total_dist_km"  : round(total_dist, 3),
            "total_fuel_L"   : round(total_fuel, 3),
            "total_co2_kg"   : round(total_co2, 3),
            "co2_saved_kg"   : saved_co2,
            "routes_detail"  : self._routes_to_json(),
        }

    @property
    def summary(self) -> dict:
        return self._event_result("CURRENT", time.perf_counter())["routes_detail"]
