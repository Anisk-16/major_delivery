"""
delivery_env.py
===============
Custom OpenAI Gym environment for the delivery route optimisation MDP.

State  : [vehicle_lat, vehicle_lon, current_time_norm, n_remaining_norm,
          mean_dist_to_remaining, mean_eta_to_remaining,
          order_i_pickup_lat, order_i_pickup_lon,
          order_i_drop_lat, order_i_drop_lon,
          order_i_dist, order_i_eta,
          order_i_traffic, order_i_weather,
          order_i_wait_pct, order_i_urgency]   ← per candidate (padded to MAX_ORDERS)

Action : discrete index j ∈ {0, …, n_remaining-1}  →  serve order j next

Reward : −(travel_km + 0.5*late_penalty)  (negative cost to minimise)
"""

import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ── constants ──────────────────────────────────────────────────────────────────
MAX_ORDERS   = 30     # episode batch size (sampled from full dataset)
MAX_TIME     = 1440   # minutes in a simulation day
SPEED_KMH    = 25.0   # average vehicle speed
R_KM         = 111.0  # km per degree (normalised [0,1] grid)
FUEL_PER_KM  = 0.12   # litres/km
LATE_PENALTY = 5.0    # extra cost per late order (km-equivalent)
TIME_WINDOW  = 60     # minutes before an order is considered "late"

TRAFFIC_FACTOR = {0: 1.0, 1: 1.2, 2: 1.5, 3: 1.8}


# ── helpers ────────────────────────────────────────────────────────────────────
def _dist_km(lat1, lon1, lat2, lon2) -> float:
    dlat = (lat2 - lat1) * R_KM
    dlon = (lon2 - lon1) * R_KM
    return math.sqrt(dlat * dlat + dlon * dlon)


def _eta_min(dist_km: float, traffic: int) -> float:
    return (dist_km / SPEED_KMH) * 60.0 * TRAFFIC_FACTOR.get(traffic, 1.0)


# ── environment ────────────────────────────────────────────────────────────────
class DeliveryEnv(gym.Env):
    """
    Single-vehicle delivery environment.
    The agent selects which order to serve next from a pool of up to MAX_ORDERS.
    """

    metadata = {"render_modes": []}

    # features per order slot in the observation
    _FEATS_PER_ORDER = 10

    def __init__(self, orders_df, max_orders: int = MAX_ORDERS, seed: int = 42):
        super().__init__()
        self.df         = orders_df.reset_index(drop=True)
        self.max_orders = max_orders
        self._rng       = np.random.default_rng(seed)

        # depot = mean of all pickup coords
        self.depot_lat = float(orders_df["pickup_lat"].mean())
        self.depot_lon = float(orders_df["pickup_lon"].mean())

        # observation: 4 vehicle features + max_orders * feats_per_order
        obs_dim = 4 + max_orders * self._FEATS_PER_ORDER
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(max_orders)

        self._reset_state()

    # ── internal state ─────────────────────────────────────────────────────────
    def _reset_state(self):
        self.vehicle_lat   = self.depot_lat
        self.vehicle_lon   = self.depot_lon
        self.current_time  = 480.0          # start at 08:00
        self.total_dist    = 0.0
        self.total_fuel    = 0.0
        self.late_count    = 0
        self.done          = False
        self.episode_orders: list[dict] = []
        self.remaining_idx: list[int]   = []

    # ── gym interface ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # sample a batch of orders
        n = min(self.max_orders, len(self.df))
        sample_idx = self._rng.choice(len(self.df), size=n, replace=False)
        self.episode_orders = self.df.iloc[sample_idx].to_dict("records")
        self.remaining_idx  = list(range(len(self.episode_orders)))

        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):
        assert not self.done, "Call reset() before step()"

        # clamp action to valid remaining orders
        if len(self.remaining_idx) == 0:
            self.done = True
            return self._build_obs(), 0.0, True, False, self._info()

        action = int(action) % len(self.remaining_idx)
        order_pos = self.remaining_idx[action]
        order = self.episode_orders[order_pos]

        traffic = int(order["Road_traffic_density"])

        # travel to pickup
        d_pickup = _dist_km(self.vehicle_lat, self.vehicle_lon,
                            order["pickup_lat"], order["pickup_lon"])
        t_pickup = _eta_min(d_pickup, traffic)

        # travel to drop
        d_drop = _dist_km(order["pickup_lat"], order["pickup_lon"],
                          order["drop_lat"], order["drop_lon"])
        t_drop = _eta_min(d_drop, traffic)

        leg_dist = d_pickup + d_drop
        leg_time = t_pickup + t_drop

        # update state
        self.current_time += leg_time
        self.vehicle_lat   = order["drop_lat"]
        self.vehicle_lon   = order["drop_lon"]
        self.total_dist   += leg_dist
        self.total_fuel   += leg_dist * FUEL_PER_KM

        # late penalty: if we arrive after order_time + TIME_WINDOW
        deadline = float(order["order_time_min"]) + TIME_WINDOW
        late = max(0.0, self.current_time - deadline)
        if late > 0:
            self.late_count += 1

        # reward = negative cost
        reward = -(leg_dist + 0.5 * (late / TIME_WINDOW) * LATE_PENALTY)

        self.remaining_idx.remove(order_pos)
        if len(self.remaining_idx) == 0 or self.current_time >= MAX_TIME:
            self.done = True

        return self._build_obs(), float(reward), self.done, False, self._info()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(4 + self.max_orders * self._FEATS_PER_ORDER, dtype=np.float32)

        # global vehicle features
        obs[0] = self.vehicle_lat
        obs[1] = self.vehicle_lon
        obs[2] = self.current_time / MAX_TIME
        obs[3] = len(self.remaining_idx) / self.max_orders

        # per-order features (padded with zeros for missing slots)
        for slot, order_pos in enumerate(self.remaining_idx):
            if slot >= self.max_orders:
                break
            o = self.episode_orders[order_pos]
            traffic = int(o["Road_traffic_density"])
            d  = _dist_km(self.vehicle_lat, self.vehicle_lon,
                          o["pickup_lat"], o["pickup_lon"])
            eta = _eta_min(d, traffic)
            deadline  = float(o["order_time_min"]) + TIME_WINDOW
            urgency   = max(0.0, (self.current_time - deadline) / TIME_WINDOW)
            wait_pct  = float(o["wait_time_min"]) / 120.0  # normalised

            base = 4 + slot * self._FEATS_PER_ORDER
            obs[base + 0] = o["pickup_lat"]
            obs[base + 1] = o["pickup_lon"]
            obs[base + 2] = o["drop_lat"]
            obs[base + 3] = o["drop_lon"]
            obs[base + 4] = min(d, 50.0) / 50.0        # dist normalised
            obs[base + 5] = min(eta, 120.0) / 120.0    # eta normalised
            obs[base + 6] = traffic / 3.0
            obs[base + 7] = int(o["Weather_conditions"]) / 6.0
            obs[base + 8] = min(wait_pct, 1.0)
            obs[base + 9] = min(urgency, 1.0)

        return obs

    def _info(self) -> dict:
        return {
            "total_dist_km" : round(self.total_dist, 3),
            "total_fuel_L"  : round(self.total_fuel, 3),
            "late_count"    : self.late_count,
            "orders_served" : self.max_orders - len(self.remaining_idx),
        }

    # valid action mask (for use with MaskablePPO or manual filtering)
    def valid_actions(self) -> list[int]:
        return list(range(len(self.remaining_idx)))
