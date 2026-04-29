"""
ortools_solver.py
=================
Capacitated VRP solver using Google OR-Tools.
Includes carbon-aware and fuel-aware multi-objective routing.

  solve_vrp(orders, n_vehicles, capacity, time_limit_sec, warm_start,
            alpha, beta, gamma, delta)
      → routes: list[list[int]]   (order indices per vehicle)
      → metrics: dict  (includes fuel_L, co2_kg, co2_saved_kg)

Multi-objective arc cost:
    cost = α × distance + β × time + γ × fuel + δ × CO2

Carbon model:
    fuel_L   = distance_km × base_rate × speed_factor × load_factor × road_factor
    co2_kg   = fuel_L × CO2_PER_LITRE (2.31 kg/L for petrol)
    savings  = baseline_co2 - actual_co2
"""

import math
import time
from typing import Optional
from delivery_env import TRAFFIC_FACTOR

import numpy as np

# ── Carbon & Fuel Constants ────────────────────────────────────────────────────
BASE_FUEL_RATE  = 0.08        # L/km at ideal conditions (unloaded, 60 km/h)
CO2_PER_LITRE   = 2.31        # kg CO2 per litre of petrol (IPCC standard)
IDEAL_SPEED     = 60.0        # km/h for baseline fuel rate
BASELINE_FUEL   = 0.18        # L/km unoptimised urban delivery baseline (fully loaded, no route planning)

# Road type fuel multipliers
ROAD_FACTOR     = {"urban": 1.30, "highway": 0.90, "rural": 1.10}

# Multi-objective weights (can be overridden per call)
DEFAULT_ALPHA   = 0.50        # weight for distance
DEFAULT_BETA    = 0.20        # weight for time
DEFAULT_GAMMA   = 0.20        # weight for fuel
DEFAULT_DELTA   = 0.10        # weight for CO2


def fuel_consumption(distance_km: float,
                     speed_kmh: float    = 25.0,
                     load_pct: float     = 0.5,
                     road_type: str      = "urban",
                     traffic_level: int  = 1) -> float:
    """
    Physics-based fuel consumption model (litres).

    Parameters
    ----------
    distance_km  : leg distance in km
    speed_kmh    : average speed on this leg (km/h)
    load_pct     : vehicle load as fraction of capacity 0→1
    road_type    : 'urban' | 'highway' | 'rural'
    traffic_level: 0=Low, 1=Medium, 2=High, 3=Very_High

    Returns
    -------
    fuel_L : float
    """
    # Speed penalty — fuel increases when away from ideal 60 km/h
    speed_factor = 1.0 + abs(speed_kmh - IDEAL_SPEED) * 0.005

    # Load penalty — heavier vehicle burns more fuel
    load_factor  = 1.0 + load_pct * 0.30

    # Road type multiplier
    rf = ROAD_FACTOR.get(road_type, 1.30)

    # Traffic stop-start penalty
    traffic_penalty = {0: 1.0, 1: 1.08, 2: 1.18, 3: 1.30}.get(traffic_level, 1.0)

    return distance_km * BASE_FUEL_RATE * speed_factor * load_factor * rf * traffic_penalty


def co2_kg(fuel_L: float) -> float:
    """Convert fuel consumption (litres) to CO2 emissions (kg)."""
    return fuel_L * CO2_PER_LITRE


def co2_saved_vs_baseline(actual_fuel_L: float, distance_km: float) -> float:
    """Carbon savings (kg CO2) vs simple 0.12 L/km flat baseline."""
    baseline_fuel = distance_km * BASELINE_FUEL
    saved_fuel    = baseline_fuel - actual_fuel_L
    return round(saved_fuel * CO2_PER_LITRE, 4)

# OR-Tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ── distance helper (normalised [0,1] coords → km ≈ × 111) ────────────────────
R_KM = 111.0

def _dist_km(lat1, lon1, lat2, lon2) -> float:
    dlat = (lat2 - lat1) * R_KM
    dlon = (lon2 - lon1) * R_KM
    return math.sqrt(dlat * dlat + dlon * dlon)


def _build_distance_matrix(orders: list[dict], depot_lat: float, depot_lon: float
                            ) -> list[list[int]]:
    """
    Build integer distance matrix (metres) for OR-Tools.
    Node 0 = depot, nodes 1..N = orders (each order = one stop combining pickup+drop).
    """
    lats = [depot_lat] + [o["drop_lat"] for o in orders]
    lons = [depot_lon] + [o["drop_lon"] for o in orders]
    n = len(lats)
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            d_km = _dist_km(lats[i], lons[i], lats[j], lons[j])
            row.append(int(d_km * 1000))   # metres as integer
        mat.append(row)
    return mat


def _build_time_matrix(orders: list[dict], depot_lat: float, depot_lon: float
                       ) -> list[list[int]]:
    """ETA matrix in minutes (integer) for time-window constraints."""
    from delivery_env import TRAFFIC_FACTOR
    lats     = [depot_lat] + [o["drop_lat"] for o in orders]
    lons     = [depot_lon] + [o["drop_lon"] for o in orders]
    traffic  = [1] + [int(o.get("Road_traffic_density", 1)) for o in orders]
    SPEED = 40.0  # km/h — must match delivery_env.py

    n = len(lats)
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            d_km = _dist_km(lats[i], lons[i], lats[j], lons[j])
            tf   = TRAFFIC_FACTOR.get(traffic[j], 1.0)
            eta  = (d_km / SPEED) * 60.0 * tf
            row.append(int(eta))
        mat.append(row)
    return mat


# ── main solver ────────────────────────────────────────────────────────────────
def solve_vrp(
    orders       : list[dict],
    depot_lat    : float,
    depot_lon    : float,
    n_vehicles   : int         = 3,
    capacity     : int         = 10,
    time_limit   : int         = 10,
    warm_start   : Optional[list[int]] = None,
    # Multi-objective weights
    alpha        : float       = DEFAULT_ALPHA,   # distance weight
    beta         : float       = DEFAULT_BETA,    # time weight
    gamma        : float       = DEFAULT_GAMMA,   # fuel weight
    delta        : float       = DEFAULT_DELTA,   # CO2 weight
) -> dict:
    """
    Carbon-aware, fuel-aware multi-objective VRP solver.

    Arc cost = α×distance + β×time + γ×fuel + δ×CO2

    Returns
    -------
    {
        "routes"        : [[order_idx, ...], ...],
        "total_dist_km" : float,
        "total_time_min": float,
        "total_fuel_L"  : float,
        "total_co2_kg"  : float,
        "co2_saved_kg"  : float,
        "solve_time_s"  : float,
        "status"        : "OPTIMAL" | "FEASIBLE" | "FAILED",
        "weights"       : {alpha, beta, gamma, delta},
    }
    """
    if not orders:
        return {"routes": [], "total_dist_km": 0, "total_time_min": 0,
                "solve_time_s": 0, "status": "EMPTY"}

    t0         = time.perf_counter()
    n          = len(orders)
    dist_mat   = _build_distance_matrix(orders, depot_lat, depot_lon)
    time_mat   = _build_time_matrix(orders, depot_lat, depot_lon)

    # ── OR-Tools routing model ─────────────────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(n + 1, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # ── pre-compute per-arc fuel & CO2 for multi-objective cost ───────────────
    lats_all  = [depot_lat] + [o["drop_lat"] for o in orders]
    lons_all  = [depot_lon] + [o["drop_lon"] for o in orders]
    traffic_all = [1] + [int(o.get("Road_traffic_density", 1)) for o in orders]

    fuel_mat  = []   # L per arc
    co2_mat   = []   # kg CO2 per arc
    for i in range(n + 1):
        fuel_row = []
        co2_row  = []
        for j in range(n + 1):
            d_km    = _dist_km(lats_all[i], lons_all[i], lats_all[j], lons_all[j])
            t_lvl   = traffic_all[j]
            load_p  = 0.5   # assume half-loaded on average per leg
            f_L     = fuel_consumption(d_km, speed_kmh=40.0,
                                       load_pct=load_p, road_type="urban",
                                       traffic_level=t_lvl)
            fuel_row.append(f_L)
            co2_row.append(co2_kg(f_L))
        fuel_mat.append(fuel_row)
        co2_mat.append(co2_row)

    # ── multi-objective arc cost: α×dist + β×time + γ×fuel + δ×CO2 ────────────
    # Scale all to same order of magnitude (all → "distance units" ~km)
    # fuel: 1 L ≈ 8.33 km at baseline; CO2: 1 kg ≈ 3.6 km equivalent
    FUEL_TO_KM = 1.0 / BASELINE_FUEL        # 1 L ÷ 0.12 L/km = 8.33 km
    CO2_TO_KM  = 1.0 / (BASELINE_FUEL * CO2_PER_LITRE)  # 1 kg CO2 → km equiv

    def multi_obj_cost_m(i, j):
        d_km  = dist_mat[i][j] / 1000.0
        t_min = time_mat[i][j]
        f_L   = fuel_mat[i][j]
        c_kg  = co2_mat[i][j]
        # All components normalised to km-equivalent, then weighted
        cost_km = (alpha * d_km
                 + beta  * (t_min / 60.0) * 40.0    # time → km at 40 km/h
                 + gamma * f_L * FUEL_TO_KM
                 + delta * c_kg * CO2_TO_KM)
        return int(cost_km * 1000)   # back to integer metres for OR-Tools

    # distance callback (raw, for reporting)
    def dist_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return dist_mat[i][j]

    # multi-objective cost callback
    def cost_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return multi_obj_cost_m(i, j)

    cost_cb_idx = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_cb_idx)
    dist_cb_idx = routing.RegisterTransitCallback(dist_callback)

    # time callback + time dimension
    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return time_mat[i][j]

    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_cb_idx,
        slack_max   = 30,    # wait up to 30 min at a node
        capacity    = 960,   # 16 hours total
        fix_start_cumul_to_zero = True,
        name        = "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    # soft time-window per order: [order_time, order_time + 60 min]
    for order_pos, o in enumerate(orders):
        node      = order_pos + 1  # node 0 = depot
        idx       = manager.NodeToIndex(node)
        t_open    = int(o.get("order_time_min", 480))
        t_close   = t_open + 60
        time_dim.CumulVar(idx).SetRange(0, t_close)

    # capacity dimension
    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return 0 if node == 0 else 1   # each order = 1 unit demand

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        slack_max         = 0,
        vehicle_capacities= [capacity] * n_vehicles,
        fix_start_cumul_to_zero = True,
        name              = "Capacity",
    )

    # ── warm start from RL ─────────────────────────────────────────────────────
    if warm_start and len(warm_start) <= n:
        # Distribute RL sequence across vehicles round-robin
        initial_routes = [[] for _ in range(n_vehicles)]
        for i, order_idx in enumerate(warm_start):
            initial_routes[i % n_vehicles].append(order_idx + 1)   # +1 for depot offset
        routing.CloseModelWithParameters(
            pywrapcp.DefaultRoutingSearchParameters()
        )
        assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)
        if assignment:
            routing.RoutesToAssignment(initial_routes, True, True, assignment)

    # ── search parameters ──────────────────────────────────────────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(time_limit)
    search_params.log_search = False

    # ── solve ──────────────────────────────────────────────────────────────────
    solution = routing.SolveWithParameters(search_params)
    solve_time = time.perf_counter() - t0

    if not solution:
        return {"routes": [], "total_dist_km": 0, "total_time_min": 0,
                "solve_time_s": round(solve_time, 2), "status": "FAILED"}

    # ── extract routes ─────────────────────────────────────────────────────────
    routes         = []
    total_dist_m   = 0
    total_time_min = 0
    total_fuel_L   = 0.0
    total_co2      = 0.0

    for v in range(n_vehicles):
        route = []
        idx   = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node - 1)
            next_idx = solution.Value(routing.NextVar(idx))
            ni = manager.IndexToNode(idx)
            nj = manager.IndexToNode(next_idx)
            total_dist_m   += dist_mat[ni][nj]
            total_time_min += time_mat[ni][nj]
            total_fuel_L   += fuel_mat[ni][nj]
            total_co2      += co2_mat[ni][nj]
            idx = next_idx
        if route:
            routes.append(route)

    total_dist_km = round(total_dist_m / 1000.0, 3)
    status = ("OPTIMAL" if routing.status() == 1 else "FEASIBLE")

    return {
        "routes"         : routes,
        "total_dist_km"  : total_dist_km,
        "total_time_min" : total_time_min,
        "total_fuel_L"   : round(total_fuel_L, 3),
        "total_co2_kg"   : round(total_co2, 3),
        "co2_saved_kg"   : co2_saved_vs_baseline(total_fuel_L, total_dist_km),
        "solve_time_s"   : round(solve_time, 2),
        "status"         : status,
        "weights"        : {"alpha": alpha, "beta": beta,
                            "gamma": gamma, "delta": delta},
    }
