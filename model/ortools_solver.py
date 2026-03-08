"""
ortools_solver.py
=================
Capacitated VRP solver using Google OR-Tools.

  solve_vrp(orders, n_vehicles, capacity, time_limit_sec, warm_start)
      → routes: list[list[int]]   (order indices per vehicle)
      → metrics: dict

The solver accepts an optional RL warm-start (permutation of order indices)
that is injected as an initial solution hint to speed up the search.
"""

import math
import time
from typing import Optional

import numpy as np

# OR-Tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ── distance helper (normalised [0,1] coords → km ≈ × 111) ────────────────────
R_KM = 111.0

TRAFFIC_FACTOR = {
    0: 1.0,
    1: 1.15,
    2: 1.35,
    3: 1.6
}

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
    SPEED = 25.0  # km/h

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
    capacity     : int         = 10,      # orders per vehicle
    time_limit   : int         = 10,      # seconds
    warm_start   : Optional[list[int]] = None,   # RL-suggested permutation
) -> dict:
    """
    Returns
    -------
    {
        "routes"      : [[order_idx, ...], ...],   # one list per vehicle
        "total_dist_km": float,
        "total_time_min": float,
        "solve_time_s" : float,
        "status"       : "OPTIMAL" | "FEASIBLE" | "FAILED",
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

    # distance callback
    def dist_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return dist_mat[i][j]

    dist_cb_idx = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

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

    for v in range(n_vehicles):
        route   = []
        idx     = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node - 1)    # back to 0-indexed order list
            next_idx = solution.Value(routing.NextVar(idx))
            total_dist_m   += dist_mat[manager.IndexToNode(idx)][manager.IndexToNode(next_idx)]
            total_time_min += time_mat[manager.IndexToNode(idx)][manager.IndexToNode(next_idx)]
            idx = next_idx
        if route:
            routes.append(route)

    status = ("OPTIMAL"  if routing.status() == 1 else "FEASIBLE")

    return {
        "routes"         : routes,
        "total_dist_km"  : round(total_dist_m / 1000.0, 3),
        "total_time_min" : total_time_min,
        "solve_time_s"   : round(solve_time, 2),
        "status"         : status,
    }
