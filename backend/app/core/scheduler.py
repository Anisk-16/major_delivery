import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core import state
from app.core.websocket import ws_manager

logger = logging.getLogger("uvicorn.error")

scheduler = AsyncIOScheduler()

def _compute_metrics_from_state(routes_detail, all_orders):
    # Quick utility to fetch metrics
    try:
        from ortools_solver import fuel_consumption, co2_kg, co2_saved_vs_baseline, BASELINE_FUEL
        total_dist = total_time = total_fuel = total_co2 = 0.0
        on_time = total_orders = 0
        n_vehicles = len([v for v in routes_detail if v.get("stops")])
        for vehicle in routes_detail:
            stops   = vehicle.get("stops", [])
            n_stops = len(stops)
            total_orders += n_stops
            for si, stop in enumerate(stops):
                dist = stop.get("distance_km") or 0
                eta  = stop.get("eta_min") or 0
                total_dist += dist
                total_time += eta
                on_time    += 1
                traffic_lvl = int(stop.get("Road_traffic_density", 1))
                load_pct    = (n_stops - si) / max(n_stops, 1)
                f = fuel_consumption(dist, speed_kmh=25.0,
                                     load_pct=load_pct, road_type="urban",
                                     traffic_level=traffic_lvl)
                total_fuel += f
                total_co2  += co2_kg(f)
        on_time_pct   = round(100 * on_time / max(total_orders, 1), 1)
        baseline_fuel = round(total_dist * BASELINE_FUEL, 3)
        saved_fuel    = round(baseline_fuel - total_fuel, 3)
        saved_co2     = round(saved_fuel * 2.31, 3)
        return {
            "total_dist_km"  : round(total_dist, 2),
            "total_time_min" : round(total_time, 1),
            "total_fuel_L"   : round(total_fuel, 3),
            "total_co2_kg"   : round(total_co2, 3),
            "baseline_fuel_L": baseline_fuel,
            "fuel_saved_L"   : saved_fuel,
            "co2_saved_kg"   : saved_co2,
            "orders_served"  : total_orders,
            "on_time_pct"    : on_time_pct,
            "vehicles_used"  : n_vehicles,
        }
    except Exception as e:
        logger.error(f"[Metrics Error]: {e}")
        return {}


async def auto_reopt_job():
    if state._router is None:
        logger.info("[Scheduler] Skipped — no active router")
        return
    logger.info("[Scheduler] Auto re-optimisation triggered")
    try:
        result  = state._router.handle_event("TRAFFIC_UPDATE", {})
        metrics = _compute_metrics_from_state(
            result["routes_detail"],
            [o for r in state._router.routes for o in r]
        )
        state._last_reopt_time = datetime.now().strftime("%H:%M:%S")
        state._reopt_count    += 1
        await ws_manager.broadcast({
            "type"          : "AUTO_REOPT",
            "strategy"      : result.get("strategy", "GLOBAL_REOPT"),
            "timestamp"     : state._last_reopt_time,
            "reopt_count"   : state._reopt_count,
            "solve_time_s"  : result.get("solve_time_s"),
            "total_dist_km" : result.get("total_dist_km"),
            "total_co2_kg"  : result.get("total_co2_kg"),
            "routes_detail" : result.get("routes_detail", []),
            "metrics"       : metrics,
        })
        logger.info(f"[Scheduler] Broadcast sent — {state._reopt_count} runs done")
    except Exception as e:
        logger.error(f"[Scheduler] Failed: {e}")
        await ws_manager.broadcast({
            "type"     : "SCHEDULER_ERROR",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "error"    : str(e),
        })
