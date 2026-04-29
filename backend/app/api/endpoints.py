import os
import json
import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from apscheduler.triggers.interval import IntervalTrigger

from app.core import state
from app.core.websocket import ws_manager
from app.core.scheduler import scheduler, _compute_metrics_from_state, auto_reopt_job
from app.schemas.models import OptimizeRequest, EventRequest, SchedulerIntervalRequest

router = APIRouter()

def _load_data() -> pd.DataFrame:
    if state._df is None or state._df.empty:
        data_path = os.path.join(state.PREP_DIR, "orders_clean.csv")
        if not os.path.exists(data_path):
            from preprocess import preprocess
            preprocess(verbose=False)
        state._df = pd.read_csv(data_path)
    return state._df


def _get_router(depot_lat, depot_lon, n_vehicles=3, capacity=10):
    rl_model = None
    vecnorm = None
    # Prefer best_model.zip (highest eval reward checkpoint) over the final checkpoint
    best_path  = os.path.join(state.MODEL_DIR, "models", "best_model.zip")
    final_path = os.path.join(state.MODEL_DIR, "models", "ppo_delivery.zip")
    model_path = best_path if os.path.exists(best_path) else final_path
    vnorm_path = os.path.join(state.MODEL_DIR, "models", "vecnorm.pkl")
    if os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            rl_model = PPO.load(model_path)
        except Exception:
            pass
    if state.HybridRouter:
        state._router = state.HybridRouter(
            depot_lat=depot_lat, depot_lon=depot_lon,
            n_vehicles=n_vehicles, capacity=capacity,
            rl_model=rl_model,
            vecnorm=vnorm_path if os.path.exists(vnorm_path) else None,
        )
    return state._router


def _pct_improvement(baseline: float, hybrid: float) -> float:
    if baseline in (None, 0):
        return 0.0
    return round(((baseline - hybrid) / baseline) * 100.0, 2)


def _build_model_scores(hybrid_metrics, baseline_metrics, hybrid_solve, baseline_solve, warm_start):
    baseline_dist = float(baseline_metrics.get("total_dist_km") or 0)
    hybrid_dist = float(hybrid_metrics.get("total_dist_km") or 0)
    baseline_fuel = float(baseline_metrics.get("total_fuel_L") or 0)
    hybrid_fuel = float(hybrid_metrics.get("total_fuel_L") or 0)
    baseline_reward = float(baseline_metrics.get("reward_score") or 0)
    hybrid_reward = float(hybrid_metrics.get("reward_score") or 0)
    baseline_solve = float(baseline_solve or 0)
    hybrid_solve = float(hybrid_solve or 0)

    return {
        "distance": {
            "baseline_km": round(baseline_dist, 3),
            "hybrid_km": round(hybrid_dist, 3),
            "improvement_km": round(baseline_dist - hybrid_dist, 3),
            "improvement_pct": _pct_improvement(baseline_dist, hybrid_dist),
        },
        "fuel": {
            "baseline_L": round(baseline_fuel, 3),
            "hybrid_L": round(hybrid_fuel, 3),
            "reduction_L": round(baseline_fuel - hybrid_fuel, 3),
            "reduction_pct": _pct_improvement(baseline_fuel, hybrid_fuel),
        },
        "solve_time": {
            "baseline_s": round(baseline_solve, 3),
            "hybrid_s": round(hybrid_solve, 3),
            "delta_s": round(baseline_solve - hybrid_solve, 3),
            "change_pct": _pct_improvement(baseline_solve, hybrid_solve),
        },
        "service_quality": {
            "on_time_pct": hybrid_metrics.get("on_time_pct", 0),
            "on_time_count": hybrid_metrics.get("on_time_count", 0),
            "late_order_count": hybrid_metrics.get("late_count", 0),
            "orders_served": hybrid_metrics.get("orders_served", 0),
        },
        "reward": {
            "baseline_score": round(baseline_reward, 3),
            "hybrid_score": round(hybrid_reward, 3),
            "improvement": round(hybrid_reward - baseline_reward, 3),
        },
        "warm_start": warm_start or {},
    }


def _run_baseline(router_obj, orders, n_vehicles, capacity, time_limit, alpha, beta, gamma, delta):
    from ortools_solver import solve_vrp
    baseline_result = solve_vrp(
        orders=orders, depot_lat=router_obj.depot_lat, depot_lon=router_obj.depot_lon,
        n_vehicles=n_vehicles, capacity=capacity,
        time_limit=time_limit, warm_start=None,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    )
    baseline_routes = [[orders[i] for i in route]
                       for route in baseline_result.get("routes", [])]
    baseline_metrics = router_obj.compute_metrics_for_routes(baseline_routes)
    baseline_metrics.update({
        "solve_time_s": baseline_result["solve_time_s"],
        "status": baseline_result["status"],
    })
    return baseline_result, baseline_metrics

@router.get("/health")
def health():
    return {
        "status"           : "ok",
        "model_loaded"     : state._router is not None,
        "scheduler_running": state._scheduler_running,
        "ws_clients"       : len(ws_manager.active),
    }

@router.get("/orders")
def get_orders(
    limit  : int = Query(default=30, ge=5, le=200),
    offset : int = Query(default=0, ge=0),
    traffic: Optional[int] = Query(default=None),
    weather: Optional[int] = Query(default=None),
):
    df  = _load_data()
    sub = df.copy()
    if traffic is not None:
        sub = sub[sub["Road_traffic_density"] == traffic]
    if weather is not None:
        sub = sub[sub["Weather_conditions"] == weather]
    sub = sub.iloc[offset : offset + limit]
    return {"total": len(df), "count": len(sub), "orders": sub.to_dict(orient="records")}

@router.post("/optimize")
async def optimize(req: OptimizeRequest):
    orders    = [o.model_dump() for o in req.orders]
    depot_lat = float(np.mean([o["pickup_lat"] for o in orders]))
    depot_lon = float(np.mean([o["pickup_lon"] for o in orders]))
    r = _get_router(depot_lat, depot_lon, req.n_vehicles, req.capacity)
    if r is None:
        raise HTTPException(status_code=500, detail="Model backend failed to initialize.")
    r.alpha = req.alpha
    r.beta  = req.beta
    r.gamma = req.gamma
    r.delta = req.delta

    baseline_result, baseline_metrics = _run_baseline(
        r, orders, req.n_vehicles, req.capacity, req.time_limit,
        req.alpha, req.beta, req.gamma, req.delta,
    )

    # Run hybrid (RL warm-start + OR-Tools).
    result  = r.optimize(orders)
    metrics = _compute_metrics_from_state(result["routes_detail"], orders)
    rl_model_metrics = r.evaluate_rl_policy(orders)
    model_scores = _build_model_scores(
        metrics,
        baseline_metrics,
        result.get("solve_time_s"),
        baseline_result.get("solve_time_s"),
        result.get("warm_start"),
    )

    response = {
        **result,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "model_scores": model_scores,
        "rl_model_metrics": rl_model_metrics,
        "weights": {"alpha": req.alpha, "beta": req.beta,
                    "gamma": req.gamma, "delta": req.delta},
    }
    await ws_manager.broadcast({
        "type"            : "OPTIMISE_COMPLETE",
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "routes_detail"   : result.get("routes_detail", []),
        "metrics"         : metrics,
        "baseline_metrics": baseline_metrics,
        "model_scores"    : model_scores,
        "rl_model_metrics": rl_model_metrics,
        "solve_time_s"    : result.get("solve_time_s"),
    })
    return response

@router.post("/event")
async def handle_event(req: EventRequest):
    if state._router is None:
        raise HTTPException(status_code=400, detail="No active session. Call /optimize first.")
    result  = state._router.handle_event(req.event_type, req.payload)
    all_orders = [o for r in state._router.routes for o in r]
    metrics = _compute_metrics_from_state(result["routes_detail"], all_orders)
    baseline_result, baseline_metrics = _run_baseline(
        state._router,
        all_orders,
        state._router.n_vehicles,
        state._router.capacity,
        state._router.or_time_limit,
        state._router.alpha,
        state._router.beta,
        state._router.gamma,
        state._router.delta,
    )
    model_scores = _build_model_scores(
        metrics,
        baseline_metrics,
        result.get("solve_time_s"),
        baseline_result.get("solve_time_s"),
        result.get("warm_start"),
    )
    rl_model_metrics = state._router.evaluate_rl_policy(all_orders)
    response = {
        **result,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "model_scores": model_scores,
        "rl_model_metrics": rl_model_metrics,
    }
    await ws_manager.broadcast({
        "type"            : "EVENT_RESULT",
        "event_type"      : req.event_type,
        "strategy"        : result.get("strategy"),
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "routes_detail"   : result.get("routes_detail", []),
        "metrics"         : metrics,
        "baseline_metrics": baseline_metrics,
        "model_scores"    : model_scores,
        "rl_model_metrics": rl_model_metrics,
        "solve_time_s"    : result.get("solve_time_s"),
    })
    return response

@router.get("/metrics")
def get_metrics():
    if state._router is None:
        raise HTTPException(status_code=400, detail="No active session.")
    all_orders  = [o for r in state._router.routes for o in r]
    routes_json = state._router.summary
    return _compute_metrics_from_state(routes_json, all_orders)

@router.get("/dataset-stats")
def dataset_stats():
    df = _load_data()
    return {
        "total_orders"      : len(df),
        "avg_distance_km"   : round(df["distance_km"].mean(), 2),
        "avg_time_taken_min": round(df["time_taken_min"].mean(), 2),
        "avg_wait_time_min" : round(df["wait_time_min"].mean(), 2),
        "avg_fuel_L"        : round(df["fuel_L"].mean(), 3),
        "traffic_dist"      : df["traffic_label"].value_counts().to_dict(),
        "weather_dist"      : df["weather_label"].value_counts().to_dict(),
        "depot"             : {
            "lat": round(df["depot_lat"].iloc[0], 4),
            "lon": round(df["depot_lon"].iloc[0], 4),
        },
    }

@router.get("/reward-curve")
def reward_curve():
    log_path = os.path.join(state.MODEL_DIR, "logs", "evaluations.npz")
    if not os.path.exists(log_path):
        return {
            "source" : "missing",
            "steps"  : [],
            "rewards": [],
        }
    data = np.load(log_path)
    return {
        "source" : "trained",
        "steps"  : data["timesteps"].tolist(),
        "rewards": data["results"].mean(axis=1).tolist(),
    }

@router.post("/preprocess")
def run_preprocess():
    from preprocess import preprocess
    df  = preprocess(verbose=False)
    state._df = df
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


@router.get("/scheduler/status")
def scheduler_status():
    return {
        "running"          : state._scheduler_running,
        "interval_seconds" : state._scheduler_interval,
        "last_reopt"       : state._last_reopt_time,
        "reopt_count"      : state._reopt_count,
        "ws_clients"       : len(ws_manager.active),
        "router_active"    : state._router is not None,
    }


@router.post("/scheduler/start")
async def scheduler_start():
    if state._router is None:
        raise HTTPException(status_code=400, detail="Run /optimize first.")
    if state._scheduler_running:
        return {"status": "already_running", "interval_seconds": state._scheduler_interval}
    
    if scheduler.get_job(state._scheduler_job_id):
        scheduler.remove_job(state._scheduler_job_id)
        
    scheduler.add_job(
        auto_reopt_job,
        trigger=IntervalTrigger(seconds=state._scheduler_interval),
        id=state._scheduler_job_id,
        name="Auto Re-optimisation",
        replace_existing=True,
    )
    state._scheduler_running = True
    await ws_manager.broadcast({
        "type"            : "SCHEDULER_STARTED",
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "interval_seconds": state._scheduler_interval,
    })
    return {"status": "started", "interval_seconds": state._scheduler_interval}


@router.post("/scheduler/stop")
async def scheduler_stop():
    if scheduler.get_job(state._scheduler_job_id):
        scheduler.remove_job(state._scheduler_job_id)
    state._scheduler_running = False
    await ws_manager.broadcast({
        "type"     : "SCHEDULER_STOPPED",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    return {"status": "stopped"}


@router.post("/scheduler/interval")
async def set_interval(req: SchedulerIntervalRequest):
    state._scheduler_interval = req.seconds
    if state._scheduler_running:
        scheduler.reschedule_job(
            state._scheduler_job_id,
            trigger=IntervalTrigger(seconds=state._scheduler_interval),
        )
    await ws_manager.broadcast({
        "type"            : "SCHEDULER_INTERVAL_CHANGED",
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "interval_seconds": state._scheduler_interval,
    })
    return {"status": "updated", "interval_seconds": state._scheduler_interval}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type"            : "CONNECTED",
            "timestamp"       : datetime.now().strftime("%H:%M:%S"),
            "scheduler_running": state._scheduler_running,
            "interval_seconds" : state._scheduler_interval,
            "reopt_count"      : state._reopt_count,
            "router_active"    : state._router is not None,
        })
    except Exception:
        ws_manager.disconnect(websocket)
        return
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({
                "type"            : "PING",
                "timestamp"       : datetime.now().strftime("%H:%M:%S"),
                "scheduler_running": state._scheduler_running,
                "reopt_count"      : state._reopt_count,
                "ws_clients"       : len(ws_manager.active),
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
