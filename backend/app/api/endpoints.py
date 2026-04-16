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
    model_path = os.path.join(state.MODEL_DIR, "models", "ppo_delivery.zip")
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
    result  = r.optimize(orders)
    metrics = _compute_metrics_from_state(result["routes_detail"], orders)
    response = {
        **result,
        "metrics": metrics,
        "weights": {"alpha": req.alpha, "beta": req.beta,
                    "gamma": req.gamma, "delta": req.delta},
    }
    await ws_manager.broadcast({
        "type"         : "OPTIMISE_COMPLETE",
        "timestamp"    : datetime.now().strftime("%H:%M:%S"),
        "routes_detail": result.get("routes_detail", []),
        "metrics"      : metrics,
        "solve_time_s" : result.get("solve_time_s"),
    })
    return response

@router.post("/event")
async def handle_event(req: EventRequest):
    if state._router is None:
        raise HTTPException(status_code=400, detail="No active session. Call /optimize first.")
    result  = state._router.handle_event(req.event_type, req.payload)
    metrics = _compute_metrics_from_state(result["routes_detail"],
                               [o for r in state._router.routes for o in r])
    response = {**result, "metrics": metrics}
    await ws_manager.broadcast({
        "type"         : "EVENT_RESULT",
        "event_type"   : req.event_type,
        "strategy"     : result.get("strategy"),
        "timestamp"    : datetime.now().strftime("%H:%M:%S"),
        "routes_detail": result.get("routes_detail", []),
        "metrics"      : metrics,
        "solve_time_s" : result.get("solve_time_s"),
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
            "source" : "representative",
            "steps"  : [5000, 10000, 25000, 50000, 100000, 200000],
            "rewards": [-210, -140, -90, -55, -30, -18],
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
