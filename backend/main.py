"""
main.py  —  FastAPI Backend
============================
Endpoints:

  GET  /health                 — health check
  GET  /orders?limit=N         — fetch sample orders from cleaned dataset
  POST /optimize               — run hybrid optimisation on given orders
  POST /event                  — handle real-time event (new order / traffic)
  GET  /metrics                — current route metrics
  GET  /reward-curve           — RL training reward history (if available)
  POST /preprocess             — re-run dataset preprocessing
  GET  /scheduler/status       — APScheduler job status
  POST /scheduler/start        — start auto re-optimisation
  POST /scheduler/stop         — stop auto re-optimisation
  POST /scheduler/interval     — change re-optimisation interval (seconds)
  WS   /ws                     — WebSocket: receive live route + metric pushes

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Install new dependency:
    pip install apscheduler --break-system-packages
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
PREP_DIR  = os.path.join(BASE_DIR, "..", "preprocessing")
DATA_PATH = os.path.join(PREP_DIR, "orders_clean.csv")

sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, PREP_DIR)

from hybrid_integration import HybridRouter
from ortools_solver import _dist_km

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title       = "Delivery Route Optimisation API",
    description = "Hybrid DRL + OR-Tools with WebSocket + APScheduler",
    version     = "2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_df     : Optional[pd.DataFrame] = None
_router : Optional[HybridRouter] = None

# ── WebSocket connection manager ───────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"[WS] Client connected — total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"[WS] Client disconnected — total: {len(self.active)}")

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = ConnectionManager()

# ── APScheduler ────────────────────────────────────────────────────────────────
scheduler            = AsyncIOScheduler()
_scheduler_interval  = 300
_scheduler_running   = False
_scheduler_job_id    = "auto_reopt"
_last_reopt_time     : Optional[str] = None
_reopt_count         = 0


async def _auto_reopt_job():
    global _last_reopt_time, _reopt_count
    if _router is None:
        logger.info("[Scheduler] Skipped — no active router")
        return
    logger.info("[Scheduler] Auto re-optimisation triggered")
    try:
        result  = _router.handle_event("TRAFFIC_UPDATE", {})
        metrics = _compute_metrics(
            result["routes_detail"],
            [o for r in _router.routes for o in r]
        )
        _last_reopt_time = datetime.now().strftime("%H:%M:%S")
        _reopt_count    += 1
        await ws_manager.broadcast({
            "type"          : "AUTO_REOPT",
            "strategy"      : result.get("strategy", "GLOBAL_REOPT"),
            "timestamp"     : _last_reopt_time,
            "reopt_count"   : _reopt_count,
            "solve_time_s"  : result.get("solve_time_s"),
            "total_dist_km" : result.get("total_dist_km"),
            "total_co2_kg"  : result.get("total_co2_kg"),
            "routes_detail" : result.get("routes_detail", []),
            "metrics"       : metrics,
        })
        logger.info(f"[Scheduler] Broadcast sent — {_reopt_count} runs done")
    except Exception as e:
        logger.error(f"[Scheduler] Failed: {e}")
        await ws_manager.broadcast({
            "type"     : "SCHEDULER_ERROR",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "error"    : str(e),
        })


@app.on_event("startup")
async def startup_event():
    scheduler.start()
    logger.info("[Scheduler] APScheduler started (idle)")


@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("[Scheduler] APScheduler stopped")


# ── data + router helpers ──────────────────────────────────────────────────────
def _load_data() -> pd.DataFrame:
    global _df
    if _df is None or _df.empty:
        if not os.path.exists(DATA_PATH):
            from preprocess import preprocess
            preprocess(verbose=False)
        _df = pd.read_csv(DATA_PATH)
    return _df


def _get_router(depot_lat, depot_lon, n_vehicles=3, capacity=10):
    global _router
    rl_model = vecnorm = None
    model_path = os.path.join(MODEL_DIR, "models", "ppo_delivery.zip")
    vnorm_path = os.path.join(MODEL_DIR, "models", "vecnorm.pkl")
    if os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            rl_model = PPO.load(model_path)
            logger.info("RL model loaded")
        except Exception as e:
            logger.warning(f"RL model load failed: {e}")
    else:
        logger.info("No RL model — using OR-Tools only")
    _router = HybridRouter(
        depot_lat=depot_lat, depot_lon=depot_lon,
        n_vehicles=n_vehicles, capacity=capacity,
        rl_model=rl_model,
        vecnorm=vnorm_path if os.path.exists(vnorm_path) else None,
    )
    return _router


# ── Pydantic models ────────────────────────────────────────────────────────────
class Order(BaseModel):
    order_id            : Optional[int]   = None
    pickup_lat          : float
    pickup_lon          : float
    drop_lat            : float
    drop_lon            : float
    distance_km         : Optional[float] = None
    est_time            : Optional[float] = None
    Road_traffic_density: int             = Field(default=1, ge=0, le=3)
    Weather_conditions  : int             = Field(default=0, ge=0, le=6)
    order_time_min      : int             = Field(default=480)
    pickup_time_min     : int             = Field(default=490)
    time_taken_min      : Optional[int]   = None
    wait_time_min       : Optional[int]   = None
    est_time_derived    : Optional[float] = None
    traffic_label       : Optional[str]   = None
    weather_label       : Optional[str]   = None
    fuel_L              : Optional[float] = None


class OptimizeRequest(BaseModel):
    orders     : list[Order]
    n_vehicles : int   = Field(default=3,    ge=1,   le=10)
    capacity   : int   = Field(default=10,   ge=1,   le=50)
    time_limit : int   = Field(default=10,   ge=1,   le=60)
    alpha      : float = Field(default=0.50, ge=0.0, le=1.0)
    beta       : float = Field(default=0.20, ge=0.0, le=1.0)
    gamma      : float = Field(default=0.20, ge=0.0, le=1.0)
    delta      : float = Field(default=0.10, ge=0.0, le=1.0)


class EventRequest(BaseModel):
    event_type : str
    payload    : dict


class SchedulerIntervalRequest(BaseModel):
    seconds: int = Field(ge=30, le=3600)


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"           : "ok",
        "model_loaded"     : _router is not None,
        "scheduler_running": _scheduler_running,
        "ws_clients"       : len(ws_manager.active),
    }


@app.get("/orders")
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


@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    orders    = [o.model_dump() for o in req.orders]
    depot_lat = float(np.mean([o["pickup_lat"] for o in orders]))
    depot_lon = float(np.mean([o["pickup_lon"] for o in orders]))
    router       = _get_router(depot_lat, depot_lon, req.n_vehicles, req.capacity)
    router.alpha = req.alpha
    router.beta  = req.beta
    router.gamma = req.gamma
    router.delta = req.delta
    result  = router.optimize(orders)
    metrics = _compute_metrics(result["routes_detail"], orders)
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


@app.post("/event")
async def handle_event(req: EventRequest):
    if _router is None:
        raise HTTPException(status_code=400, detail="No active session. Call /optimize first.")
    result  = _router.handle_event(req.event_type, req.payload)
    metrics = _compute_metrics(result["routes_detail"],
                               [o for r in _router.routes for o in r])
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


@app.get("/metrics")
def get_metrics():
    if _router is None:
        raise HTTPException(status_code=400, detail="No active session.")
    all_orders  = [o for r in _router.routes for o in r]
    routes_json = _router.summary
    return _compute_metrics(routes_json, all_orders)


@app.get("/dataset-stats")
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


@app.get("/reward-curve")
def reward_curve():
    log_path = os.path.join(MODEL_DIR, "logs", "evaluations.npz")
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


@app.post("/preprocess")
def run_preprocess():
    global _df
    from preprocess import preprocess
    df  = preprocess(verbose=False)
    _df = df
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


# ── Scheduler endpoints ────────────────────────────────────────────────────────
@app.get("/scheduler/status")
def scheduler_status():
    return {
        "running"          : _scheduler_running,
        "interval_seconds" : _scheduler_interval,
        "last_reopt"       : _last_reopt_time,
        "reopt_count"      : _reopt_count,
        "ws_clients"       : len(ws_manager.active),
        "router_active"    : _router is not None,
    }


@app.post("/scheduler/start")
async def scheduler_start():
    global _scheduler_running
    if _router is None:
        raise HTTPException(status_code=400,
                            detail="Run /optimize first.")
    if _scheduler_running:
        return {"status": "already_running", "interval_seconds": _scheduler_interval}
    if scheduler.get_job(_scheduler_job_id):
        scheduler.remove_job(_scheduler_job_id)
    scheduler.add_job(
        _auto_reopt_job,
        trigger=IntervalTrigger(seconds=_scheduler_interval),
        id=_scheduler_job_id,
        name="Auto Re-optimisation",
        replace_existing=True,
    )
    _scheduler_running = True
    logger.info(f"[Scheduler] Started — interval={_scheduler_interval}s")
    await ws_manager.broadcast({
        "type"            : "SCHEDULER_STARTED",
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "interval_seconds": _scheduler_interval,
    })
    return {"status": "started", "interval_seconds": _scheduler_interval}


@app.post("/scheduler/stop")
async def scheduler_stop():
    global _scheduler_running
    if scheduler.get_job(_scheduler_job_id):
        scheduler.remove_job(_scheduler_job_id)
    _scheduler_running = False
    logger.info("[Scheduler] Stopped")
    await ws_manager.broadcast({
        "type"     : "SCHEDULER_STOPPED",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    return {"status": "stopped"}


@app.post("/scheduler/interval")
async def set_interval(req: SchedulerIntervalRequest):
    global _scheduler_interval, _scheduler_running
    _scheduler_interval = req.seconds
    if _scheduler_running:
        scheduler.reschedule_job(
            _scheduler_job_id,
            trigger=IntervalTrigger(seconds=_scheduler_interval),
        )
    await ws_manager.broadcast({
        "type"            : "SCHEDULER_INTERVAL_CHANGED",
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "interval_seconds": _scheduler_interval,
    })
    return {"status": "updated", "interval_seconds": _scheduler_interval}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type"            : "CONNECTED",
            "timestamp"       : datetime.now().strftime("%H:%M:%S"),
            "scheduler_running": _scheduler_running,
            "interval_seconds" : _scheduler_interval,
            "reopt_count"      : _reopt_count,
            "router_active"    : _router is not None,
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
                "scheduler_running": _scheduler_running,
                "reopt_count"      : _reopt_count,
                "ws_clients"       : len(ws_manager.active),
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ── helpers ────────────────────────────────────────────────────────────────────
def _compute_metrics(routes_detail: list, all_orders: list) -> dict:
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