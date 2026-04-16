"""
main.py  —  FastAPI Backend (Modularized)
=========================================
Entry point for the Delivery Route Optimizer.
Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.core.scheduler import scheduler

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

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    logger.info("[Scheduler] APScheduler started (idle)")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("[Scheduler] APScheduler stopped")