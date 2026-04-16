import os
import sys
from typing import Optional
import pandas as pd
from .websocket import ws_manager

# Insert the model and prep directories into path
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
PREP_DIR  = os.path.join(BASE_DIR, "..", "preprocessing")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
if PREP_DIR not in sys.path:
    sys.path.insert(0, PREP_DIR)

# Proceed with imports that rely on sys.path modifications
try:
    from hybrid_integration import HybridRouter
except ImportError:
    HybridRouter = None

_df: Optional[pd.DataFrame] = None
_router: Optional['HybridRouter'] = None

# Scheduler State
_scheduler_interval = 300
_scheduler_running = False
_scheduler_job_id = "auto_reopt"
_last_reopt_time: Optional[str] = None
_reopt_count = 0
