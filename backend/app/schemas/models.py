from pydantic import BaseModel, Field
from typing import Optional

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
