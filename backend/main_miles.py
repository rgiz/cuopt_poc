#!/usr/bin/env python3
"""
Backend (miles edition) for the Dynamic Trip Rescheduling demo.

- Expects private data in a directory (env PRIVATE_DATA_DIR, default ./data/private)
  * distance_miles_matrix.npz
  * time_minutes_matrix.npz
  * location_index.csv   (columns: name,center_id[,postcode,lat,lon])
  * driver_states.json   (optional, but recommended)

- Cost model is read from environment via ENV_COMPAT_SNIPPET.read_cost_env_defaults()
  Supported env (examples): DELAY_COST_PER_MIN, DEADHEAD_COST_PER_MILE, OUTSOURCING_PER_MILE, OVERTIME_COST_PER_MINUTE, etc.

- POST /solve payload:
  {
    "disrupted_trips": [ { "id": "...", "start_location": "...", "end_location": "...",
                           "duration_minutes": 123, "trip_miles": 148.0, ... }, ... ],
    "candidates_per_trip": { "<trip_id>": [ { "driver_id": "...", "type": "reassigned",
                           "deadhead_miles": 3.2, "overtime_minutes": 15, ... }, ... ] },
    "params": { "cost_weight": 0.5, "service_weight": 0.5 }   # optional
  }

- Returns:
  {
    "objective_value": 1234.56,
    "assignments": [ { "trip_id": "...", "type": "reassigned" | "outsourced",
                       "driver_id": "X" | null, "cost": 42.0, ... }, ... ],
    "details": { "backend": "cuopt_http-miles+overtime" | "greedy-miles+overtime", ... }
  }

Run locally:
  uvicorn backend.main_miles:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]   # /app
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter
from urllib.parse import urljoin
import requests, time, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from backend.plan_routes import create_router as create_plan_router
from opt.cuopt_model_miles import CuOptModel

# Make sure we can import src/opt/*
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# Settings router
from backend.settings_routes import router as settings_router  # type: ignore

# Cost env shim
try:
    from backend.ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore
except Exception:
    # Fallback to local file if not packaged
    sys.path.append(str(THIS_FILE.parent))
    from ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore

# cuOpt adapter: prefer *_miles; fallback to the older module name
try:
    from opt.cuopt_model_miles import CuOptModel  # type: ignore
except Exception:
    from opt.cuopt_model import CuOptModel  # type: ignore

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("PRIVATE_DATA_DIR", "./data/private")).resolve()
# If an "active" subfolder exists, use it; otherwise use BASE_DIR directly
DATASET_DIR = BASE_DIR / "active" if (BASE_DIR / "active").exists() else BASE_DIR

CUOPT_URL = os.getenv("CUOPT_URL", "http://cuopt:5000")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")

DISTANCE_FILE       = DATASET_DIR / "distance_miles_matrix.npz"
TIME_FILE           = DATASET_DIR / "time_minutes_matrix.npz"
LOC_INDEX_FILE      = DATASET_DIR / "location_index.csv"
DRIVER_STATES_FILE  = DATASET_DIR / "driver_states.json"
# DATASET_DIR = PRIVATE_DATA_DIR / os.getenv("DATASET_ID", "active")
SLA_FILE = DATASET_DIR / "sla_windows.json"

# ------------------------------------------------------------------------------
# App (create FIRST, then add routes)
# ------------------------------------------------------------------------------
app = FastAPI(title="Dynamic Trip Rescheduling (cuOpt, miles)")
admin = APIRouter(prefix="/admin", tags=["Admin"])

allow_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",")] if CORS_ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
plan_router = create_plan_router(lambda: DATA, lambda: COST_CONFIG, lambda: CUOPT_URL)  # NEW
app.include_router(plan_router)

# ------------------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------------------
def load_sla_windows() -> Dict[int, Dict[str, int]]:
    """
    Expected schema (keys can be strings or ints):
    {
      "1": {"depart_after_slack_min": 30,  "arrive_before_slack_min": 30},
      "2": {"depart_after_slack_min": 90,  "arrive_before_slack_min": 90},
      "3": {"depart_after_slack_min": 120, "arrive_before_slack_min": 120},
      "4": {"depart_after_slack_min": 240, "arrive_before_slack_min": 240},
      "5": {"depart_after_slack_min": 480, "arrive_before_slack_min": 480}
    }
    """
    defaults = {
        1: {"depart_after_slack_min": 30,  "arrive_before_slack_min": 30},
        2: {"depart_after_slack_min": 90,  "arrive_before_slack_min": 90},
        3: {"depart_after_slack_min": 120, "arrive_before_slack_min": 120},
        4: {"depart_after_slack_min": 240, "arrive_before_slack_min": 240},
        5: {"depart_after_slack_min": 480, "arrive_before_slack_min": 480},
    }
    try:
        if SLA_FILE.exists():
            raw = json.loads(SLA_FILE.read_text(encoding="utf-8"))
            out = {}
            for k, v in (raw or {}).items():
                pk = int(k)
                out[pk] = {
                    "depart_after_slack_min": int(v.get("depart_after_slack_min", defaults.get(pk, {}).get("depart_after_slack_min", 120))),
                    "arrive_before_slack_min": int(v.get("arrive_before_slack_min", defaults.get(pk, {}).get("arrive_before_slack_min", 120))),
                }
            # ensure all 1..5 exist
            for pk, dv in defaults.items():
                out.setdefault(pk, dv)
            return out
    except Exception as e:
        print(f"[startup] WARN: failed to read sla_windows.json: {e}", flush=True)
    return defaults

def _load_npz_any(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    z = np.load(path, allow_pickle=False)
    if isinstance(z, np.lib.npyio.NpzFile):
        for k in ("matrix", "arr", "arr_0"):
            if k in z:
                return z[k]
        # fallback to first array
        for k in z.files:
            return z[k]
        raise ValueError(f"NPZ {path} contains no arrays.")
    return z  # npy array

def load_private_data() -> Dict[str, Any]:
    dist = _load_npz_any(DISTANCE_FILE)
    tmat = _load_npz_any(TIME_FILE)

    if not LOC_INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing file: {LOC_INDEX_FILE}")
    li = pd.read_csv(LOC_INDEX_FILE)
    if not {"name", "center_id"}.issubset(li.columns):
        raise ValueError("location_index.csv must have columns: name,center_id")

    li = li.copy()
    li["center_id"] = li["center_id"].astype(int)
    n = int(li["center_id"].max()) + 1

    if dist.shape != (n, n):
        raise ValueError(f"distance_miles_matrix shape {dist.shape} incompatible with location_index size {n}")
    if tmat.shape != (n, n):
        raise ValueError(f"time_minutes_matrix shape {tmat.shape} incompatible with location_index size {n}")

    location_to_index = dict(zip(li["name"], li["center_id"]))

    # Driver states (optional)
    if DRIVER_STATES_FILE.exists():
        try:
            driver_states = json.loads(DRIVER_STATES_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to read driver_states.json: {e}")
    else:
        driver_states = {}

    return {
        "distance": dist,
        "time": tmat,
        "location_to_index": location_to_index,
        "driver_states": driver_states,
        "location_index_size": n,
    }

# ------------------------------------------------------------------------------
# SELF TEST ENDPOINT
# ------------------------------------------------------------------------------

def _safe_json(resp):
    """Return (is_json, data_or_fallback). Never raises."""
    try:
        return True, resp.json()
    except Exception:
        return False, {"status_code": resp.status_code, "text": (resp.text or "")[:500]}

def _ok(payload, status=200):
    return JSONResponse(status_code=status, content={"ok": True, **payload})

def _fail(step, base, extra=None, status=500):
    out = {"ok": False, "step": step, "base": base}
    if extra:
        out.update(extra)
    return JSONResponse(status_code=status, content=out)

@app.get("/admin/cuopt_selftest")
def cuopt_selftest():
    
    ok = False
    step = "start"
    meta = {}
    try:
        model = CuOptModel(
            driver_states=DATA["driver_states"],
            distance_miles_matrix=DATA["distance"],
            time_minutes_matrix=DATA["time"],
            location_to_index=DATA["location_to_index"],
            cost_config=COST_CONFIG,
            server_url=CUOPT_URL,
            max_solve_time_seconds=10,
        )
        ping = requests.get(urljoin(CUOPT_URL.rstrip('/')+'/','')).json()
        step = "submit"
        req_id = model._submit_request({"ping":"ok"})  # harmless payload
        step = "poll"
        res = model._poll_result(req_id)
        ok = True
        return {"ok": ok, "step": step, "ping": ping, "req_id": req_id, "result_keys": list(res.keys())}
    except Exception as e:
        return {"ok": ok, "step": step if step else "error", "error": str(e), **meta}

app.include_router(admin)

# ------------------------------------------------------------------------------
# Global state + lifecycle
# ------------------------------------------------------------------------------
DATA: Optional[Dict[str, Any]] = None
COST_CONFIG = read_cost_env_defaults()

SLA_WINDOWS = load_sla_windows()

@app.on_event("startup")
def _startup_try_load():
    global DATA
    try:
        DATA = load_private_data()
        print(f"[startup] Loaded private data OK: "
              f"D={DATA['distance'].shape}, T={DATA['time'].shape}, "
              f"locations={DATA['location_index_size']}", flush=True)
    except Exception as e:
        print(f"[startup] WARN: private data not loaded: {e}", flush=True)
    # make SLA available to other routers/endpoints
    app.state.sla_windows = SLA_WINDOWS

def reload_private_data() -> Dict[str, Any]:
    global DATA, COST_CONFIG
    DATA = load_private_data()
    COST_CONFIG = read_cost_env_defaults()
    return {
        "distance_shape": list(DATA["distance"].shape),
        "time_shape": list(DATA["time"].shape),
        "locations": DATA["location_index_size"],
        "has_driver_states": bool(DATA["driver_states"]),
        "cost_keys": list(COST_CONFIG.keys()),
    }

@app.post("/admin/reload")
def admin_reload():
    info = reload_private_data()
    global SLA_WINDOWS
    SLA_WINDOWS = load_sla_windows()
    app.state.sla_windows = SLA_WINDOWS
    return {"status": "ok", "reloaded": {**info, "sla_keys": sorted(SLA_WINDOWS.keys())}}

# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------
class Candidate(BaseModel):
    candidate_id: Optional[str] = None
    driver_id: Optional[str] = None
    type: str = Field(..., description="'reassigned' or 'outsourced'")
    deadhead_miles: Optional[float] = 0.0
    deadhead_minutes: Optional[float] = 0.0
    delay_minutes: Optional[float] = 0.0
    overtime_minutes: Optional[float] = 0.0
    miles_delta: Optional[float] = 0.0
    uses_emergency_rest: Optional[bool] = False
    trip_miles: Optional[float] = None
    trip_minutes: Optional[float] = None
    duration_minutes: Optional[float] = None  # for outsource fallback if trip_minutes omitted
    route_id: Optional[str] = None  # optional alias

class Trip(BaseModel):
    id: str
    start_location: str
    end_location: str
    duration_minutes: Optional[float] = 0.0
    trip_miles: Optional[float] = None

class SolveParams(BaseModel):
    cost_weight: Optional[float] = 0.5
    service_weight: Optional[float] = 0.5

class SolveRequest(BaseModel):
    disrupted_trips: List[Trip]
    candidates_per_trip: Dict[str, List[Candidate]]
    params: Optional[SolveParams] = SolveParams()

class Assignment(BaseModel):
    trip_id: str
    type: str
    driver_id: Optional[str]
    candidate_id: Optional[str] = None
    delay_minutes: Optional[float] = 0.0
    uses_emergency_rest: Optional[bool] = False
    deadhead_miles: Optional[float] = 0.0
    overtime_minutes: Optional[float] = 0.0
    miles_delta: Optional[float] = 0.0
    cost: float

class SolveResponse(BaseModel):
    objective_value: float
    assignments: List[Assignment]
    details: Dict[str, Any]

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    base = {
        "status": "ok" if DATA is not None else "needs_data",
        "private_data_dir": str(BASE_DIR),
        "dataset_dir": str(DATASET_DIR),
        "cuopt_url": CUOPT_URL,
    }
    if DATA is None:
        return {**base, "message": "Private data not loaded. Upload/build and POST /admin/reload."}
    return {
        **base,
        "distance_shape": list(DATA["distance"].shape),
        "time_shape": list(DATA["time"].shape),
        "locations": DATA["location_index_size"],
        "has_driver_states": bool(DATA["driver_states"]),
        "has_sla_windows": bool(app.state.sla_windows),
    }

@app.get("/config")
def config():
    return {
        "cost_config": COST_CONFIG,
        "cors_allow_origins": allow_origins,
        "private_data_dir": str(BASE_DIR),
    }

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if DATA is None:
        raise HTTPException(status_code=503, detail="Private data not loaded. Upload/build and POST /admin/reload.")
    if not req.disrupted_trips:
        raise HTTPException(status_code=400, detail="No disrupted_trips provided.")

    model = CuOptModel(
        driver_states=DATA["driver_states"],
        distance_miles_matrix=DATA["distance"],
        time_minutes_matrix=DATA["time"],
        location_to_index=DATA["location_to_index"],
        cost_config=COST_CONFIG,
        server_url=CUOPT_URL,
    )

    trips = [t.dict() for t in req.disrupted_trips]
    cands = {k: [c.dict() for c in v] for k, v in req.candidates_per_trip.items()}
    params = (req.params or SolveParams()).dict()

    sol = model.solve(trips, cands, params)

    return SolveResponse(
        objective_value=float(sol.objective_value),
        assignments=[Assignment(**a) for a in sol.assignments],
        details=sol.details,
    )

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main_miles:app", host="0.0.0.0", port=port, reload=False)
