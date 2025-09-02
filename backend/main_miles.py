#!/usr/bin/env python3
"""
Backend (miles edition) for the Dynamic Trip Rescheduling demo.

Run locally:
  uvicorn backend.main_miles:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from urllib.parse import urljoin
from src.plan.router import create_router as create_plan_router


# --------------------------------------------------------------------------------------
# Import path setup (repo root + src/*)
# --------------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]          # repo root (/app)
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Settings route (unchanged)
from backend.settings_routes import router as settings_router  # type: ignore

# Cost env shim
try:
    from backend.ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore
except Exception:
    sys.path.append(str(THIS_FILE.parent))
    from ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore

# cuOpt adapter: prefer *_miles; fallback to the older module name
try:
    from opt.cuopt_model_miles import CuOptModel  # type: ignore
except Exception:
    from opt.cuopt_model import CuOptModel  # type: ignore

# NEW: modular planner router
from src.plan.router import create_router as create_plan_router

# --------------------------------------------------------------------------------------
# FastAPI app + global config
# --------------------------------------------------------------------------------------
DEBUG_API = os.getenv("DEBUG_API", "1") == "1"

app = FastAPI(title="Dynamic Trip Rescheduling (cuOpt, miles)")

@app.exception_handler(Exception)
async def all_exc_handler(request, exc):
    tb = traceback.format_exc()
    payload = {"error": str(exc)}
    if DEBUG_API:
        payload["traceback"] = tb
    return JSONResponse(status_code=500, content=payload)

# Basic config / CORS
BASE_DIR = Path(os.getenv("PRIVATE_DATA_DIR", "./data")).resolve()
DATASET_DIR = BASE_DIR / "active" if (BASE_DIR / "active").exists() else BASE_DIR

CUOPT_URL = os.getenv("CUOPT_URL", "http://cuopt:5000")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",")] if CORS_ALLOW_ORIGINS else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount other routers
app.include_router(settings_router)

# Admin router (kept)
admin = APIRouter(prefix="/admin", tags=["Admin"])

# --------------------------------------------------------------------------------------
# Files we expect in the private dataset folder
# --------------------------------------------------------------------------------------
DISTANCE_FILE       = DATASET_DIR / "distance_miles_matrix.npz"
TIME_FILE           = DATASET_DIR / "time_minutes_matrix.npz"
LOC_INDEX_FILE      = DATASET_DIR / "location_index.csv"
DRIVER_STATES_FILE  = DATASET_DIR / "driver_states.json"
LOCATIONS_CSV_FILE  = DATASET_DIR / "locations.csv"   # old RSL-style
CENTERS_CSV_FILE    = DATASET_DIR / "centers.csv"     # new data_prep output

# --------------------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------------------
def _load_npz_any(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    z = np.load(path, allow_pickle=False)
    if isinstance(z, np.lib.npyio.NpzFile):
        for k in ("matrix", "arr", "arr_0"):
            if k in z:
                return z[k]
        # fallback: first array
        for k in z.files:
            return z[k]
        raise ValueError(f"NPZ {path} contains no arrays.")
    return z  # npy array

def _maybe_read_locations_df() -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame describing locations for geo metadata.
    Prefers RSL-style 'locations.csv'. If absent, adapts 'centers.csv' to
    the columns expected by src.plan.geo.build_loc_meta_from_locations_csv():
      - "Mapped Name A", "From Site", "Lat_A", "Long_A", "Mapped Postcode A"
    """
    if LOCATIONS_CSV_FILE.exists():
        try:
            df = pd.read_csv(LOCATIONS_CSV_FILE)
            # Make sure required columns exist—best effort; planner is robust to empties.
            for c in ["Mapped Name A", "Lat_A", "Long_A"]:
                if c not in df.columns:
                    raise ValueError(f"locations.csv missing column '{c}'")
            # Fill aliases if missing
            if "From Site" not in df.columns:
                df["From Site"] = df["Mapped Name A"]
            if "Mapped Postcode A" not in df.columns:
                df["Mapped Postcode A"] = None
            return df
        except Exception as e:
            print(f"[startup] WARN: failed to read {LOCATIONS_CSV_FILE.name}: {e}", flush=True)
            return None

    if CENTERS_CSV_FILE.exists():
        try:
            c = pd.read_csv(CENTERS_CSV_FILE)
            # Expected columns from data_prep: name, [postcode], [lat], [lon]
            out = pd.DataFrame()
            out["Mapped Name A"] = c.get("name", pd.Series(dtype=str)).astype(str)
            out["From Site"] = out["Mapped Name A"]
            out["Lat_A"] = pd.to_numeric(c.get("lat", pd.Series(dtype=float)), errors="coerce")
            out["Long_A"] = pd.to_numeric(c.get("lon", pd.Series(dtype=float)), errors="coerce")
            out["Mapped Postcode A"] = c.get("postcode", pd.Series(dtype=str))
            return out
        except Exception as e:
            print(f"[startup] WARN: failed to adapt centers.csv: {e}", flush=True)
            return None

    return None

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

    # IMPORTANT: keys in uppercase (planner looks up with .upper())
    location_to_index = {str(name).upper(): int(idx) for name, idx in zip(li["name"], li["center_id"])}

    # Driver states (optional)
    if DRIVER_STATES_FILE.exists():
        try:
            driver_states = json.loads(DRIVER_STATES_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to read driver_states.json: {e}")
    else:
        driver_states = {}

    # Optional locations_df for geo meta
    locations_df = _maybe_read_locations_df()

    return {
        "distance": dist,
        "time": tmat,
        "location_to_index": location_to_index,
        "driver_states": driver_states,
        "location_index_size": n,
        "locations_df": locations_df,  # may be None; planner handles that
    }

# --------------------------------------------------------------------------------------
# Admin: cuOpt self-test
# --------------------------------------------------------------------------------------
def _safe_json(resp):
    try:
        return True, resp.json()
    except Exception:
        return False, {"status_code": resp.status_code, "text": (resp.text or "")[:500]}

@admin.get("/cuopt_selftest")
def cuopt_selftest():
    ok = False
    step = "start"
    meta: Dict[str, Any] = {}
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
        ping = requests.get(urljoin(CUOPT_URL.rstrip('/')+'/', '')).json()
        step = "submit"
        req_id = model._submit_request({"ping": "ok"})
        step = "poll"
        res = model._poll_result(req_id)
        ok = True
        return {"ok": ok, "step": step, "ping": ping, "req_id": req_id, "result_keys": list(res.keys())}
    except Exception as e:
        return {"ok": ok, "step": step if step else "error", "error": str(e), **meta}

app.include_router(admin)

# --------------------------------------------------------------------------------------
# Global state + lifecycle
# --------------------------------------------------------------------------------------
DATA: Optional[Dict[str, Any]] = None
COST_CONFIG = read_cost_env_defaults()

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
        "has_locations_df": DATA["locations_df"] is not None,
    }

@app.post("/admin/reload")
def admin_reload():
    info = reload_private_data()
    return {"status": "ok", "reloaded": info}

# --------------------------------------------------------------------------------------
# Public health/config endpoints
# --------------------------------------------------------------------------------------
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
        "has_locations_df": DATA["locations_df"] is not None,
    }

@app.get("/config")
def config():
    return {
        "cost_config": COST_CONFIG,
        "cors_allow_origins": allow_origins,
        "private_data_dir": str(BASE_DIR),
    }

# --------------------------------------------------------------------------------------
# Solve (unchanged)
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Include the NEW plan router (modular)
# --------------------------------------------------------------------------------------
# These callbacks give the planner access to the in-memory data & cost config
def _get_data():
    return DATA

def _get_cost_config():
    return COST_CONFIG

def _get_cuopt_url():
    return CUOPT_URL

plan_router = create_plan_router(lambda: DATA, lambda: COST_CONFIG, lambda: CUOPT_URL)
app.include_router(plan_router)

# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main_miles:app", host="0.0.0.0", port=port, reload=False)
