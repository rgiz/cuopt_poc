#!/usr/bin/env python3

"""
Backend (miles edition) for the Dynamic Trip Rescheduling demo.

Run locally:
  uvicorn backend.main_miles:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import sys
import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from urllib.parse import urljoin

from backend.settings_routes import router as settings_router  # type: ignore

try:
    from backend.ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore
except ImportError:
    from ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore

try:
    from src.opt.cuopt_model_miles import CuOptModel  # type: ignore
except ImportError:
    from src.opt.cuopt_model import CuOptModel  # type: ignore

from src.plan.router import create_router as create_plan_router


# --------------------------------------------------------------------------------------------------
# Global Constants & Environment
# --------------------------------------------------------------------------------------------------
DEBUG_API = os.getenv("DEBUG_API", "1") == "1"
BASE_DIR = Path(os.getenv("PRIVATE_DATA_DIR", "./data")).resolve()
DATASET_DIR = BASE_DIR / "active" if (BASE_DIR / "active").exists() else BASE_DIR
CUOPT_URL = os.getenv("CUOPT_URL", "http://cuopt:5000").rstrip("/v2").rstrip("/")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()] or ["*"]

DISTANCE_FILE = DATASET_DIR / "distance_miles_matrix.npz"
TIME_FILE = DATASET_DIR / "time_minutes_matrix.npz"
LOC_INDEX_FILE = DATASET_DIR / "location_index.csv"
DRIVER_STATES_FILE = DATASET_DIR / "driver_states.json"
LOCATIONS_CSV_FILE = DATASET_DIR / "locations.csv"
CENTERS_CSV_FILE = DATASET_DIR / "centers.csv"

DATA: Optional[Dict[str, Any]] = None
COST_CONFIG: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------------------------------
# Admin Router (cuOpt self-test, reload)
# --------------------------------------------------------------------------------------------------
def admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["Admin"])

    @router.get("/cuopt_selftest")
    def cuopt_selftest():
        step = "start"
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
            
            step = "submit"
            # Test with proper payload format
            test_payload = {
                "fleet_data": {"vehicles": []},
                "task_data": {"tasks": []},
                "solver_config": {"time_limit": 5}
            }
            
            # Try direct synchronous call instead of async
            if model._solve_path == "cuopt/request":
                # This endpoint returns results immediately, not a request ID
                step = "sync_call"
                result = model._post_json("cuopt/request", test_payload)
                return {"ok": True, "step": "complete", "result_keys": list(result.keys())}
            else:
                step = "unknown_path"
                return {"ok": False, "step": step, "error": f"Unexpected solve path: {model._solve_path}"}
                
        except Exception as e:
            return {"ok": False, "step": step, "error": str(e)}

    @router.post("/reload")
    def admin_reload():
        global DATA, COST_CONFIG
        DATA = load_private_data()
        COST_CONFIG = read_cost_env_defaults()
        return {
                "status": "ok",
                "reloaded": {
                "distance_shape": list(DATA["distance"].shape),
                "time_shape": list(DATA["time"].shape),
                "locations": int(DATA["location_index_size"]),
                "has_driver_states": bool(DATA["driver_states"]),
                "cost_keys": list(COST_CONFIG.keys()),
                "has_locations_df": DATA["locations_df"] is not None,
            },
        }

    return router

# --------------------------------------------------------------------------------------------------
# Public Endpoints
# --------------------------------------------------------------------------------------------------
def register_routes(app: FastAPI):
    @app.get("/health")
    def health():
        base = {
            "status": "ok" if DATA else "needs_data",
            "private_data_dir": str(BASE_DIR),
            "dataset_dir": str(DATASET_DIR),
            "cuopt_url": CUOPT_URL,
        }
        if DATA is None:
            return {**base, "message": "Private data not loaded. POST /admin/reload."}
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
            "cors_allow_origins": ALLOW_ORIGINS,
            "private_data_dir": str(BASE_DIR),
        }

# --------------------------------------------------------------------------------------------------
# FastAPI App (with lifespan)
# --------------------------------------------------------------------------------------------------
def create_app() -> FastAPI:
    async def lifespan(app: FastAPI):
        global DATA, COST_CONFIG
        try:
            DATA = load_private_data()
            COST_CONFIG = read_cost_env_defaults()
            print(f"[startup] Loaded private data OK: D={DATA['distance'].shape}, T={DATA['time'].shape}, locations={DATA['location_index_size']}")
        except Exception as e:
            print(f"[startup] WARN: private data not loaded: {e}")
        yield

    app = FastAPI(title="Dynamic Trip Rescheduling (cuOpt, miles)", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        payload = {"error": str(exc)}
        if DEBUG_API:
            payload["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=500, content=payload)

    app.include_router(settings_router)
    app.include_router(admin_router())
    app.include_router(create_plan_router(lambda: DATA, lambda: COST_CONFIG, lambda: CUOPT_URL))

    register_routes(app)
    return app

app = create_app()


# --------------------------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------------------------
def _load_npz_any(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    for k in ("matrix", "arr", "arr_0", *z.files):
        if k in z:
            return z[k]
    raise ValueError(f"NPZ {path} contains no usable arrays.")


def _maybe_read_locations_df() -> Optional[pd.DataFrame]:
    if LOCATIONS_CSV_FILE.exists():
        try:
            df = pd.read_csv(LOCATIONS_CSV_FILE)
            if {"Mapped Name A", "Lat_A", "Long_A"}.issubset(df.columns):
                df.setdefault("From Site", df["Mapped Name A"])
                df.setdefault("Mapped Postcode A", None)
                return df
        except Exception as e:
            print(f"[startup] WARN: failed to read {LOCATIONS_CSV_FILE.name}: {e}")

    if CENTERS_CSV_FILE.exists():
        try:
            c = pd.read_csv(CENTERS_CSV_FILE)
            return pd.DataFrame({
                "Mapped Name A": c.get("name", pd.Series(dtype=str)),
                "From Site": c.get("name", pd.Series(dtype=str)),
                "Lat_A": pd.to_numeric(c.get("lat", pd.Series(dtype=float)), errors="coerce"),
                "Long_A": pd.to_numeric(c.get("lon", pd.Series(dtype=float)), errors="coerce"),
                "Mapped Postcode A": c.get("postcode", pd.Series(dtype=str)),
            })
        except Exception as e:
            print(f"[startup] WARN: failed to adapt centers.csv: {e}")

    return None


def load_private_data() -> Dict[str, Any]:
    dist = _load_npz_any(DISTANCE_FILE)
    time = _load_npz_any(TIME_FILE)
    li = pd.read_csv(LOC_INDEX_FILE)

    if not {"name", "center_id"}.issubset(li.columns):
        raise ValueError("location_index.csv must have columns: name, center_id")

    li["center_id"] = li["center_id"].astype(int)

    n = int(li["center_id"].max()) + 1

    if dist.shape != (n, n) or time.shape != (n, n):
        raise ValueError("Distance/time matrices do not match location_index size.")

    return {
        "distance": dist,
        "time": time,
        "location_to_index": {str(name).upper(): int(idx) for name, idx in zip(li["name"], li["center_id"])},
        "driver_states": json.loads(DRIVER_STATES_FILE.read_text()) if DRIVER_STATES_FILE.exists() else {},
        "location_index_size": int(n),
        "locations_df": _maybe_read_locations_df(),
    }


# --------------------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main_miles:app", host="0.0.0.0", port=port, reload=False)
