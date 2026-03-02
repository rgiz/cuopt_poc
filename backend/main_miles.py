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
import os, time

from backend.settings_routes import router as settings_router  # type: ignore

try:
    from backend.ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore
except ImportError:
    from ENV_COMPAT_SNIPPET import read_cost_env_defaults  # type: ignore

from src.plan.router import create_router as create_plan_router


# --------------------------------------------------------------------------------------------------
# Global Constants & Environment
# --------------------------------------------------------------------------------------------------
DEBUG_API = os.getenv("DEBUG_API", "1") == "1"
BASE_DIR = Path(os.getenv("PRIVATE_DATA_DIR", "./data")).resolve()
DATASET_DIR = BASE_DIR / "private" / "active" if (BASE_DIR / "private" / "active").exists() else BASE_DIR

print(f"[startup] Using DATASET_DIR: {DATASET_DIR}")

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
        """Test cuOpt connectivity using the official client"""
        step = "import"
        try:
            # Import the official client
            try:
                from cuopt_sh_client import CuOptServiceSelfHostClient
                step = "client_available"
            except ImportError as e:
                return {
                    "ok": False, 
                    "step": "import_failed", 
                    "error": f"cuopt-sh-client not installed: {e}"
                }
            
            # Create client
            step = "create_client"
            cuopt_host = os.getenv("CUOPT_HOST", "cuopt")
            cuopt_client = CuOptServiceSelfHostClient(
                ip=cuopt_host,
                port=5000,
                polling_timeout=10,
                timeout_exception=False
            )
            
            # Test with minimal valid payload
            step = "build_payload"
            test_data = {
                "cost_matrix_data": {
                    "data": {"1": [[0, 1], [1, 0]]}  # 2x2 cost matrix
                },
                "fleet_data": {
                    "vehicle_locations": [[0, 0]],   # 1 vehicle at location 0
                    "vehicle_types": [1],            # Vehicle type 1
                    "capacities": [[100]]            # Vehicle capacity
                },
                "task_data": {
                    "task_locations": [1],           # 1 task at location 1
                    "demand": [[1]]                  # Task demand
                }
            }
            
            # Submit to cuOpt
            step = "submit"
            print(f"[selftest] Testing cuOpt at {cuopt_host}:5000...")
            solution = cuopt_client.get_optimized_routes(test_data)
            
            # Handle async response (repoll if needed)
            step = "repoll"
            if "reqId" in solution and "response" not in solution:
                req_id = solution["reqId"]
                print(f"[selftest] Repolling reqId: {req_id}")
                
                for i in range(10):
                    solution = cuopt_client.repoll(req_id, response_type="dict")
                    if "response" in solution:
                        print(f"[selftest] Repoll successful after {i+1} attempts")
                        break
                    import time
                    time.sleep(0.5)
            
            # Check result
            step = "validate_response"
            if solution and "response" in solution:
                solver_response = solution["response"].get("solver_response", {})
                status = solver_response.get("status", -1)
                cost = solver_response.get("solution_cost", 0)
                
                if status == 0:
                    return {
                        "ok": True,
                        "step": "complete",
                        "status": status,
                        "cost": cost,
                        "message": f"cuOpt server healthy at {cuopt_host}:5000"
                    }
                else:
                    return {
                        "ok": False,
                        "step": "solver_failed",
                        "status": status,
                        "error": f"Solver returned status {status}"
                    }
            else:
                return {
                    "ok": False,
                    "step": "no_response",
                    "error": "No valid response from cuOpt"
                }
                
        except Exception as e:
            import traceback
            return {
                "ok": False,
                "step": step,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    @router.post("/reload")
    def admin_reload():
        """Reload data from disk"""
        global DATA, COST_CONFIG
        try:
            DATA = load_private_data()  # ✅ This is the correct function name
            COST_CONFIG = read_cost_env_defaults()  # ✅ This is also correct
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
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
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
    app.include_router(create_plan_router(lambda: DATA, lambda: COST_CONFIG))

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
    """Read locations data for enhanced geo lookups"""
    
    # Try locations.csv first (NEW format)
    if LOCATIONS_CSV_FILE.exists():
        try:
            df = pd.read_csv(LOCATIONS_CSV_FILE)
            # Check if it has our new format: name, postcode, lat, lon
            if {"name", "lat", "lon"}.issubset(df.columns):
                return pd.DataFrame({
                    "Mapped Name A": df["name"],
                    "From Site": df["name"],
                    "Lat_A": pd.to_numeric(df["lat"], errors="coerce"),
                    "Long_A": pd.to_numeric(df["lon"], errors="coerce"),
                    "Mapped Postcode A": df.get("postcode", pd.Series(dtype=str)),
                })
            # Fall back to old format if it exists
            elif {"Mapped Name A", "Lat_A", "Long_A"}.issubset(df.columns):
                df.setdefault("From Site", df["Mapped Name A"])
                df.setdefault("Mapped Postcode A", None)
                return df
        except Exception as e:
            print(f"[startup] WARN: failed to read {LOCATIONS_CSV_FILE.name}: {e}")

    # Try centers.csv as fallback (for backwards compatibility)
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
