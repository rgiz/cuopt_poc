# backend/settings_routes.py
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(tags=["settings"])

# ---- Paths & helpers ---------------------------------------------------------

PRIVATE_DATA_DIR = Path(os.getenv("PRIVATE_DATA_DIR", "./data/private")).resolve()
GLOBAL_SETTINGS_PATH = PRIVATE_DATA_DIR / "global_settings.json"

DATASETS_DIR = PRIVATE_DATA_DIR / "datasets"
ACTIVE_LINK = PRIVATE_DATA_DIR / "active"  # symlink or directory

def ensure_dirs():
    PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def get_active_dir() -> Path:
    """
    Returns the active dataset path.
    Fallbacks (in order):
      1. /data/private/active (symlink or dir)
      2. /data/private/ (flat mode)
    """
    ensure_dirs()
    if ACTIVE_LINK.exists():
        return ACTIVE_LINK.resolve()
    return PRIVATE_DATA_DIR

def default_global_settings() -> Dict[str, Any]:
    # Safe defaults (edit later in Settings UI)
    return {
        "cost_config": {
            "delay_cost_per_minute": 10.0,
            "deadhead_cost_per_mile": 2.0,
            "reassignment_admin_cost": 10.0,
            "emergency_rest_penalty": 50.0,
            "outsourcing_base_cost": 200.0,
            "outsourcing_per_mile": 2.0,
            "overtime_cost_per_minute": 3.0,
            "rank_deadhead_miles_weight": 1.0,
            "rank_deadhead_minutes_weight": 0.15,
            "rank_overtime_minutes_weight": 2.0,
            "rank_penalty_append": 30.0,
        },
        "sla_penalty": {
            "1": [[15, 2.0], [60, 5.0], [9999, 10.0]],
            "2": [[15, 1.5], [60, 4.0], [9999, 8.0]],
            "3": [[30, 1.0], [120, 3.0], [9999, 6.0]],
            "4": [[30, 0.5], [120, 2.0], [9999, 4.0]],
            "5": [[60, 0.25], [180, 1.0], [9999, 2.0]],
        },
        "constraints": {
            "max_daily_minutes": 13 * 60,
            "min_rest_minutes": 11 * 60,
            "emergency_rest_minutes": 9 * 60,
            "weekend_rest_minutes": 45 * 60,
            "emergency_rest_quota_week": 2,
            "overtime_cap_minutes": 120,
        },
        "cascade": {"max_depth": 2, "max_drivers_affected": 3, "top_k_per_depth": 20},
    }

def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read {path.name}: {e}")
    return default or {}

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write {path.name}: {e}")

# ---- global settings ---------------------------------------------------------

@router.get("/settings")
def get_settings():
    ensure_dirs()
    cfg = load_json(GLOBAL_SETTINGS_PATH, default_global_settings())
    return cfg

@router.post("/settings")
def post_settings(payload: Dict[str, Any]):
    ensure_dirs()
    # (Optional) light validation of top-level keys
    for k in ["cost_config", "sla_penalty", "constraints", "cascade"]:
        if k not in payload:
            raise HTTPException(status_code=400, detail=f"Missing key: {k}")
    save_json(GLOBAL_SETTINGS_PATH, payload)
    return {"status": "ok", "message": "global_settings.json updated"}

# ---- priority map (per dataset; default 'active') ----------------------------

def priority_path(dataset_id: str) -> Path:
    d = get_active_dir() if dataset_id == "active" else (DATASETS_DIR / dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / "priority_map.json"

@router.get("/priority_map")
def get_priority_map(dataset_id: str = Query("active")):
    path = priority_path(dataset_id)
    if not path.exists():
        return JSONResponse(content={}, status_code=200)
    return JSONResponse(content=load_json(path, {}))

@router.post("/priority_map")
def post_priority_map(payload: Dict[str, int], dataset_id: str = Query("active")):
    # Normalize to uppercase keys; keep numeric values
    normalized = {}
    for k, v in payload.items():
        try:
            normalized[str(k).upper()] = int(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid value for key '{k}': {v}")
    path = priority_path(dataset_id)
    save_json(path, normalized)
    return {"status": "ok", "rows": len(normalized), "message": f"priority_map.json updated ({dataset_id})"}

# ---- locations (per dataset; default 'active') -------------------------------

def locations_paths(dataset_id: str) -> Dict[str, Path]:
    d = get_active_dir() if dataset_id == "active" else (DATASETS_DIR / dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "root": d,
        "locations": d / "locations.csv",
        "location_index": d / "location_index.csv",
    }

def normalize_location_index(df: pd.DataFrame) -> pd.DataFrame:
    import re
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def pick(colnames, *cands):
        for c in cands:
            if c in colnames:
                return c
        return None

    cols = set(df.columns)

    name_col = pick(cols, "name", "mapped_name", "mapped_name_a", "site_name", "location_name")
    if not name_col:
        for c in cols:
            if "name" in c:
                name_col = c
                break
    if not name_col:
        raise HTTPException(status_code=400, detail="locations.csv must have a 'name' column.")

    postcode_col = pick(cols, "postcode", "post_code", "zip", "zip_code", "from_postcode", "to_postcode")
    lat_col = pick(cols, "lat", "latitude", "from_lat", "to_lat")
    lon_col = pick(cols, "lon", "lng", "long", "longitude", "from_long", "to_long")

    out = pd.DataFrame({"name": df[name_col].astype(str).str.upper().str.strip()})

    if postcode_col:
        out["postcode"] = df[postcode_col].astype(str).str.strip()
        out.loc[out["postcode"].str.lower().isin(["", "nan", "none", "no_data"]), "postcode"] = pd.NA
    else:
        out["postcode"] = pd.NA

    uk_pc = re.compile(
        r"\b(GIR\s?0AA|[A-PR-UWYZ][0-9]{1,2}"
        r"|[A-PR-UWYZ][A-HK-Y][0-9]{1,2}"
        r"|[A-PR-UWYZ][0-9][A-HJKS-UW]"
        r"|[A-PR-UWYZ][A-HK-Y][0-9][ABEHMNPRV-Y])\s?[0-9][ABD-HJLNP-UW-Z]{2}\b",
        re.IGNORECASE,
    )
    def extract_pc(s: str):
        m = uk_pc.search(s or "")
        return m.group(0).upper().replace(" ", "") if m else None

    need_pc = out["postcode"].isna()
    if need_pc.any():
        out.loc[need_pc, "postcode"] = out.loc[need_pc, "name"].map(lambda s: extract_pc(s) or pd.NA)

    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else pd.NA
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else pd.NA

    out = out.drop_duplicates(subset=["name"]).sort_values("name").reset_index(drop=True)
    out.insert(0, "center_id", range(len(out)))
    return out

@router.get("/locations")
def get_locations(dataset_id: str = Query("active")):
    p = locations_paths(dataset_id)
    if not p["locations"].exists():
        raise HTTPException(status_code=404, detail=f"No locations.csv for dataset '{dataset_id}'")
    return FileResponse(p["locations"])

@router.post("/locations")
async def post_locations(
    file: UploadFile = File(..., description="CSV with columns: name, postcode, lat, lon"),
    dataset_id: str = Query("active"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    p = locations_paths(dataset_id)
    # Save original
    p["locations"].write_bytes(raw)

    # Build location_index.csv
    idx_df = normalize_location_index(df)
    idx_df.to_csv(p["location_index"], index=False, encoding="utf-8")

    # Note: matrices still need to be rebuilt separately; we only update locations here.
    missing_latlon_rows = int(((idx_df["lat"].isna()) | (idx_df["lon"].isna())).sum())

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "locations_rows": int(len(idx_df)),
        "missing_latlon_rows": missing_latlon_rows,
        "message": "locations.csv saved and location_index.csv regenerated",
    }
