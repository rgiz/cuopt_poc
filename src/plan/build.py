# src/plan/build.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

def _index_map(locs: pd.DataFrame, matrices: Dict[str, Any]) -> dict[str, int]:
    # If a location_index was loaded, prefer it; else derive by order in locs
    if "index" in matrices and isinstance(matrices["index"], pd.DataFrame) and "matrix_index" in matrices["index"].columns:
        idx_df = matrices["index"]
        # expect columns: location_id, matrix_index
        m = {str(r["location_id"]): int(r["matrix_index"]) for _, r in idx_df.iterrows()}
        return m
    return {str(loc_id): i for i, loc_id in enumerate(locs["location_id"].tolist())}

def from_rsl(
    locations: pd.DataFrame,
    duties: pd.DataFrame,
    drivers: pd.DataFrame,
    matrices: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Output dict structure expected by your solver adapter.
    """
    cfg = cfg or {}
    id2idx = _index_map(locations, matrices)

    # Vehicles
    vehicles: List[Dict[str, Any]] = []
    for _, r in drivers.iterrows():
        vehicles.append({
            "id": str(r["driver_id"]),
            "start_index": None,  # optional: set a depot index if you have one
            "end_index": None,
            "time_window": (int(r.get("shift_start", 0)), int(r.get("shift_end", 1440))),
            "breaks": [],  # TODO if you have breaks
            "capacity": {},  # placeholder if you model capacities
        })

    # Orders
    orders: List[Dict[str, Any]] = []
    missing_locs: set[str] = set()
    for _, r in duties.iterrows():
        lid = str(r["location_id"])
        if lid not in id2idx:
            missing_locs.add(lid)
            continue
        orders.append({
            "id": f'{r["duty_id"]}:{r["sequence"]}',
            "index": id2idx[lid],
            "service_time": int(r.get("service_duration", 0)),
            "time_window": (int(r.get("window_start", 0)), int(r.get("window_end", 1440))),
            "priority": int(r.get("priority", 3)),
            "pickup_index": None,
            "dropoff_index": None,
            "duty_id": r["duty_id"],
            "driver_id_plan": r["driver_id"],  # planned driver (useful for warm starts)
        })

    if missing_locs:
        # soft warning; they should already be filtered
        pass

    problem = {
        "vehicles": vehicles,
        "orders": orders,
        "matrices": {
            "time": matrices["time"],   # minutes
            "dist": matrices["dist"],   # miles
        },
        "index_map": id2idx,
        "location_id_by_index": {v: k for k, v in id2idx.items()},
    }
    return problem
