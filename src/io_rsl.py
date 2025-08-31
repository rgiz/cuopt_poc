# src/io_rsl.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import os
import pandas as pd

from .geocode_uk import enrich_locations
from .timeparse import combine_dt, minutes_since_midnight, window_from_priority

@dataclass
class ColumnMap:
    # RSL / duties
    duty_id: str = "DutyID"
    driver_id: str = "DriverID"
    sequence: str = "Seq"
    location_id: str = "LocID"
    service_date: str = "ServiceDate"
    start_time: str = "StartTime"
    end_time: str | None = None
    service_mins: str | None = "ServiceMins"
    priority: str | None = "Priority"
    type: str | None = "Type"

    # Locations
    loc_location_id: str = "location_id"
    loc_name: str | None = "name"
    loc_lat: str = "lat"
    loc_lon: str = "lon"
    loc_postcode: str | None = "postcode"

def _get_colmap(cfg: Dict[str, Any]) -> ColumnMap:
    m = ColumnMap()
    overrides = (cfg or {}).get("column_map", {})
    for k, v in overrides.items():
        if hasattr(m, k):
            setattr(m, k, v)
    return m

def load_and_normalize(rsl_path: str, locations_path: str | None, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (locations_df, duties_df, drivers_df)
    locations_df: columns -> location_id, name?, lat, lon, postcode?
    duties_df:    duty_id, driver_id, sequence, location_id, service_start_plan, service_duration, priority, type, window_start, window_end
    drivers_df:   driver_id, shift_start, shift_end
    """
    col = _get_colmap(config)

    # --- Load RSL ---
    rsl = pd.read_csv(rsl_path, dtype=str)
    # coerce service mins to numeric if present
    if col.service_mins and col.service_mins in rsl.columns:
        rsl[col.service_mins] = pd.to_numeric(rsl[col.service_mins], errors="coerce").fillna(0).astype(int)

    # --- Load locations ---
    if locations_path and os.path.exists(locations_path):
        locs_raw = pd.read_csv(locations_path)
        # allow either already-normalized columns or raw ones
        rename = {}
        if col.loc_location_id in locs_raw.columns:
            rename[col.loc_location_id] = "location_id"
        if col.loc_lat in locs_raw.columns:
            rename[col.loc_lat] = "lat"
        if col.loc_lon in locs_raw.columns:
            rename[col.loc_lon] = "lon"
        if col.loc_postcode and (col.loc_postcode in locs_raw.columns):
            rename[col.loc_postcode] = "postcode"
        if col.loc_name and (col.loc_name in locs_raw.columns):
            rename[col.loc_name] = "name"
        locs = locs_raw.rename(columns=rename)
    else:
        # If locations.csv not provided, derive unique locations from RSL minimal info
        locs = rsl[[col.location_id]].drop_duplicates().rename(columns={col.location_id: "location_id"})
        locs["lat"] = None
        locs["lon"] = None
        locs["postcode"] = None

    # --- Enrich & filter locations ---
    locs, issues = enrich_locations(locs, id_col="location_id", lat_col="lat", lon_col="lon", pc_col="postcode" if "postcode" in locs.columns else None)
    issues.to_json(os.path.join(os.getenv("DATA_ROOT", "/data"), "prep_report_locations.json"), orient="records", lines=True)
    locs = locs.dropna(subset=["lat", "lon"]).copy()

    # --- Attach parsed datetimes & windows to RSL ---
    def _build_times(row):
        dt_start = combine_dt(row[col.service_date], row[col.start_time])
        svc = 0
        if col.service_mins and (col.service_mins in rsl.columns):
            svc = int(row[col.service_mins] or 0)
        elif col.end_time and (col.end_time in rsl.columns):
            dt_end = combine_dt(row[col.service_date], row[col.end_time])
            svc = int(max(0, (dt_end - dt_start).total_seconds() // 60))
        start_min = minutes_since_midnight(dt_start)
        # SLA window
        priority_val = int(row.get(col.priority, 3)) if (col.priority and col.priority in rsl.columns) else 3
        w_start, w_end = window_from_priority(priority_val, dt_start)
        return pd.Series({"service_start_plan": start_min, "service_duration": svc, "window_start": w_start, "window_end": w_end, "priority": priority_val})

    times = rsl.apply(_build_times, axis=1)
    duties = pd.DataFrame({
        "duty_id": rsl[col.duty_id],
        "driver_id": rsl[col.driver_id],
        "sequence": pd.to_numeric(rsl[col.sequence], errors="coerce").fillna(0).astype(int),
        "location_id": rsl[col.location_id],
        "type": rsl[col.type] if (col.type and col.type in rsl.columns) else "task",
    }).join(times)

    # drop duties whose location_id is not geocoded
    geocoded_ids = set(locs["location_id"])
    duties = duties[duties["location_id"].isin(geocoded_ids)].reset_index(drop=True)

    # --- Basic drivers table (shift bounds defaulted if not provided) ---
    drivers = duties[["driver_id"]].drop_duplicates().reset_index(drop=True)
    drivers["shift_start"] = 0
    drivers["shift_end"] = 1440

    return locs.reset_index(drop=True), duties, drivers
