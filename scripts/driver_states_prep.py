#!/usr/bin/env python3
"""
Driver duty windows from df_rsl_clean (STRICT logic per your rules)

Rules implemented:
1) Per Duty ID, find the row with Element Type == "START FACILITY".
   Take its Commencement Time as the duty start.
2) Per Duty ID, find the row with Element Type == "END FACILITY".
   Take its Ending Time as the duty end.
3) If end < start, add 24h to end (cross-midnight).
4) Allowed days come from the boolean "days of week" on the START row.
   - Supports either Mon/Tue/... or Monday/Tuesday/... column names.
   - Accepts Y/Yes/True/1 (case-insensitive). If missing, fallback to the
     union of days across all rows for that duty; if still missing, default to all days.
5) Home center inferred from START row "Mapped Name A" via location_index.csv if provided.

Outputs:
- driver_states.json (consumed by backend)
- driver_states.csv (QA summary)

Requirements: pandas, numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAYSETS = [
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
]


def parse_hms_to_minutes(x):
    """Convert 'HH:MM:SS' to minutes since midnight (float)."""
    try:
        h, m, s = str(x).split(":")
        return int(h) * 60 + int(m) + int(s) / 60.0
    except Exception:
        return np.nan


def truthy(v):
    """Return True for Y/Yes/True/1 (case-insensitive), else False."""
    if isinstance(v, (int, float)):
        return int(v) != 0
    if not isinstance(v, str):
        return False
    s = v.strip().lower()
    return s in {"y", "yes", "true", "1", "t"}


def find_day_cols(df: pd.DataFrame):
    """Detect day-of-week columns (Mon..Sun or Monday..Sunday, case-insensitive)."""
    # Exact case match
    for cand in DAYSETS:
        if all(col in df.columns for col in cand):
            return cand
    # Case-insensitive match
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in DAYSETS:
        if all(day.lower() in lower_cols for day in cand):
            return [lower_cols[day.lower()] for day in cand]
    return []


def main(args):
    df = pd.read_csv(args.csv, dtype=str)

    # Required columns
    required = ["Duty ID", "Element Type", "Commencement Time", "Ending Time"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # Day-of-week columns
    day_cols = find_day_cols(df)
    if not day_cols:
        print("WARNING: No day-of-week columns detected. Assuming all days.", file=sys.stderr)
        day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Precompute minutes
    df["start_min"] = df["Commencement Time"].apply(parse_hms_to_minutes)
    df["end_min"] = df["Ending Time"].apply(parse_hms_to_minutes)

    # Normalize element type for robust matching
    def norm_et(s):
        return (s or "").strip().upper()

    df["ET_NORM"] = df["Element Type"].apply(norm_et)

    # Optional centers mapping for home_center_id
    name_to_id = {}
    if args.location_index and Path(args.location_index).exists():
        li = pd.read_csv(args.location_index)
        if {"name", "center_id"}.issubset(li.columns):
            name_to_id = dict(zip(li["name"], li["center_id"]))

    states = {}
    qa_rows = []

    for duty_id, g in df.groupby("Duty ID", dropna=True):
        g = g.copy()

        # START / END rows by your rules
        start_rows = g[g["ET_NORM"] == "START FACILITY"]
        end_rows = g[g["ET_NORM"] == "END FACILITY"]

        if start_rows.empty:
            # Fallback: earliest start_min row
            start_row = g.sort_values("start_min", ascending=True).iloc[0]
        else:
            # If multiple, take earliest commencement
            start_row = start_rows.sort_values("start_min", ascending=True).iloc[0]

        if end_rows.empty:
            # Fallback: latest end_min row
            end_row = g.sort_values("end_min", ascending=False).iloc[0]
        else:
            end_row = end_rows.sort_values("end_min", ascending=False).iloc[0]

        start = float(start_row["start_min"])
        end = float(end_row["end_min"])

        # Cross-midnight: e.g., Mon 21:00 -> Tue 04:00 => 21:00 -> 28:00
        if end < start:
            end += 24 * 60

        # Allowed days primarily from the START row
        allowed_days = []
        for dcol in day_cols:
            if dcol in start_row and truthy(start_row[dcol]):
                allowed_days.append(dcol)

        # Fallback: union across all rows for the duty (any element that starts that day)
        if not allowed_days:
            for dcol in day_cols:
                vals = g[dcol] if dcol in g.columns else []
                if any(truthy(v) for v in vals):
                    allowed_days.append(dcol)

        # Ultimate fallback: all days
        if not allowed_days:
            allowed_days = day_cols[:]

        # Home center from START row "Mapped Name A"
        home_center_id = None
        start_name = start_row.get("Mapped Name A")
        if isinstance(start_name, str) and start_name.strip():
            home_center_id = int(name_to_id.get(start_name, -1))

        # Optional attributes
        grade = None
        if "Driver Grade" in g.columns and g["Driver Grade"].notna().any():
            grade = g["Driver Grade"].dropna().iloc[0]

        vehicle_types = []
        if "Vehicle Type" in g.columns:
            vehicle_types = sorted(
                v
                for v in g["Vehicle Type"].dropna().unique()
                if v and str(v).strip().lower() != "no_data"
            )

        # Same (start,end) window for each allowed day; end can exceed 1440 when crossing midnight
        availability = {d: [{"start": int(round(start)), "end": int(round(end))}] for d in allowed_days}

        # Assemble state
        states[str(duty_id)] = {
            "home_center_id": home_center_id,
            "grade": grade,
            "vehicle_types": vehicle_types,
            "availability": availability,
            "max_daily_minutes": 13 * 60,
            "min_rest_minutes": 11 * 60,
            "weekend_rest_minutes": 45 * 60,
            "allowed_days": allowed_days,
            "emergency_rest_quota": 2,
        }

        qa_rows.append(
            {
                "duty_id": duty_id,
                "window_start_min": start,
                "window_end_min": end,
                "allowed_days": ",".join(allowed_days),
                "home_center_id": home_center_id,
                "grade": grade,
                "vehicle_types": "|".join(vehicle_types),
            }
        )

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(states, indent=2))
    pd.DataFrame(qa_rows).to_csv(out_json.with_suffix(".csv"), index=False)
    print(f"Wrote {out_json} and {out_json.with_suffix('.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to df_rsl_clean-like CSV")
    ap.add_argument(
        "--location_index",
        default="",
        help="Optional centers mapping (location_index.csv with columns: name,center_id)",
    )
    ap.add_argument("--out", default="driver_states.json", help="Output JSON path")
    args = ap.parse_args()
    main(args)
