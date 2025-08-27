#!/usr/bin/env python3
"""
Driver duty windows from df_rsl_clean (STRICT rules)

1) START FACILITY -> Commencement Time = start
2) END FACILITY   -> Ending Time       = end
3) If end < start, add 24h (cross-midnight)
4) Allowed days = Y flags on START row (fallback: union across duty rows)
5) Home center from START row "Mapped Name A" via location_index.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

DAYSETS = [
    ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
]

def parse_hms_to_minutes(x):
    try:
        h,m,s = str(x).split(":")
        return int(h)*60 + int(m) + int(s)/60.0
    except Exception:
        return np.nan

def truthy(v):
    if isinstance(v, (int,float)):
        return int(v) != 0
    if not isinstance(v, str):
        return False
    return v.strip().lower() in {"y","yes","true","1","t"}

def find_day_cols(df):
    for cand in DAYSETS:
        if all(col in df.columns for col in cand):
            return cand
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in DAYSETS:
        if all(day.lower() in lower_cols for day in cand):
            return [lower_cols[day.lower()] for day in cand]
    return []

def main(args):
    df = pd.read_csv(args.csv, dtype=str)
    req = ["Duty ID","Element Type","Commencement Time","Ending Time"]
    for c in req:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    day_cols = find_day_cols(df)
    if not day_cols:
        print("WARNING: No day-of-week columns detected. Assuming all days.", file=sys.stderr)
        day_cols = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    df["start_min"] = df["Commencement Time"].apply(parse_hms_to_minutes)
    df["end_min"]   = df["Ending Time"].apply(parse_hms_to_minutes)
    df["ET_NORM"] = df["Element Type"].str.upper().fillna("")

    name_to_id = {}
    if args.location_index and Path(args.location_index).exists():
        li = pd.read_csv(args.location_index)
        if {"name","center_id"}.issubset(li.columns):
            name_to_id = dict(zip(li["name"], li["center_id"]))

    states = {}
    rows = []

    for duty_id, g in df.groupby("Duty ID", dropna=True):
        g = g.copy()

        start_rows = g[g["ET_NORM"] == "START FACILITY"]
        end_rows   = g[g["ET_NORM"] == "END FACILITY"]

        if start_rows.empty:
            start_row = g.sort_values("start_min", ascending=True).iloc[0]
        else:
            start_row = start_rows.sort_values("start_min", ascending=True).iloc[0]

        if end_rows.empty:
            end_row = g.sort_values("end_min", ascending=False).iloc[0]
        else:
            end_row = end_rows.sort_values("end_min", ascending=False).iloc[0]

        start = float(start_row["start_min"])
        end   = float(end_row["end_min"])
        if end < start:
            end += 24*60

        allowed_days = []
        for dcol in day_cols:
            if dcol in start_row and truthy(start_row[dcol]):
                allowed_days.append(dcol)
        if not allowed_days:
            for dcol in day_cols:
                vals = g[dcol] if dcol in g.columns else []
                if any(truthy(v) for v in vals):
                    allowed_days.append(dcol)
        if not allowed_days:
            allowed_days = day_cols[:]

        home_center_id = None
        start_name = start_row.get("Mapped Name A")
        if isinstance(start_name, str) and start_name.strip():
            home_center_id = int(name_to_id.get(start_name, -1))

        grade = None
        if "Driver Grade" in g.columns and g["Driver Grade"].notna().any():
            grade = g["Driver Grade"].dropna().iloc[0]

        vehicle_types = []
        if "Vehicle Type" in g.columns:
            vehicle_types = sorted([v for v in g["Vehicle Type"].dropna().unique() if v and str(v).strip().lower() != "no_data"])

        availability = {d: [{"start": int(round(start)), "end": int(round(end))}] for d in allowed_days}

        states[str(duty_id)] = {
            "home_center_id": home_center_id,
            "grade": grade,
            "vehicle_types": vehicle_types,
            "availability": availability,
            "max_daily_minutes": 13*60,
            "min_rest_minutes": 11*60,
            "weekend_rest_minutes": 45*60,
            "allowed_days": allowed_days,
            "emergency_rest_quota": 2,
        }
        rows.append({
            "duty_id": duty_id,
            "window_start_min": start,
            "window_end_min": end,
            "allowed_days": ",".join(allowed_days),
            "home_center_id": home_center_id,
            "grade": grade,
            "vehicle_types": "|".join(vehicle_types),
        })

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(states, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_json.with_suffix(".csv"), index=False, encoding="utf-8")
    print(f"Wrote {out_json} and {out_json.with_suffix('.csv')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to df_rsl_clean-like CSV")
    ap.add_argument("--location_index", default="", help="Optional centers mapping (location_index.csv)")
    ap.add_argument("--out", default="driver_states.json", help="Output JSON path")
    args = ap.parse_args()
    main(args)
