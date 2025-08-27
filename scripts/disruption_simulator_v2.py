#!/usr/bin/env python3
"""
disruption_simulator_v2.py

Samples N TRAVEL legs from df_rsl_clean (complete mileage + coords) and
creates disrupted trips with:
- 'day' (first Y among Mon..Sun)
- 'dep_minutes' (Commencement Time in minutes)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def _to_na(x):
    if isinstance(x, str) and x.strip().lower() == "no_data":
        return np.nan
    return x

def parse_hms_to_minutes(x):
    try:
        h,m,s = str(x).split(":")
        return int(h)*60 + int(m) + int(s)/60.0
    except Exception:
        return np.nan

def truthy(v):
    if isinstance(v, (int,float)): return int(v)!=0
    if not isinstance(v, str): return False
    return v.strip().lower() in {"y","yes","true","1","t"}

def main(a):
    df = pd.read_csv(a.csv, dtype=str).applymap(_to_na)
    for col in ["From Lat","From Long","To Lat","To Long","Leg Mileage"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["elem_minutes"] = df["Element Time"].apply(parse_hms_to_minutes) if "Element Time" in df.columns else np.nan
    df["ET"] = df["Element Type"].str.upper()

    travel = df[df["ET"]=="TRAVEL"].copy()
    cond = (
        travel["Leg Mileage"].notna() & (travel["Leg Mileage"] > 0) &
        travel["From Lat"].notna() & travel["From Long"].notna() &
        travel["To Lat"].notna() & travel["To Long"].notna()
    )
    travel = travel[cond].reset_index(drop=True)
    if travel.empty:
        raise SystemExit("No eligible TRAVEL legs with mileage and lat/long.")

    li = pd.read_csv(a.location_index)
    name_to_id = dict(zip(li["name"], li["center_id"]))

    def coalesce_from(row):
        v = row.get("Mapped Name A"); return v if isinstance(v,str) and v.strip() else row.get("From Postcode")
    def coalesce_to(row):
        v = row.get("Mapped Name B"); return v if isinstance(v,str) and v.strip() else row.get("To Postcode")

    travel["from_name_norm"] = travel.apply(coalesce_from, axis=1)
    travel["to_name_norm"] = travel.apply(coalesce_to, axis=1)
    travel["from_id"] = travel["from_name_norm"].map(name_to_id).astype("Int64")
    travel["to_id"] = travel["to_name_norm"].map(name_to_id).astype("Int64")
    travel = travel.dropna(subset=["from_id","to_id"]).copy()
    travel["from_id"] = travel["from_id"].astype(int)
    travel["to_id"] = travel["to_id"].astype(int)

    def pick_day(row):
        for d in DAYS:
            if d in row and truthy(row[d]):
                return d
        return "Mon"

    travel["dep_minutes"] = travel["Commencement Time"].apply(parse_hms_to_minutes)

    n = min(int(a.n), len(travel))
    sample = travel.sample(n=n, random_state=int(a.seed)).reset_index(drop=True)

    disrupted = []
    for i, r in sample.iterrows():
        disrupted.append({
            "id": f"DUTY_{r['Duty ID']}_ROW_{int(i)}",
            "duty_id": r["Duty ID"],
            "day": pick_day(r),
            "dep_minutes": float(r["dep_minutes"]) if pd.notna(r["dep_minutes"]) else 0.0,
            "start_location": r["from_name_norm"],
            "end_location": r["to_name_norm"],
            "start_center_id": int(r["from_id"]),
            "end_center_id": int(r["to_id"]),
            "trip_miles": float(r["Leg Mileage"]) if pd.notna(r["Leg Mileage"]) else None,
            "duration_minutes": float(r["elem_minutes"]) if pd.notna(r["elem_minutes"]) else None
        })

    out = {"disrupted_trips": disrupted, "candidates_per_trip": {}}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {a.out} with {len(disrupted)} disrupted trips.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="df_rsl_clean.csv")
    ap.add_argument("--location_index", required=True, help="location_index.csv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="disruptions.json")
    a = ap.parse_args()
    main(a)
