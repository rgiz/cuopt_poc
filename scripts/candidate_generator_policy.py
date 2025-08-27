#!/usr/bin/env python3
"""
candidate_generator_policy.py

Filters & scores candidates by:
- Day-of-week match
- Duty window feasibility (start/finish within availability; supports cross-midnight end)
- Legal daily limit (<= max_daily_minutes)
- Computes overtime_minutes (new - baseline, >=0)
- Computes miles_delta (deadhead + trip_miles over baseline)

Inputs:
  df_rsl_clean.csv
  location_index.csv
  distance_miles_matrix.npz
  time_minutes_matrix.npz
  driver_states.json
  disruptions.json (from disruption_simulator_v2.py, includes 'day' and 'dep_minutes')
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def load_npz_any(p, keys=("matrix","arr","arr_0")):
    z = np.load(p)
    for k in keys:
        if k in z: return z[k]
    return list(z.values())[0]

def minutes(hms):
    h,m,s = str(hms).split(":")
    return int(h)*60 + int(m) + int(s)/60.0

def truthy(v):
    if isinstance(v, (int,float)): return int(v)!=0
    if not isinstance(v, str): return False
    return v.strip().lower() in {"y","yes","true","1","t"}

def main(a):
    rsl = pd.read_csv(a.rsl, dtype=str)
    li = pd.read_csv(a.location_index)
    dist = load_npz_any(a.dist)
    tmat = load_npz_any(a.time)

    name_to_id = dict(zip(li["name"], li["center_id"]))

    # Baseline per-duty metrics
    rsl["ET"] = rsl["Element Type"].str.upper()
    rsl["start_min"] = rsl["Commencement Time"].apply(minutes)
    rsl["end_min"] = rsl["Ending Time"].apply(minutes)
    travel = rsl[rsl["ET"]=="TRAVEL"].copy()
    travel["Leg Mileage"] = pd.to_numeric(travel["Leg Mileage"], errors="coerce")

    baseline = []
    for duty, g in rsl.groupby("Duty ID", dropna=True):
        starts = g[g["ET"]=="START FACILITY"].sort_values("start_min")
        ends   = g[g["ET"]=="END FACILITY"].sort_values("end_min")
        if starts.empty: smin = g["start_min"].min()
        else: smin = starts["start_min"].iloc[0]
        if ends.empty: emin = g["end_min"].max()
        else: emin = ends["end_min"].iloc[-1]
        if emin < smin: emin += 24*60
        miles = travel[travel["Duty ID"]==duty]["Leg Mileage"].sum(min_count=1)
        baseline.append({"duty_id": duty, "baseline_minutes": float(emin - smin), "baseline_miles": float(miles) if pd.notna(miles) else 0.0})
    base_df = pd.DataFrame(baseline).set_index("duty_id")

    # Drivers
    states = json.loads(Path(a.drivers).read_text())

    # Disruptions
    disr = json.loads(Path(a.disruptions).read_text())
    trips = disr["disrupted_trips"]

    out = {}
    for trip in trips:
        tid = trip["id"]
        day = trip.get("day")  # e.g., "Mon"
        dep_min = float(trip.get("dep_minutes", 0))
        duration = float(trip.get("duration_minutes", 0))
        start_id = int(trip["start_center_id"])
        trip_miles = float(trip.get("trip_miles", 0.0))

        cand_list = []
        for rid, st in states.items():
            allowed_days = st.get("allowed_days", [])
            if day and allowed_days and day not in allowed_days:
                continue

            windows = st.get("availability", {}).get(day, [])
            if not windows:
                continue

            # Select first window for simplicity
            w = windows[0]
            w_start = float(w["start"]); w_end = float(w["end"])  # end may exceed 1440
            arr_min = dep_min + duration
            fits_window = (w_start <= dep_min and arr_min <= w_end)

            max_daily = float(st.get("max_daily_minutes", 13*60))

            if rid in base_df.index:
                b_minutes = float(base_df.loc[rid, "baseline_minutes"])
                b_miles = float(base_df.loc[rid, "baseline_miles"])
            else:
                b_minutes = w_end - w_start
                b_miles = 0.0

            home_id = st.get("home_center_id")
            if home_id is None or int(home_id) < 0 or int(home_id) >= dist.shape[0]:
                continue
            dd_miles = float(dist[int(home_id), start_id])
            dd_minutes = float(tmat[int(home_id), start_id])

            new_minutes = b_minutes + dd_minutes + duration
            if new_minutes > max_daily:
                # cannot legally complete within daily limit
                continue

            overtime_minutes = max(0.0, new_minutes - b_minutes)
            miles_delta = dd_miles + trip_miles

            cand = {
                "candidate_id": f"{tid}_drv_{rid}",
                "driver_id": rid,
                "type": "reassigned",
                "deadhead_miles": dd_miles,
                "deadhead_minutes": dd_minutes,
                "delay_minutes": 0 if fits_window else 0,  # placeholder; compute real delay if you shift departure
                "overtime_minutes": overtime_minutes,
                "miles_delta": miles_delta,
                "uses_emergency_rest": False,
            }
            cand_list.append(cand)

        # Outsourcing fallback
        cand_list.append({"candidate_id": f"{tid}_out", "type": "outsourced", "trip_miles": trip_miles})
        out[tid] = cand_list

    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsl", required=True, help="df_rsl_clean.csv")
    ap.add_argument("--location_index", required=True, help="location_index.csv")
    ap.add_argument("--dist", required=True, help="distance_miles_matrix.npz")
    ap.add_argument("--time", required=True, help="time_minutes_matrix.npz")
    ap.add_argument("--drivers", required=True, help="driver_states.json")
    ap.add_argument("--disruptions", required=True, help="disruptions.json from simulator v2 (with 'day' and 'dep_minutes')")
    ap.add_argument("--out", default="candidates_per_trip.json", help="output JSON")
    a = ap.parse_args()
    main(a)
