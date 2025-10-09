#!/usr/bin/env python3
# scripts/data_prep.py
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.geocode_uk import enrich_locations

# --------- helpers ---------
def parse_hms_to_minutes(x):
    try:
        h, m, s = str(x).split(":")
        return int(h)*60 + int(m) + int(int(float(s))/60*60)
    except Exception:
        return np.nan

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

# --------- main pipeline ---------
def main(args):
    # Map your input column names
    COLS = {
        "from_name": "Mapped Name A",
        "to_name":   "Mapped Name B",
        "from_pc":   "From Postcode",
        "to_pc":     "To Postcode",
        "from_lat":  "From Lat",
        "from_lon":  "From Long",
        "to_lat":    "To Lat",
        "to_lon":    "To Long",
        "elem_type": "Element Type",
        "elem_time": "Element Time",
        "leg_miles": "Leg Mileage",
        "duty_id":   "Duty ID",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load the source CSV
    df = pd.read_csv(args.csv, dtype=str).applymap(
        lambda x: np.nan if isinstance(x, str) and x.strip().lower() == "no_data" else x
    )

    # 2) Coerce numeric columns
    for col in [COLS["from_lat"], COLS["from_lon"], COLS["to_lat"], COLS["to_lon"], COLS["leg_miles"]]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if COLS["elem_time"] in df.columns:
        df["elem_minutes"] = df[COLS["elem_time"]].apply(parse_hms_to_minutes)
    else:
        df["elem_minutes"] = np.nan

    # 3) Coalesce a usable "name" for endpoints (prefer mapped name, else postcode)
    def coalesce(row, name_col, pc_col):
        v = row.get(name_col)
        if pd.isna(v) or v == "":
            v = row.get(pc_col)
        return v

    df["from_name_norm"] = df.apply(lambda r: coalesce(r, COLS["from_name"], COLS["from_pc"]), axis=1)
    df["to_name_norm"]   = df.apply(lambda r: coalesce(r, COLS["to_name"],   COLS["to_pc"]), axis=1)

    # 4) Build raw centers table (unique locations)
    centers_from = df[[
        "from_name_norm", COLS["from_pc"], COLS["from_lat"], COLS["from_lon"]
    ]].drop_duplicates().rename(columns={
        "from_name_norm": "name", COLS["from_pc"]: "postcode",
        COLS["from_lat"]: "lat",  COLS["from_lon"]: "lon"
    })
    centers_to = df[[
        "to_name_norm", COLS["to_pc"], COLS["to_lat"], COLS["to_lon"]
    ]].drop_duplicates().rename(columns={
        "to_name_norm": "name", COLS["to_pc"]: "postcode",
        COLS["to_lat"]: "lat",  COLS["to_lon"]: "lon"
    })
    centers = pd.concat([centers_from, centers_to], ignore_index=True)
    centers = centers.loc[~centers["name"].isna()].copy()
    centers = centers.drop_duplicates(subset=["name"]).reset_index(drop=True)

    # 5) Enrich with geocoding (fills missing lat/lon from postcode)
    centers_enriched, issues = enrich_locations(
        centers.rename(columns={"name": "location_id"}),  # temporarily treat 'name' as id
        id_col="location_id", lat_col="lat", lon_col="lon", pc_col="postcode" if "postcode" in centers.columns else None
    )

    # Write enrichment report
    issues.to_json(out / "prep_report_locations.json", orient="records", lines=True)

    # 6) Drop unresolved locations (no lat/lon after enrichment)
    locs = centers_enriched.dropna(subset=["lat", "lon"]).copy()

    # 7) Create stable IDs & index
    locs = locs.reset_index(drop=True)
    locs["matrix_index"] = pd.factorize(locs["location_id"])[0]
    locs = locs.sort_values("matrix_index").reset_index(drop=True)

    # ✅ UPDATED: Output location_index.csv with correct column names
    location_index = locs[["location_id", "matrix_index", "postcode", "lat", "lon"]].copy()
    location_index = location_index.rename(columns={
        "location_id": "name",
        "matrix_index": "center_id"
    })
    # Ensure correct column order expected by all scripts
    location_index = location_index[["name", "center_id", "postcode", "lat", "lon"]]
    location_index.to_csv(out / "location_index.csv", index=False)
    
    # ✅ UPDATED: Output locations.csv (optional, for geo lookups)
    locations_out = locs[["location_id", "postcode", "lat", "lon"]].copy()
    locations_out = locations_out.rename(columns={"location_id": "name"})
    locations_out.to_csv(out / "locations.csv", index=False)
    
    # ✅ REMOVED: Delete centers.csv generation (redundant)
    # The old code that wrote centers.csv is no longer needed

    # 8) Map the raw travel legs to indices
    name_to_id = dict(zip(locs["location_id"], locs["matrix_index"]))

    # Reconstruct from_id/to_id:
    # first map from the normalized strings used when deduping locations
    # (Use original normalized names to look up the enriched location_id)
    # Build a reverse lookup: original 'name' -> canonical 'location_id'
    # Here, since we used the 'name' as initial location_id, we can map directly:
    df["from_id"] = df["from_name_norm"].map(lambda n: name_to_id.get(n))
    df["to_id"]   = df["to_name_norm"].map(lambda n: name_to_id.get(n))

    travel = df[df[COLS["elem_type"]].str.upper() == "TRAVEL"].copy()
    travel = travel.dropna(subset=["from_id", "to_id"])
    travel["from_id"] = travel["from_id"].astype(int)
    travel["to_id"]   = travel["to_id"].astype(int)

    # 9) Quality filter on leg samples for mph
    valid = travel.copy()
    valid = valid[valid[COLS["leg_miles"]].notna() & valid["elem_minutes"].notna()]
    valid = valid[(valid["elem_minutes"] > 0) & (valid[COLS["leg_miles"]] > 0)]
    valid["mph"] = valid[COLS["leg_miles"]] / (valid["elem_minutes"] / 60.0)
    valid = valid[(valid["mph"] >= 10) & (valid["mph"] <= 65)]
    typical_mph = float(valid["mph"].median()) if len(valid) else (args.mph if args.mph else 40.0)

    # 10) Aggregate legs to OD means
    legs = travel.groupby(["from_id", "to_id"]).agg(
        miles=(COLS["leg_miles"], "mean"),
        minutes=("elem_minutes", "mean"),
    ).reset_index()

    # 11) Build matrices with symmetry and haversine/time fallbacks
    N = len(locs)
    dist_mat = np.full((N, N), np.nan, dtype=float)
    time_mat = np.full((N, N), np.nan, dtype=float)

    for _, r in legs.iterrows():
        i, j = int(r["from_id"]), int(r["to_id"])
        if np.isfinite(r["miles"]):   dist_mat[i, j] = r["miles"]
        if np.isfinite(r["minutes"]): time_mat[i, j] = r["minutes"]

    # symmetry + one-sided fill
    for i in range(N):
        for j in range(i, N):
            a, b = dist_mat[i, j], dist_mat[j, i]
            if np.isfinite(a) and np.isfinite(b):
                avg = 0.5 * (a + b); dist_mat[i, j] = dist_mat[j, i] = avg
            elif np.isfinite(a) and not np.isfinite(b):
                dist_mat[j, i] = a
            elif np.isfinite(b) and not np.isfinite(a):
                dist_mat[i, j] = b

            a, b = time_mat[i, j], time_mat[j, i]
            if np.isfinite(a) and np.isfinite(b):
                avg = 0.5 * (a + b); time_mat[i, j] = time_mat[j, i] = avg
            elif np.isfinite(a) and not np.isfinite(b):
                time_mat[j, i] = a
            elif np.isfinite(b) and not np.isfinite(a):
                time_mat[i, j] = b

    # haversine fallback for missing distances, then time via typical mph
    lat = locs["lat"].to_numpy()
    lon = locs["lon"].to_numpy()
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_mat[i, j] = 0.0
                time_mat[i, j] = 0.0
                continue
            if not np.isfinite(dist_mat[i, j]):
                if np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(lat[j]) and np.isfinite(lon[j]):
                    dist_mat[i, j] = haversine_miles(lat[i], lon[i], lat[j], lon[j])
            if not np.isfinite(time_mat[i, j]) and np.isfinite(dist_mat[i, j]):
                time_mat[i, j] = (dist_mat[i, j] / max(1e-3, typical_mph)) * 60.0

    # final NaN fill (worst-observed value to avoid zero-cheating)
    if np.isnan(dist_mat).any():
        maxd = np.nanmax(dist_mat[np.isfinite(dist_mat)]) if np.isfinite(dist_mat).any() else 0.0
        dist_mat = np.nan_to_num(dist_mat, nan=maxd)
    if np.isnan(time_mat).any():
        maxt = np.nanmax(time_mat[np.isfinite(time_mat)]) if np.isfinite(time_mat).any() else 0.0
        time_mat = np.nan_to_num(time_mat, nan=maxt)

    # 12) Save artifacts (unnamed arrays -> np.load(... )["arr_0"])
    np.savez_compressed(out / "distance_miles_matrix.npz", dist_mat)
    np.savez_compressed(out / "time_minutes_matrix.npz", time_mat)

    # 13) Logs
    print(f"Typical mph (median of valid legs): {typical_mph:.1f}")
    print("Saved:", (out / "distance_miles_matrix.npz").resolve())
    print("Saved:", (out / "time_minutes_matrix.npz").resolve())
    print("Saved:", (out / "locations.csv").resolve())
    print("Saved:", (out / "location_index.csv").resolve())
    print("Saved:", (out / "prep_report_locations.json").resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to cleaned private CSV")
    ap.add_argument("--out", default="/data", help="Output directory (mounted /data recommended)")
    ap.add_argument("--mph", type=float, default=None, help="Fallback typical mph (if no valid legs)")
    args = ap.parse_args()
    main(args)
