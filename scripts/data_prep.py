#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

def parse_hms_to_minutes(x):
    try:
        h,m,s = str(x).split(":")
        return int(h)*60 + int(m) + int(s)/60.0
    except Exception:
        return np.nan

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def main(args):
    COLS = {
        "from_name": "Mapped Name A",
        "to_name": "Mapped Name B",
        "from_pc": "From Postcode",
        "to_pc": "To Postcode",
        "from_lat": "From Lat",
        "from_lon": "From Long",
        "to_lat": "To Lat",
        "to_lon": "To Long",
        "elem_type": "Element Type",
        "elem_time": "Element Time",
        "leg_miles": "Leg Mileage",
        "duty_id": "Duty ID",
    }

    df = pd.read_csv(args.csv, dtype=str).applymap(lambda x: np.nan if isinstance(x,str) and x.strip().lower()=="no_data" else x)
    for col in [COLS["from_lat"], COLS["from_lon"], COLS["to_lat"], COLS["to_lon"], COLS["leg_miles"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["elem_minutes"] = df[COLS["elem_time"]].apply(parse_hms_to_minutes)

    def coalesce(row, name_col, pc_col):
        v = row.get(name_col)
        if pd.isna(v) or v=="":
            v = row.get(pc_col)
        return v

    df["from_name_norm"] = df.apply(lambda r: coalesce(r, COLS["from_name"], COLS["from_pc"]), axis=1)
    df["to_name_norm"] = df.apply(lambda r: coalesce(r, COLS["to_name"], COLS["to_pc"]), axis=1)

    centers_from = df[[
        "from_name_norm", COLS["from_pc"], COLS["from_lat"], COLS["from_lon"]
    ]].drop_duplicates().rename(columns={
        "from_name_norm":"name", COLS["from_pc"]:"postcode",
        COLS["from_lat"]:"lat", COLS["from_lon"]:"lon"
    })
    centers_to = df[[
        "to_name_norm", COLS["to_pc"], COLS["to_lat"], COLS["to_lon"]
    ]].drop_duplicates().rename(columns={
        "to_name_norm":"name", COLS["to_pc"]:"postcode",
        COLS["to_lat"]:"lat", COLS["to_lon"]:"lon"
    })
    centers = pd.concat([centers_from, centers_to], ignore_index=True).drop_duplicates(subset=["name"])
    centers = centers.loc[~centers["name"].isna()].copy()
    centers["center_id"] = pd.factorize(centers["name"])[0]
    centers = centers.sort_values("center_id").reset_index(drop=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    centers.to_csv(out / "centers.csv", index=False)
    centers[["name","center_id"]].to_csv(out / "location_index.csv", index=False)
    name_to_id = dict(zip(centers["name"], centers["center_id"]))
    df["from_id"] = df["from_name_norm"].map(name_to_id)
    df["to_id"] = df["to_name_norm"].map(name_to_id)
    travel = df[df[COLS["elem_type"]].str.upper() == "TRAVEL"].copy()
    travel = travel.dropna(subset=["from_id","to_id"])
    travel["from_id"] = travel["from_id"].astype(int)
    travel["to_id"] = travel["to_id"].astype(int)

    valid = travel.copy()
    valid = valid[valid[COLS["leg_miles"]].notna() & valid["elem_minutes"].notna()]
    valid = valid[(valid["elem_minutes"] > 0) & (valid[COLS["leg_miles"]] > 0)]
    valid["mph"] = valid[COLS["leg_miles"]] / (valid["elem_minutes"]/60.0)
    valid = valid[(valid["mph"] >= 10) & (valid["mph"] <= 65)]
    typical_mph = float(valid["mph"].median()) if len(valid) else 40.0

    legs = travel.groupby(["from_id","to_id"]).agg(
        miles=(COLS["leg_miles"], "mean"),
        minutes=("elem_minutes", "mean")
    ).reset_index()

    N = len(centers)
    dist_mat = np.full((N,N), np.nan, dtype=float)
    time_mat = np.full((N,N), np.nan, dtype=float)

    for _, r in legs.iterrows():
        i, j = int(r["from_id"]), int(r["to_id"])
        if np.isfinite(r["miles"]): dist_mat[i,j] = r["miles"]
        if np.isfinite(r["minutes"]): time_mat[i,j] = r["minutes"]

    for i in range(N):
        for j in range(i, N):
            a, b = dist_mat[i,j], dist_mat[j,i]
            if np.isfinite(a) and np.isfinite(b):
                avg = 0.5*(a+b); dist_mat[i,j] = dist_mat[j,i] = avg
            elif np.isfinite(a) and not np.isfinite(b):
                dist_mat[j,i] = a
            elif np.isfinite(b) and not np.isfinite(a):
                dist_mat[i,j] = b
            a, b = time_mat[i,j], time_mat[j,i]
            if np.isfinite(a) and np.isfinite(b):
                avg = 0.5*(a+b); time_mat[i,j] = time_mat[j,i] = avg
            elif np.isfinite(a) and not np.isfinite(b):
                time_mat[j,i] = a
            elif np.isfinite(b) and not np.isfinite(a):
                time_mat[i,j] = b

    lat = centers["lat"].to_numpy(); lon = centers["lon"].to_numpy()
    for i in range(N):
        for j in range(N):
            if i==j:
                dist_mat[i,j] = 0.0; time_mat[i,j] = 0.0; continue
            if not np.isfinite(dist_mat[i,j]):
                if np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(lat[j]) and np.isfinite(lon[j]):
                    dist_mat[i,j] = haversine_miles(lat[i], lon[i], lat[j], lon[j])
            if not np.isfinite(time_mat[i,j]) and np.isfinite(dist_mat[i,j]):
                time_mat[i,j] = (dist_mat[i,j] / typical_mph) * 60.0

    if np.isnan(dist_mat).sum():
        maxd = np.nanmax(dist_mat[np.isfinite(dist_mat)]) if np.isfinite(dist_mat).any() else 0.0
        dist_mat = np.nan_to_num(dist_mat, nan=maxd)
    if np.isnan(time_mat).sum():
        maxt = np.nanmax(time_mat[np.isfinite(time_mat)]) if np.isfinite(time_mat).any() else 0.0
        time_mat = np.nan_to_num(time_mat, nan=maxt)

    np.savez_compressed(out / "distance_miles_matrix.npz", matrix=dist_mat)
    np.savez_compressed(out / "time_minutes_matrix.npz", matrix=time_mat)

    print("Typical mph:", typical_mph)
    print("Saved:", (out / "distance_miles_matrix.npz").resolve())
    print("Saved:", (out / "time_minutes_matrix.npz").resolve())
    print("Saved:", (out / "centers.csv").resolve())
    print("Saved:", (out / "location_index.csv").resolve())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to cleaned private CSV")
    ap.add_argument("--out", default="out_private_data", help="Output directory")
    args = ap.parse_args()
    main(args)
