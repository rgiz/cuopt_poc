import argparse, math
import numpy as np, pandas as pd
from pathlib import Path

def build_from_location_index(li: pd.DataFrame, mph=45.0):
    li = li.copy()
    li.columns = [c.lower() for c in li.columns]
    names = li["name"].astype(str).str.upper().tolist()
    K = len(names)
    D = np.zeros((K,K), float)
    T = np.zeros((K,K), float)

    have_ll = {"lat","lon"}.issubset(li.columns)
    def hav_miles(a, b):
        R = 3958.7613
        la1, lo1, la2, lo2 = map(math.radians, [a["lat"],a["lon"],b["lat"],b["lon"]])
        dlat=la2-la1; dlon=lo2-lo1
        h=math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(h))

    for i in range(K):
        for j in range(K):
            if i == j: continue
            if have_ll and pd.notna(li.iloc[i]["lat"]) and pd.notna(li.iloc[i]["lon"]) \
                       and pd.notna(li.iloc[j]["lat"]) and pd.notna(li.iloc[j]["lon"]):
                miles = hav_miles(li.iloc[i], li.iloc[j])
            else:
                miles = 0.0
            D[i,j] = miles
            T[i,j] = miles / mph * 60.0
    return D, T

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations_index", default="data/private/location_index.csv")
    ap.add_argument("--outdir", default="data/private")
    ap.add_argument("--mph", type=float, default=45.0)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    li = pd.read_csv(args.locations_index)
    if not {"name","center_id"}.issubset(li.columns):
        raise SystemExit("location_index.csv must have columns: name,center_id (plus optional postcode,lat,lon)")

    D, T = build_from_location_index(li, mph=args.mph)
    np.savez_compressed(outdir/"distance_miles_matrix.npz", matrix=D)
    np.savez_compressed(outdir/"time_minutes_matrix.npz",  matrix=T)
    print("Wrote:", outdir/"distance_miles_matrix.npz", outdir/"time_minutes_matrix.npz")