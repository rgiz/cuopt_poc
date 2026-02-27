import argparse
import numpy as np, pandas as pd
from pathlib import Path
from src.matrix_builder import build_from_location_index

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