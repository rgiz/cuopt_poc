# tests/utils/build_real_mini_dataset.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

DEFAULT_NAMES = [
    "BIRMINGHAM MAIL CENTRE",
    "MIDLANDS SUPER HUB",
    "SOUTH MIDLANDS MAIL CENTRE",
]

def build_real_mini_dataset(data_root: Path, target_root: Path, names=None):
    """
    Slice the real matrices (distance_miles_matrix.npz, time_minutes_matrix.npz)
    and location_index.csv to just the selected locations.
    Falls back to names in DEFAULT_NAMES.
    """
    names = [str(n).upper().strip() for n in (names or DEFAULT_NAMES)]
    data_root = Path(data_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # Load full location index (must match your production loader)
    # Accept either 'location_index.csv' (preferred) or 'locations.csv' with a column of names.
    loc_index_path = data_root / "location_index.csv"
    if loc_index_path.exists():
        df = pd.read_csv(loc_index_path)
        col = "name"
    else:
        locs_csv = data_root / "locations.csv"
        if not locs_csv.exists():
            raise FileNotFoundError("No location_index.csv or locations.csv found in DATA_ROOT")
        df = pd.read_csv(locs_csv)
        # try common name columns
        for c in ["name","NAME","site_name","Site","site","location_id","Mapped Name A"]:
            if c in df.columns:
                col = c
                break
        else:
            raise ValueError("Could not find a column with location names in locations.csv")
        df[col] = df[col].astype(str).str.upper().str.strip()

    # Build mapping full -> idx
    names_full = df[col].astype(str).str.upper().str.strip().tolist()
    loc2idx = {n:i for i,n in enumerate(names_full)}

    # Confirm all requested names exist
    missing = [n for n in names if n not in loc2idx]
    if missing:
        raise ValueError(f"Requested test locations missing from DATA_ROOT: {missing}")

    idxs = [loc2idx[n] for n in names]

    # Load full matrices and slice
    dist_npz = np.load(data_root / "distance_miles_matrix.npz")
    time_npz = np.load(data_root / "time_minutes_matrix.npz")
    dist_full = dist_npz[list(dist_npz.files)[0]]
    time_full = time_npz[list(time_npz.files)[0]]

    dist = dist_full[np.ix_(idxs, idxs)]
    time = time_full[np.ix_(idxs, idxs)]

    # Save sliced matrices
    np.savez(target_root / "distance_miles_matrix.npz", dist)
    np.savez(target_root / "time_minutes_matrix.npz", time)

    # Write sliced location_index.csv in the correct order
    pd.DataFrame({"name": names}).to_csv(target_root / "location_index.csv", index=False)

    # Minimal driver_states for realism: add one real A->B leg at 10:00–12:00
    driver_states = {
        "drivers": {
            "D_REAL": {
                "start_min": 0,
                "end_min": 24*60,
                "elements": [
                    {"is_travel": True, "from": names[0], "to": names[1],
                     "start_min": 10*60, "end_min": 12*60, "priority": 3}
                ]
            }
        }
    }
    (target_root / "driver_states.json").write_text(json.dumps(driver_states), encoding="utf-8")

    # You can copy priority_map.json from your real DATA_ROOT if you want:
    pm = data_root / "priority_map.json"
    if pm.exists():
        (target_root / "priority_map.json").write_bytes(pm.read_bytes())
    else:
        (target_root / "priority_map.json").write_text("{}", encoding="utf-8")
