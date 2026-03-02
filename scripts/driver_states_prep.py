#!/usr/bin/env python3
"""
Driver duty windows from df_rsl_clean (STRICT logic per your rules)

Outputs:
- driver_states.json (consumed by backend)
- driver_states.csv (QA summary)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.driver_states_builder import build_driver_states
from src.rsl_helpers import find_day_cols


def _resolve_priority_map_for_output(out_json_path: Path) -> dict[str, Any]:
    candidate_paths = [
        out_json_path.parent / "priority_map.json",
        out_json_path.parent.parent / "priority_map.json",
        Path("data/private/active/priority_map.json"),
        Path("data/priority_map.json"),
    ]

    for path in candidate_paths:
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
                print(f"WARNING: Priority map at {path} is not an object", file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: Failed to parse priority map at {path}: {exc}", file=sys.stderr)
            continue
    return {}


def main(args):
    df = pd.read_csv(args.csv, dtype=str)
    out_json = Path(args.out)

    day_cols = find_day_cols(df)
    if not day_cols:
        print("WARNING: No day-of-week columns detected. Assuming all days.", file=sys.stderr)

    priority_map = _resolve_priority_map_for_output(out_json)
    if priority_map:
        print(f"Loaded priority map entries: {len(priority_map)}")
    else:
        print("WARNING: No priority map found; priorities may default to 3", file=sys.stderr)

    states, qa_df = build_driver_states(df, args.location_index, priority_map=priority_map)

    payload = {
        "schema_version": "2.0",
        "producer": "scripts/driver_states_prep.py",
        "drivers": states,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qa_df.to_csv(out_json.with_suffix(".csv"), index=False, encoding="utf-8")
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
