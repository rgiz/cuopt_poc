#!/usr/bin/env python3
"""
Driver duty windows from df_rsl_clean (STRICT rules)

This script is retained as a CLI alias and delegates to the shared builder.
"""

import argparse

from scripts.driver_states_prep import main


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to df_rsl_clean-like CSV")
    ap.add_argument("--location_index", default="", help="Optional centers mapping (location_index.csv)")
    ap.add_argument("--out", default="driver_states.json", help="Output JSON path")
    args = ap.parse_args()
    main(args)
