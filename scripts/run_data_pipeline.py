#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from src.runtime import configure_logging, run_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified data pipeline entrypoint")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "/data"), help="Dataset directory")
    parser.add_argument("--mph", type=float, default=28.0, help="Fallback matrix speed used by quick matrix builder")
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="Skip backend /admin/reload call at the end",
    )
    parser.add_argument(
        "--reload-url",
        default=os.getenv("BACKEND_RELOAD_URL", "http://backend:8000/admin/reload"),
        help="Backend reload URL",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def call_reload(url: str, logger) -> None:
    request = Request(url=url, method="POST")
    try:
        with urlopen(request, timeout=10) as response:
            logger.info("Backend reload status: %s", response.status)
    except Exception as exc:
        logger.warning("Backend reload skipped/failed: %s", exc)


def main() -> None:
    args = parse_args()
    logger = configure_logging("pipeline")

    root = repo_root()
    data_dir = Path(args.data_dir).expanduser().resolve()
    input_csv = data_dir / "df_rsl_clean.csv"
    location_index = data_dir / "location_index.csv"

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)

    python_exe = sys.executable

    logger.info("[1/4] Building location artifacts from RSL")
    run_step(
        [python_exe, str(root / "scripts" / "data_prep.py"), "--csv", str(input_csv), "--out", str(data_dir)],
        cwd=root,
        env=env,
        logger=logger,
    )

    logger.info("[2/4] Building quick proximity matrices")
    run_step(
        [
            python_exe,
            str(root / "scripts" / "quick_build_matrices.py"),
            "--locations_index",
            str(location_index),
            "--outdir",
            str(data_dir),
            "--mph",
            str(args.mph),
        ],
        cwd=root,
        env=env,
        logger=logger,
    )

    logger.info("[3/4] Building driver states")
    run_step(
        [
            python_exe,
            str(root / "scripts" / "driver_states_prep.py"),
            "--csv",
            str(input_csv),
            "--location_index",
            str(location_index),
            "--out",
            str(data_dir / "driver_states.json"),
        ],
        cwd=root,
        env=env,
        logger=logger,
    )

    logger.info("[4/4] Optional backend reload")
    if not args.skip_reload:
        call_reload(args.reload_url, logger)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
