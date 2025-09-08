# tests/conftest.py
import os
import json
from pathlib import Path
from importlib import reload

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def _env_test_data(monkeypatch, tmp_path: Path):
    """
    Prepare dummy test data unless TEST_DATASET=real.
    """
    if os.getenv("TEST_DATASET", "toy").lower() == "real":
        # Real data should be picked up by load_private_data() in app
        yield
        return

    """
    Prepare a tiny dataset under a temp DATA_ROOT so tests can use
    abstract locations 'A','B','C' without relying on the real bundle.
    """
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # ---- location_index.csv MUST have columns: name,center_id ----
    loc_index = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "center_id": [0, 1, 2],   # <— required by backend.main_miles
        }
    )
    loc_index.to_csv(data_root / "location_index.csv", index=False)

    # Optional: centers.csv (startup logs sometimes look for it)
    centers = pd.DataFrame(
        {
            "Name": ["A", "B", "C"],
            "Postcode": ["A1 1AA", "B1 1BB", "C1 1CC"],
            "Lat": [52.5, 52.6, 52.7],
            "Long": [-1.90, -1.91, -1.92],
        }
    )
    centers.to_csv(data_root / "centers.csv", index=False)

    # Tiny 3x3 time (minutes) and distance (miles) matrices
    t = np.array(
        [
            [0, 60, 90],
            [60, 0, 45],
            [90, 45, 0],
        ],
        dtype=np.float32,
    )
    d = np.array(
        [
            [0, 40, 70],
            [40, 0, 35],
            [70, 35, 0],
        ],
        dtype=np.float32,
    )
    np.savez(data_root / "time_minutes_matrix.npz", matrix=t)
    np.savez(data_root / "distance_miles_matrix.npz", matrix=d)

    # Minimal driver_states.json
    driver_states = {
        "drivers": {
            "D1": {
                "start_loc": "A",
                "start_min": 0,
                "end_min": 24 * 60,
                "elements": [
                    {
                        "is_travel": True,
                        "from": "A",
                        "to": "B",
                        "start_min": 9 * 60,
                        "end_min": 11 * 60,
                        "priority": 3,
                    }
                ],
            }
        }
    }
    (data_root / "driver_states.json").write_text(
        json.dumps(driver_states), encoding="utf-8"
    )

    PRIORITY_MAP = {
        "CONTAINER REPATRIATION": 4,
        "2C 48 MAIL": 2,
        "1C 24 MAIL": 1,
        "DELIVERY": 1,
        "travel_no_data": 3,
        "PF 48 PARCELS": 2,
        "PF 24 PARCELS": 1,
        "TRACKED COLLECTION": 2,
        "RDC PRESORT": 4,
        "HV RETURNS": 4,
        "TRACKED": 2,
        "COLLECTION": 2,
        "RDC 48 TRACKED": 2,
        "RDC 24 TRACKED": 1,
        "UNIT ONLY": 1,
        "EMPTY": 5,
        "INTERNATIONAL": 3,
        "D2D": 5,
        "RDC TRACKED": 2,
        "FLEX": 5,
        "ULD REPATRIATION": 4,
        "COMMERCIAL": 1,
        "RM RELAY": 1,
        "HV RETURNS DELIVERIES": 2,
        "DEFAULT": 3
    }
    (data_root / "priority_map.json").write_text(
        json.dumps(PRIORITY_MAP), encoding="utf-8"
    )

    # Point the app to our temp data dir
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("PRIVATE_DATA_DIR", str(data_root))

    # cuOpt URL won’t be hit in these tests; keep it harmless
    monkeypatch.setenv("CUOPT_URL", "http://dummy-cuopt:5000/v2")

    yield  # tmp_path is auto-cleaned


@pytest.fixture
def app(_env_test_data):
    # Import AFTER env vars/files so startup readers find our toy data
    import backend.main_miles as mm
    mm = reload(mm)
    return mm.app


@pytest.fixture
def client(app):
    c = TestClient(app)

    r = c.post("/admin/reload")

    if os.getenv("TEST_DATASET", "toy").lower() == "real":
        # In real mode, if data isn't present, fail loudly
        assert r.status_code == 200, f"[real mode] /admin/reload failed: {r.status_code} {r.text}"
    else:
        # In dummy mode, reload may return 204 (no driver states etc.)
        assert r.status_code in (200, 204), f"[toy mode] /admin/reload failed: {r.status_code} {r.text}"

    return c
