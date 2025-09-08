import os
import pytest
import pandas as pd
from math import radians, cos, sin, asin, sqrt

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_DATASET", "toy").lower() != "real",
    reason="Only runs with TEST_DATASET=real"
)

def get_coords(name, centers):
    row = centers[centers["name"].str.upper() == name.upper()]
    assert not row.empty, f"Missing location: {name}"
    return row.iloc[0]["lat"], row.iloc[0]["lon"]

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def test_candidates_are_local_real_data(client):
    origin = "BIRMINGHAM MAIL CENTRE"
    payload = {
        "start_location": origin,
        "end_location": "MIDLANDS SUPER HUB",
        "mode": "depart_after",
        "when_local": "2025-09-01T09:00",
        "priority": 2,
        "top_n": 5
    }
    r = client.post("/plan/candidates", json=payload)
    assert r.status_code == 200
    cands = r.json()["candidates"]
    assert cands, "No candidates returned"

    centers = pd.read_csv(os.environ["PRIVATE_DATA_DIR"] + "/centers.csv")
    origin_lat, origin_lon = get_coords(origin, centers)

    for cand in cands:
        lat, lon = get_coords(cand["from"], centers)
        distance = haversine(origin_lat, origin_lon, lat, lon)
        assert distance < 50, f"Candidate {cand['from']} is too far: {distance:.1f} miles"

def test_solve_multi_valid_cascades(client):
    payload = {
        "start_location": "BIRMINGHAM MAIL CENTRE",
        "end_location": "MIDLANDS SUPER HUB",
        "mode": "depart_after",
        "when_local": "2025-09-01T08:00",
        "priority": 1,
        "top_n_per_step": 3,
        "max_cascades": 2,
        "max_drivers_affected": 2,
        "max_solutions": 2,
        "use_cuopt": False
    }
    r = client.post("/plan/solve_multi", json=payload)
    assert r.status_code == 200
    solutions = r.json()["solutions"]
    assert solutions, "No solutions returned"

    for sol in solutions:

        driver_ids = set()

        actions = sol.get("actions")
        assert actions is not None and len(actions) >= 2

        before = sol.get("driver_schedules_before")
        after = sol.get("driver_schedules_after")
        assert before is not None and after is not None, "Missing before/after schedules"

        for act in actions:
            assert all(k in act for k in ["driver_id", "from", "to", "start_min", "end_min", "priority"])
            driver_ids.add(act["driver_id"])

        for d in driver_ids:
            assert d in before, f"Missing before-schedule for {d}"
            assert d in after, f"Missing after-schedule for {d}"

        displaced = sol.get("displaced_jobs", [])
        reassigned_ids = [a.get("job_id") for a in actions if a.get("job_id")]
        for dj in displaced:
            assert dj["job_id"] in reassigned_ids, f"Displaced job {dj['job_id']} not reassigned"

# Placeholders for negative test ideas
def test_invalid_location_rejected(client):
    payload = {
        "start_location": "INVALID DEPOT",
        "end_location": "MIDLANDS SUPER HUB",
        "mode": "depart_after",
        "when_local": "2025-09-01T09:00",
        "priority": 2,
        "top_n": 5
    }
    r = client.post("/plan/candidates", json=payload)
    assert r.status_code in (400, 422, 500)

def test_empty_result_when_no_candidates(client):
    # Assuming very late time or distant location produces no results
    payload = {
        "start_location": "BIRMINGHAM MAIL CENTRE",
        "end_location": "MIDLANDS SUPER HUB",
        "mode": "depart_after",
        "when_local": "2030-01-01T04:00",
        "priority": 2,
        "top_n": 5
    }
    r = client.post("/plan/candidates", json=payload)
    assert r.status_code == 200
    candidates = r.json()["candidates"]
    assert all(c["delay_minutes"] > 1000 or "fallback" in c["candidate_id"] for c in candidates)

