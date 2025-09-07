import numpy as np
from src.plan.candidates import generate_candidates
from src.plan.config import load_priority_map, load_sla_windows

def test_generate_candidates_minimum_smoke():
    # Small 3-node network: A,B,C
    loc2idx = {"A":0, "B":1, "C":2}
    dist = np.array([[0,50,90],[50,0,40],[90,40,0]], dtype=float)
    tmat = np.array([[0,60,110],[60,0,50],[110,50,0]], dtype=float)

    priority = load_priority_map()
    sla = load_sla_windows()  # ok if empty

    DATA = {
        "distance": dist,
        "time": tmat,
        "location_to_index": loc2idx,
        "driver_states": {"drivers": {
            "D1":{"elements":[{"is_travel":True,"from":"A","to":"B","start_min":9,"end_min":11,"priority":3}]}
        }},
        "locations_df": None,
    }
    M = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {"deadhead_cost_per_mile": 1.0, "overtime_cost_per_minute": 1.0, "reassignment_admin_cost": 10.0}

    # lightweight request object
    Req = type("Req", (), {})
    req = Req()
    req.start_location="A"
    req.end_location="B"
    req.mode="depart_after"
    req.when_local="2025-09-02T10:30"
    req.priority=2
    req.top_n=5
    req.trip_minutes=None
    req.trip_miles=None

    weekday, trip_minutes, trip_miles, cands = generate_candidates(req, DATA, M, cfg, {}, sla)
    assert weekday in ("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
    assert isinstance(trip_minutes, float)
    assert isinstance(trip_miles, float)
    assert isinstance(cands, list)
