import numpy as np

from src.plan.candidates import generate_candidates
from src.plan.config import load_sla_windows


def test_append_ranking_uses_multicriteria_score_beyond_est_cost():
    loc2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    dist = np.array(
        [
            [0, 10, 5, 40],
            [10, 0, 15, 45],
            [5, 15, 0, 50],
            [40, 45, 50, 0],
        ],
        dtype=float,
    )
    tmat = np.array(
        [
            [0, 20, 8, 60],
            [20, 0, 25, 65],
            [8, 25, 0, 70],
            [60, 65, 70, 0],
        ],
        dtype=float,
    )

    data = {
        "driver_states": {
            "drivers": {
                "D_NEAR": {
                    "home_center_id": 2,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "C",
                            "to": "C",
                            "from_id": 2,
                            "to_id": 2,
                            "start_min": 620,
                            "end_min": 680,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                },
                "D_FAR": {
                    "home_center_id": 3,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "D",
                            "to": "D",
                            "from_id": 3,
                            "to_id": 3,
                            "start_min": 620,
                            "end_min": 680,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                },
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 0.0,
        "overtime_cost_per_minute": 0.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
        "rank_deadhead_miles_weight": 1.0,
        "rank_deadhead_minutes_weight": 0.1,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = None
    req.trip_miles = None

    _, _, _, candidates = generate_candidates(req, data, matrices, cfg, {}, load_sla_windows())

    append_candidates = [c for c in candidates if "::append" in c.candidate_id]
    assert len(append_candidates) >= 2
    assert append_candidates[0].driver_id == "D_NEAR"
