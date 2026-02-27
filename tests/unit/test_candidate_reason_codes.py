import numpy as np

from src.plan.candidates import generate_candidates
from src.plan.config import load_sla_windows


def test_candidates_include_reason_codes_when_generated():
    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 10], [10, 0]], dtype=float)
    tmat = np.array([[0, 20], [20, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D100": {
                    "home_center_id": 0,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 680,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                }
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 1.0,
        "overtime_cost_per_minute": 1.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 10
    req.trip_minutes = None
    req.trip_miles = None

    weekday, trip_minutes, trip_miles, candidates = generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        load_sla_windows(),
    )

    assert weekday == "Tue"
    assert trip_minutes > 0
    assert trip_miles > 0
    assert candidates
    assert all(c.reason_code for c in candidates)


def test_generates_overlap_prepickup_reason_code():
    loc2idx = {"A": 0, "B": 1, "C": 2}
    dist = np.array(
        [
            [0, 10, 8],
            [10, 0, 6],
            [8, 6, 0],
        ],
        dtype=float,
    )
    tmat = np.array(
        [
            [0, 20, 10],
            [20, 0, 12],
            [10, 12, 0],
        ],
        dtype=float,
    )

    data = {
        "driver_states": {
            "drivers": {
                "D200": {
                    "home_center_id": 2,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "C",
                            "to": "B",
                            "from_id": 2,
                            "to_id": 1,
                            "start_min": 600,
                            "end_min": 680,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                }
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 1.0,
        "overtime_cost_per_minute": 1.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
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

    assert any(c.reason_code == "RULE_OVERLAP_PREPICKUP" for c in candidates)


def test_generates_overlap_postdrop_reason_code():
    loc2idx = {"A": 0, "B": 1, "C": 2}
    dist = np.array(
        [
            [0, 10, 7],
            [10, 0, 5],
            [7, 5, 0],
        ],
        dtype=float,
    )
    tmat = np.array(
        [
            [0, 20, 12],
            [20, 0, 8],
            [12, 8, 0],
        ],
        dtype=float,
    )

    data = {
        "driver_states": {
            "drivers": {
                "D300": {
                    "home_center_id": 0,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "C",
                            "from_id": 0,
                            "to_id": 2,
                            "start_min": 600,
                            "end_min": 680,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                }
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 1.0,
        "overtime_cost_per_minute": 1.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
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

    assert any(c.reason_code == "RULE_OVERLAP_POSTDROP" for c in candidates)


def test_generates_nearby_bilateral_substitution_reason_code():
    loc2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    dist = np.array(
        [
            [0, 15, 8, 45],
            [15, 0, 20, 7],
            [8, 20, 0, 18],
            [45, 7, 18, 0],
        ],
        dtype=float,
    )
    tmat = np.array(
        [
            [0, 30, 12, 70],
            [30, 0, 35, 10],
            [12, 35, 0, 28],
            [70, 10, 28, 0],
        ],
        dtype=float,
    )

    data = {
        "driver_states": {
            "drivers": {
                "D400": {
                    "home_center_id": 2,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "C",
                            "to": "D",
                            "from_id": 2,
                            "to_id": 3,
                            "start_min": 620,
                            "end_min": 700,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                }
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 1.0,
        "overtime_cost_per_minute": 1.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
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

    assert any(c.reason_code == "RULE_NEARBY_BILATERAL_SUBSTITUTION" for c in candidates)


def test_nearby_bilateral_substitution_not_generated_when_far():
    loc2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    dist = np.array(
        [
            [0, 15, 80, 90],
            [15, 0, 85, 95],
            [80, 85, 0, 18],
            [90, 95, 18, 0],
        ],
        dtype=float,
    )
    tmat = np.array(
        [
            [0, 30, 120, 130],
            [30, 0, 125, 135],
            [120, 125, 0, 28],
            [130, 135, 28, 0],
        ],
        dtype=float,
    )

    data = {
        "driver_states": {
            "drivers": {
                "D401": {
                    "home_center_id": 2,
                    "daily_windows": {"Tue": {"start_min": 500, "end_min": 900, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "C",
                            "to": "D",
                            "from_id": 2,
                            "to_id": 3,
                            "start_min": 620,
                            "end_min": 700,
                            "priority": 3,
                            "Tue": 1,
                        }
                    ],
                }
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 1.0,
        "overtime_cost_per_minute": 1.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
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

    assert all(c.reason_code != "RULE_NEARBY_BILATERAL_SUBSTITUTION" for c in candidates)
