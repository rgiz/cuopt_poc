import importlib

import numpy as np


def _reload_with_strict_legal(monkeypatch):
    monkeypatch.setenv("ENABLE_STRICT_LEGAL_CONSTRAINTS", "true")
    import src.plan.config as plan_config
    import src.plan.candidates as candidates_mod

    importlib.reload(plan_config)
    candidates_mod = importlib.reload(candidates_mod)
    return candidates_mod, plan_config


def test_strict_legal_daily_driving_limit_marks_candidate_infeasible(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 20], [20, 0]], dtype=float)
    tmat = np.array([[0, 30], [30, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D_LIMIT": {
                    "home_center_id": 0,
                    "daily_windows": {"Tue": {"start_min": 0, "end_min": 1440, "crosses_midnight": False}},
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "B",
                            "to": "A",
                            "from_id": 1,
                            "to_id": 0,
                            "start_min": 60,
                            "end_min": 560,
                            "duration_min": 500,
                            "priority": 3,
                            "Tue": 1,
                        },
                        {
                            "element_type": "Meal Relief",
                            "start_min": 560,
                            "end_min": 605,
                            "duration_min": 45,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 650,
                            "duration_min": 30,
                            "priority": 3,
                            "Tue": 1,
                        },
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
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_min_break_minutes": 45,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 120
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    swap_leg = next(c for c in candidates if c.candidate_id.startswith("D_LIMIT::swap_leg@"))
    assert swap_leg.feasible_hard is False


def test_meal_relief_resets_continuous_driving_limit(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1, "C": 2}
    dist = np.array(
        [
            [0, 20, 10],
            [20, 0, 15],
            [10, 15, 0],
        ],
        dtype=float,
    )
    tmat = np.array(
        [
            [0, 30, 20],
            [30, 0, 25],
            [20, 25, 0],
        ],
        dtype=float,
    )

    driver_with_break = {
        "home_center_id": 2,
        "daily_windows": {"Tue": {"start_min": 0, "end_min": 1440, "crosses_midnight": False}},
        "elements": [
            {
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": "C",
                "to": "A",
                "from_id": 2,
                "to_id": 0,
                "start_min": 300,
                "end_min": 540,
                "duration_min": 240,
                "priority": 3,
                "Tue": 1,
            },
            {
                "element_type": "Meal Relief",
                "start_min": 540,
                "end_min": 585,
                "duration_min": 45,
                "Tue": 1,
            },
            {
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": "A",
                "to": "B",
                "from_id": 0,
                "to_id": 1,
                "start_min": 620,
                "end_min": 650,
                "duration_min": 30,
                "priority": 3,
                "Tue": 1,
            },
        ],
    }

    driver_without_break = {
        "home_center_id": 2,
        "daily_windows": {"Tue": {"start_min": 0, "end_min": 1440, "crosses_midnight": False}},
        "elements": [
            {
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": "C",
                "to": "A",
                "from_id": 2,
                "to_id": 0,
                "start_min": 300,
                "end_min": 540,
                "duration_min": 240,
                "priority": 3,
                "Tue": 1,
            },
            {
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": "A",
                "to": "B",
                "from_id": 0,
                "to_id": 1,
                "start_min": 620,
                "end_min": 650,
                "duration_min": 30,
                "priority": 3,
                "Tue": 1,
            },
        ],
    }

    data = {
        "driver_states": {
            "drivers": {
                "D_BREAK": driver_with_break,
                "D_NOBREAK": driver_without_break,
            }
        }
    }

    matrices = {"dist": dist, "time": tmat, "loc2idx": loc2idx}
    cfg = {
        "deadhead_cost_per_mile": 1.0,
        "overtime_cost_per_minute": 1.0,
        "reassignment_admin_cost": 10.0,
        "max_duty_minutes": 13 * 60,
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_min_break_minutes": 45,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 120
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    by_id = {c.candidate_id: c for c in candidates if "::swap_leg@" in c.candidate_id}

    assert by_id["D_BREAK::swap_leg@620"].feasible_hard is True
    assert by_id["D_NOBREAK::swap_leg@620"].feasible_hard is False


def test_strict_legal_weekly_driving_limit_marks_candidate_infeasible(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 20], [20, 0]], dtype=float)
    tmat = np.array([[0, 30], [30, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D_WEEK": {
                    "home_center_id": 0,
                    "week_drive_minutes": (56 * 60) - 20,
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 650,
                            "duration_min": 30,
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
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_max_weekly_driving_minutes": 56 * 60,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 60
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    swap_leg = next(c for c in candidates if c.candidate_id.startswith("D_WEEK::swap_leg@"))
    assert swap_leg.feasible_hard is False


def test_strict_legal_fortnight_driving_limit_marks_candidate_infeasible(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 20], [20, 0]], dtype=float)
    tmat = np.array([[0, 30], [30, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D_2WEEK": {
                    "home_center_id": 0,
                    "week_drive_minutes": (56 * 60) - 60,
                    "previous_week_drive_minutes": (34 * 60) + 31,
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 650,
                            "duration_min": 30,
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
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_max_weekly_driving_minutes": 56 * 60,
        "legal_max_fortnight_driving_minutes": 90 * 60,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 60
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    swap_leg = next(c for c in candidates if c.candidate_id.startswith("D_2WEEK::swap_leg@"))
    assert swap_leg.feasible_hard is False


def test_strict_legal_weekly_rest_periods_limit_marks_candidate_infeasible(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 20], [20, 0]], dtype=float)
    tmat = np.array([[0, 30], [30, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D_REST_PERIODS": {
                    "home_center_id": 0,
                    "consecutive_24h_periods_since_weekly_rest": 7,
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 650,
                            "duration_min": 30,
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
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_max_24h_periods_between_weekly_rests": 6,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 60
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    swap_leg = next(c for c in candidates if c.candidate_id.startswith("D_REST_PERIODS::swap_leg@"))
    assert swap_leg.feasible_hard is False


def test_strict_legal_uncompensated_reduced_rest_marks_candidate_infeasible(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 20], [20, 0]], dtype=float)
    tmat = np.array([[0, 30], [30, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D_REST_COMP": {
                    "home_center_id": 0,
                    "reduced_rest_compensation_minutes_due": 30,
                    "weekly_rest_last_duration_minutes": 24 * 60,
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 650,
                            "duration_min": 30,
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
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_allow_uncompensated_reduced_rest": False,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 60
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    swap_leg = next(c for c in candidates if c.candidate_id.startswith("D_REST_COMP::swap_leg@"))
    assert swap_leg.feasible_hard is False


def test_strict_legal_weekly_working_time_cap_marks_candidate_infeasible(monkeypatch):
    candidates_mod, plan_config = _reload_with_strict_legal(monkeypatch)

    loc2idx = {"A": 0, "B": 1}
    dist = np.array([[0, 20], [20, 0]], dtype=float)
    tmat = np.array([[0, 30], [30, 0]], dtype=float)

    data = {
        "driver_states": {
            "drivers": {
                "D_WORK_CAP": {
                    "home_center_id": 0,
                    "week_drive_minutes": 1000,
                    "week_work_minutes": (60 * 60) - 20,
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "from_id": 0,
                            "to_id": 1,
                            "start_min": 620,
                            "end_min": 650,
                            "duration_min": 30,
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
        "legal_max_daily_driving_minutes": 9 * 60,
        "legal_max_continuous_driving_minutes": int(4.5 * 60),
        "legal_max_weekly_driving_minutes": 56 * 60,
        "legal_max_fortnight_driving_minutes": 90 * 60,
        "legal_enforce_weekly_work_minutes": True,
        "legal_max_weekly_work_minutes": 60 * 60,
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.mode = "depart_after"
    req.when_local = "2025-09-02T10:30"
    req.priority = 2
    req.top_n = 20
    req.trip_minutes = 60
    req.trip_miles = 20

    _, _, _, candidates = candidates_mod.generate_candidates(
        req,
        data,
        matrices,
        cfg,
        {},
        plan_config.load_sla_windows(),
    )

    swap_leg = next(c for c in candidates if c.candidate_id.startswith("D_WORK_CAP::swap_leg@"))
    assert swap_leg.feasible_hard is False
