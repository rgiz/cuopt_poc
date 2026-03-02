import numpy as np
import src.plan.cascade_candidates as cascade_mod

from src.plan.cascade_candidates import (
    CascadeCandidateOut,
    _reconstruct_as_directed_replacement,
    _reconstruct_leg_replacement,
    _validate_schedule_continuity,
    _validate_home_base_return,
    _build_true_cascade_scaffold,
    _extract_displaced_task_from_strategy,
    _attempt_secondary_assignment_for_displaced,
    _determine_insertion_strategy,
    _enhanced_to_candidate_out,
)


def test_validate_schedule_continuity_detects_overlap():
    schedule = [
        {"element_type": "TRAVEL", "from": "A", "to": "B", "start_min": 100, "end_min": 140, "is_travel": True},
        {"element_type": "TRAVEL", "from": "B", "to": "C", "start_min": 130, "end_min": 170, "is_travel": True},
    ]

    valid, reason = _validate_schedule_continuity(schedule)

    assert valid is False
    assert reason == "time_overlap"


def test_reconstruct_as_directed_replacement_within_block():
    original = [
        {
            "element_type": "START FACILITY",
            "from": "A",
            "to": "A",
            "start_min": 480,
            "end_min": 490,
            "is_travel": False,
        },
        {
            "element_type": "AS DIRECTED",
            "from": "A",
            "to": "A",
            "start_min": 500,
            "end_min": 700,
            "is_travel": False,
            "priority": 3,
        },
        {
            "element_type": "END FACILITY",
            "from": "A",
            "to": "A",
            "start_min": 710,
            "end_min": 720,
            "is_travel": False,
        },
    ]

    strategy = {
        "type": "as_directed_replacement",
        "target_index": 1,
        "target_element": original[1],
        "new_service": {"from": "A", "to": "B"},
    }

    req = type("Req", (), {})()
    req.start_location = "A"
    req.end_location = "B"
    req.trip_minutes = 60
    req.priority = 1

    time_matrix = np.array([[0, 20], [20, 0]], dtype=float)
    dist_matrix = np.array([[0, 10], [10, 0]], dtype=float)
    matrices = {"time": time_matrix, "dist": dist_matrix, "loc2idx": {"A": 0, "B": 1}}

    reconstructed = _reconstruct_as_directed_replacement(strategy, original, req, matrices, {})

    assert reconstructed is not None
    changes = [e.get("changes") for e in reconstructed if e.get("changes")]
    assert any(ch == "AS_DIRECTED_REPLACED" for ch in changes)
    valid, reason = _validate_schedule_continuity(reconstructed)
    assert valid is True, reason


def test_validate_home_base_return_strict_fail_and_pass():
    schedule_not_home = [
        {"element_type": "TRAVEL", "from": "A", "to": "B", "start_min": 100, "end_min": 150, "is_travel": True},
    ]
    schedule_home = [
        {"element_type": "TRAVEL", "from": "A", "to": "A", "start_min": 100, "end_min": 150, "is_travel": True},
    ]

    ok_fail, reason_fail = _validate_home_base_return(schedule_not_home, "A", enforce_strict=True)
    ok_pass, reason_pass = _validate_home_base_return(schedule_home, "A", enforce_strict=True)

    assert ok_fail is False
    assert reason_fail == "not_returned_to_home_base"
    assert ok_pass is True
    assert reason_pass == "ok"


def test_true_cascade_scaffold_shape():
    chain = _build_true_cascade_scaffold("D1", max_depth=2)
    assert len(chain) == 1
    assert chain[0]["vehicle_id"] == "D1"
    assert chain[0]["status"] == "scaffold"


def test_extract_displaced_task_from_empty_replacement_strategy():
    strategy = {
        "type": "empty_replacement",
        "target_element": {
            "from": "A",
            "to": "B",
            "start_min": 100,
            "end_min": 160,
            "priority": 5,
            "element_type": "TRAVEL",
            "is_travel": True,
        },
    }

    displaced = _extract_displaced_task_from_strategy(strategy)

    assert displaced is not None
    assert displaced["from"] == "A"
    assert displaced["to"] == "B"
    assert displaced["priority"] == 5


def test_extract_displaced_task_from_leg_replacement_strategy():
    strategy = {
        "type": "leg_replacement",
        "target_element": {
            "from": "A",
            "to": "B",
            "start_min": 100,
            "end_min": 160,
            "priority": 4,
            "element_type": "TRAVEL",
            "is_travel": True,
        },
    }

    displaced = _extract_displaced_task_from_strategy(strategy)

    assert displaced is not None
    assert displaced["priority"] == 4


def test_determine_insertion_strategy_prefers_leg_replacement_for_exact_match():
    schedule = [
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "A",
            "to": "B",
            "start_min": 600,
            "end_min": 660,
            "priority": 4,
        }
    ]
    structure = {
        "start_facility": None,
        "end_facility": None,
        "as_directed_blocks": [],
        "meal_reliefs": [],
        "empty_legs": [],
        "home_base": None,
    }
    task_map = {"0": {"type": "NEW_SERVICE", "from": "A", "to": "B", "priority": 2, "duration": 60}}

    strategy = _determine_insertion_strategy(
        task_ids=["0"],
        arrival_stamps=[620],
        task_map=task_map,
        structure=structure,
        original_schedule=schedule,
        matrices={"time": np.array([[0, 20], [20, 0]], dtype=float), "dist": np.array([[0, 10], [10, 0]], dtype=float), "loc2idx": {"A": 0, "B": 1}},
    )

    assert strategy["type"] == "leg_replacement"


def test_reconstruct_leg_replacement_marks_change():
    original = [
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "A",
            "to": "B",
            "start_min": 600,
            "end_min": 660,
            "priority": 4,
        }
    ]
    strategy = {"target_index": 0, "target_element": original[0]}
    req = type("Req", (), {})()
    req.priority = 2

    reconstructed = _reconstruct_leg_replacement(strategy, original, req, {}, {})

    assert reconstructed[0]["changes"] == "LEG_REPLACED"
    assert reconstructed[0]["priority"] == 2


def test_reconstruct_overlap_prepickup_inserts_deadhead_and_preserves_continuity():
    original = [
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "MSH",
            "to": "BMC",
            "start_min": 300,
            "end_min": 360,
            "priority": 3,
        },
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "BMC",
            "to": "EXETER",
            "start_min": 360,
            "end_min": 480,
            "priority": 3,
        },
        {
            "element_type": "LOAD(ASSIST)",
            "is_travel": False,
            "from": "EXETER",
            "to": "EXETER",
            "start_min": 480,
            "end_min": 540,
            "priority": 3,
        },
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "EXETER",
            "to": "MSH",
            "start_min": 540,
            "end_min": 660,
            "priority": 3,
        },
        {
            "element_type": "END FACILITY",
            "is_travel": False,
            "from": "MSH",
            "to": "MSH",
            "start_min": 660,
            "end_min": 675,
            "priority": 3,
        },
    ]

    strategy = {
        "type": "overlap_prepickup_replacement",
        "target_index": 3,
        "target_element": original[3],
    }

    req = type("Req", (), {})()
    req.start_location = "BMC"
    req.end_location = "MSH"
    req.trip_minutes = 240
    req.priority = 3

    matrices = {
        "loc2idx": {"MSH": 0, "BMC": 1, "EXETER": 2},
        "time": np.array(
            [
                [0, 60, 120],
                [60, 0, 180],
                [120, 180, 0],
            ],
            dtype=float,
        ),
        "dist": np.array(
            [
                [0, 30, 80],
                [30, 0, 120],
                [80, 120, 0],
            ],
            dtype=float,
        ),
    }

    reconstructed = _reconstruct_leg_replacement(strategy, original, req, matrices, {})

    changes = [row.get("changes") for row in reconstructed]
    assert "OVERLAP_PREPICKUP_DEADHEAD" in changes
    assert "OVERLAP_PREPICKUP_REPLACED" in changes

    valid, reason = _validate_schedule_continuity(reconstructed)
    assert valid is True, reason


def test_determine_insertion_strategy_detects_overlap_prepickup_replacement():
    schedule = [
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "C",
            "to": "B",
            "start_min": 600,
            "end_min": 660,
            "priority": 4,
        }
    ]
    structure = {
        "start_facility": None,
        "end_facility": None,
        "as_directed_blocks": [],
        "meal_reliefs": [],
        "empty_legs": [],
        "home_base": None,
    }
    task_map = {"0": {"type": "NEW_SERVICE", "from": "A", "to": "B", "priority": 2, "duration": 60}}

    strategy = _determine_insertion_strategy(
        task_ids=["0"],
        arrival_stamps=[620],
        task_map=task_map,
        structure=structure,
        original_schedule=schedule,
        matrices={"time": np.array([[0, 20], [20, 0]], dtype=float), "dist": np.array([[0, 10], [10, 0]], dtype=float), "loc2idx": {"A": 0, "B": 1, "C": 2}},
    )

    assert strategy["type"] == "overlap_prepickup_replacement"


def test_determine_insertion_strategy_detects_nearby_substitution_replacement():
    schedule = [
        {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": "C",
            "to": "D",
            "start_min": 600,
            "end_min": 660,
            "priority": 4,
        }
    ]
    structure = {
        "start_facility": None,
        "end_facility": None,
        "as_directed_blocks": [],
        "meal_reliefs": [],
        "empty_legs": [],
        "home_base": None,
    }
    task_map = {"0": {"type": "NEW_SERVICE", "from": "A", "to": "B", "priority": 2, "duration": 60}}
    dist = np.array(
        [
            [0, 20, 8, 40],
            [20, 0, 25, 7],
            [8, 25, 0, 18],
            [40, 7, 18, 0],
        ],
        dtype=float,
    )

    strategy = _determine_insertion_strategy(
        task_ids=["0"],
        arrival_stamps=[620],
        task_map=task_map,
        structure=structure,
        original_schedule=schedule,
        matrices={"time": dist, "dist": dist, "loc2idx": {"A": 0, "B": 1, "C": 2, "D": 3}},
    )

    assert strategy["type"] == "nearby_substitution_replacement"


def test_extract_displaced_task_from_nearby_substitution_strategy():
    strategy = {
        "type": "nearby_substitution_replacement",
        "target_element": {
            "from": "C",
            "to": "D",
            "start_min": 100,
            "end_min": 160,
            "priority": 4,
            "element_type": "TRAVEL",
            "is_travel": True,
        },
    }

    displaced = _extract_displaced_task_from_strategy(strategy)

    assert displaced is not None
    assert displaced["from"] == "C"
    assert displaced["to"] == "D"
    assert displaced["priority"] == 4


def test_enhanced_candidate_out_includes_cascade_reason_metadata():
    cascade_result = CascadeCandidateOut(
        candidate_id="CUOPT_D1",
        primary_driver_id="D1",
        total_system_cost=123.45,
        drivers_affected=2,
        cascade_chain=[
            {"step": 1, "vehicle_id": "D2", "status": "assigned", "detail": "ok"},
            {"step": 2, "vehicle_id": "D3", "status": "assigned", "detail": "ok"},
        ],
        before_after_schedules={"D1": {"before": [], "after": []}, "D2": {"before": [], "after": []}},
        is_fully_feasible=True,
        uncovered_p4_tasks=[],
        disposed_p5_tasks=[],
    )

    out = _enhanced_to_candidate_out(cascade_result)

    assert out.reason_code == "CASCADE_MULTI_DRIVER"
    assert out.reason_detail is not None
    assert "chain_depth=2" in out.reason_detail
    assert "assigned_steps=2" in out.reason_detail


def test_enhanced_candidate_out_marks_partial_unresolved():
    cascade_result = CascadeCandidateOut(
        candidate_id="CUOPT_D1",
        primary_driver_id="D1",
        total_system_cost=140.0,
        drivers_affected=1,
        cascade_chain=[
            {"step": 1, "vehicle_id": "D2", "status": "blocked", "detail": "no_secondary_driver_fit"},
        ],
        before_after_schedules={"D1": {"before": [], "after": []}},
        is_fully_feasible=False,
        uncovered_p4_tasks=[{"from": "A", "to": "B", "priority": 3}],
        disposed_p5_tasks=[],
    )

    out = _enhanced_to_candidate_out(cascade_result)

    assert out.reason_code == "CASCADE_PARTIAL_UNRESOLVED"
    assert out.reason_detail is not None
    assert "blocked_steps=1" in out.reason_detail
    assert "uncovered_p4=1" in out.reason_detail


def test_attempt_secondary_assignment_for_displaced_assigns_other_driver():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "home_loc": "B",
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 120,
                            "end_min": 180,
                            "priority": 5,
                            "Tue": 1,
                        }
                    ]
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1},
        "time": np.array([[0, 20], [20, 0]], dtype=float),
        "dist": np.array([[0, 10], [10, 0]], dtype=float),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 100,
        "end_min": 160,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
    )

    assert assignment["assigned"] is True
    assert assignment["secondary_driver_id"] == "D2"
    assert assignment["chain_step"]["status"] == "assigned"


def test_attempt_secondary_assignment_for_displaced_recurses_to_depth_two():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 220,
                            "end_min": 280,
                            "priority": 4,
                            "Tue": 1,
                        }
                    ]
                },
                "D3": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 320,
                            "end_min": 360,
                            "priority": 4,
                            "Tue": 1,
                        }
                    ]
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1},
        "time": np.array([[0, 20], [20, 0]], dtype=float),
        "dist": np.array([[0, 10], [10, 0]], dtype=float),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 100,
        "end_min": 160,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
    )

    assert assignment["assigned"] is True
    assert len(assignment["chain_steps"]) == 2
    assert assignment["chain_steps"][0]["vehicle_id"] == "D2"
    assert assignment["chain_steps"][1]["vehicle_id"] == "D3"
    assert "D2" in assignment["driver_schedules"]
    assert "D3" in assignment["driver_schedules"]


def test_attempt_secondary_assignment_rejects_duty_append_fallback():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "elements": [
                        {
                            "element_type": "START FACILITY",
                            "is_travel": False,
                            "from": "A",
                            "to": "A",
                            "start_min": 500,
                            "end_min": 520,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "B",
                            "to": "C",
                            "start_min": 600,
                            "end_min": 660,
                            "priority": 4,
                            "Tue": 1,
                        },
                        {
                            "element_type": "END FACILITY",
                            "is_travel": False,
                            "from": "C",
                            "to": "C",
                            "start_min": 700,
                            "end_min": 715,
                            "Tue": 1,
                        },
                    ]
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1, "C": 2, "D": 3},
        "time": np.array(
            [
                [0, 20, 30, 40],
                [20, 0, 15, 35],
                [30, 15, 0, 25],
                [40, 35, 25, 0],
            ],
            dtype=float,
        ),
        "dist": np.array(
            [
                [0, 10, 20, 30],
                [10, 0, 8, 22],
                [20, 8, 0, 18],
                [30, 22, 18, 0],
            ],
            dtype=float,
        ),
    }

    displaced_task = {
        "from": "A",
        "to": "D",
        "start_min": 640,
        "end_min": 700,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
        max_drivers_affected=5,
    )

    assert assignment["assigned"] is False
    assert assignment["chain_step"]["detail"] == "no_secondary_driver_fit"


def test_attempt_secondary_assignment_rejects_as_directed_fallback_append():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "home_loc": "H",
                    "elements": [
                        {
                            "element_type": "START FACILITY",
                            "is_travel": False,
                            "from": "H",
                            "to": "H",
                            "start_min": 480,
                            "end_min": 510,
                            "Tue": 1,
                        },
                        {
                            "element_type": "AS DIRECTED",
                            "is_travel": False,
                            "from": "H",
                            "to": "H",
                            "start_min": 600,
                            "end_min": 620,
                            "priority": 3,
                            "Tue": 1,
                        },
                        {
                            "element_type": "END FACILITY",
                            "is_travel": False,
                            "from": "H",
                            "to": "H",
                            "start_min": 900,
                            "end_min": 915,
                            "Tue": 1,
                        },
                    ],
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1, "H": 2},
        "time": np.array(
            [
                [0, 30, 20],
                [30, 0, 20],
                [20, 20, 0],
            ],
            dtype=float,
        ),
        "dist": np.array(
            [
                [0, 15, 10],
                [15, 0, 10],
                [10, 10, 0],
            ],
            dtype=float,
        ),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 605,
        "end_min": 665,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
        max_drivers_affected=5,
    )

    assert assignment["assigned"] is False
    assert assignment["chain_step"]["detail"] == "no_secondary_driver_fit"


def test_attempt_secondary_assignment_rejects_overlap_secondary_strategy():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "home_loc": "B",
                    "elements": [
                        {
                            "element_type": "START FACILITY",
                            "is_travel": False,
                            "from": "C",
                            "to": "C",
                            "start_min": 80,
                            "end_min": 100,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "C",
                            "to": "B",
                            "start_min": 120,
                            "end_min": 180,
                            "priority": 4,
                            "Tue": 1,
                        },
                    ],
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1, "C": 2},
        "time": np.array(
            [
                [0, 30, 20],
                [30, 0, 20],
                [20, 20, 0],
            ],
            dtype=float,
        ),
        "dist": np.array(
            [
                [0, 15, 10],
                [15, 0, 10],
                [10, 10, 0],
            ],
            dtype=float,
        ),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 120,
        "end_min": 180,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
        max_drivers_affected=5,
    )

    assert assignment["assigned"] is False
    assert assignment["chain_step"]["detail"] == "no_secondary_driver_fit"


def test_attempt_secondary_assignment_respects_max_drivers_affected_limit():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "B",
                            "to": "A",
                            "start_min": 200,
                            "end_min": 240,
                            "Tue": 1,
                        }
                    ]
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1},
        "time": np.array([[0, 20], [20, 0]], dtype=float),
        "dist": np.array([[0, 10], [10, 0]], dtype=float),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 100,
        "end_min": 160,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
        max_drivers_affected=1,
    )

    assert assignment["assigned"] is False
    assert assignment["chain_step"]["detail"] == "max_drivers_affected_reached"


def test_attempt_secondary_assignment_enforces_home_base_when_strict(monkeypatch):
    monkeypatch.setattr(cascade_mod, "ENABLE_STRICT_LEGAL_CONSTRAINTS", True)

    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "home_loc": "C",
                    "elements": [
                        {
                            "element_type": "START FACILITY",
                            "is_travel": False,
                            "from": "C",
                            "to": "C",
                            "start_min": 50,
                            "end_min": 60,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "C",
                            "to": "A",
                            "start_min": 200,
                            "end_min": 240,
                            "Tue": 1,
                        },
                    ]
                },
                "D3": {
                    "home_loc": "B",
                    "elements": [
                        {
                            "element_type": "START FACILITY",
                            "is_travel": False,
                            "from": "B",
                            "to": "B",
                            "start_min": 50,
                            "end_min": 60,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "B",
                            "to": "A",
                            "start_min": 80,
                            "end_min": 120,
                            "priority": 5,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 120,
                            "end_min": 180,
                            "priority": 3,
                            "Tue": 1,
                        },
                    ]
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1, "C": 2},
        "time": np.array(
            [
                [0, 20, 30],
                [20, 0, 25],
                [30, 25, 0],
            ],
            dtype=float,
        ),
        "dist": np.array(
            [
                [0, 10, 15],
                [10, 0, 12],
                [15, 12, 0],
            ],
            dtype=float,
        ),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 100,
        "end_min": 160,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
        max_drivers_affected=5,
    )

    assert assignment["assigned"] is True
    assert assignment["secondary_driver_id"] == "D3"


def test_secondary_assignment_preserves_end_facility_structure():
    data = {
        "driver_states": {
            "drivers": {
                "D1": {
                    "elements": [
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 100,
                            "end_min": 160,
                            "Tue": 1,
                        }
                    ]
                },
                "D2": {
                    "home_loc": "B",
                    "elements": [
                        {
                            "element_type": "START FACILITY",
                            "is_travel": False,
                            "from": "H",
                            "to": "H",
                            "start_min": 500,
                            "end_min": 525,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "H",
                            "to": "A",
                            "start_min": 560,
                            "end_min": 620,
                            "priority": 5,
                            "Tue": 1,
                        },
                        {
                            "element_type": "TRAVEL",
                            "is_travel": True,
                            "from": "A",
                            "to": "B",
                            "start_min": 620,
                            "end_min": 680,
                            "priority": 3,
                            "Tue": 1,
                        },
                        {
                            "element_type": "END FACILITY",
                            "is_travel": False,
                            "from": "B",
                            "to": "B",
                            "start_min": 900,
                            "end_min": 915,
                            "Tue": 1,
                        },
                    ]
                },
            }
        }
    }

    matrices = {
        "loc2idx": {"A": 0, "B": 1, "H": 2, "X": 3},
        "time": np.array(
            [
                [0, 30, 40, 20],
                [30, 0, 50, 25],
                [40, 50, 0, 20],
                [20, 25, 20, 0],
            ],
            dtype=float,
        ),
        "dist": np.array(
            [
                [0, 20, 30, 10],
                [20, 0, 35, 12],
                [30, 35, 0, 8],
                [10, 12, 8, 0],
            ],
            dtype=float,
        ),
    }

    displaced_task = {
        "from": "A",
        "to": "B",
        "start_min": 620,
        "end_min": 680,
        "priority": 3,
        "element_type": "TRAVEL",
        "is_travel": True,
    }

    assignment = _attempt_secondary_assignment_for_displaced(
        displaced_task=displaced_task,
        primary_driver_id="D1",
        DATA=data,
        matrices=matrices,
        weekday="Tue",
        max_depth=2,
        max_drivers_affected=5,
    )

    assert assignment["assigned"] is True
    assert assignment["secondary_driver_id"] == "D2"
    after = assignment["after_schedule"]
    assert after
    assert str(after[-1].get("element_type", "")).upper() == "END FACILITY"
