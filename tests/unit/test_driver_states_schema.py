from src.driver_states_schema import normalize_driver_states_payload


def test_normalize_minimal_payload_to_drivers_envelope():
    legacy = {
        "schema_version": "2.0",
        "drivers": {
            "D001": {
                "days": ["Mon"],
                "daily_windows": {"Mon": {"start_min": 480, "end_min": 1080, "crosses_midnight": False}},
                "elements": [],
                "vehicle_type": "7.5T",
            }
        },
    }

    out = normalize_driver_states_payload(legacy)

    assert "drivers" in out
    assert "D001" in out["drivers"]
    d = out["drivers"]["D001"]
    assert d["daily_windows"]["Mon"]["start_min"] == 480
    assert d["daily_windows"]["Mon"]["end_min"] == 1080
    assert d["elements"] == []
    assert d["vehicle_type"] == "7.5T"


def test_preserve_rich_payload_structure():
    rich = {
        "schema_version": "2.0",
        "drivers": {
            "D010": {
                "duty_id": "D010",
                "days": ["Tue"],
                "daily_windows": {"Tue": {"start_min": 600, "end_min": 1200, "crosses_midnight": False}},
                "elements": [{"element_type": "TRAVEL", "is_travel": True, "from": "A", "to": "B"}],
                "weekly_emergency_rest_quota": 2,
            }
        },
    }

    out = normalize_driver_states_payload(rich)

    assert out["schema_version"] == "2.0"
    assert "D010" in out["drivers"]
    d = out["drivers"]["D010"]
    assert d["duty_id"] == "D010"
    assert d["days"] == ["Tue"]
    assert len(d["elements"]) == 1
