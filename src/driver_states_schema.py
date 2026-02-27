from __future__ import annotations

from typing import Any, Dict


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _normalize_days(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    out: list[str] = []
    for value in values:
        text = str(value).strip().title()[:3]
        if text in WEEKDAYS and text not in out:
            out.append(text)
    return out


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def normalize_driver_states_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"schema_version": "2.0", "drivers": {}}

    source_drivers = payload.get("drivers") if isinstance(payload.get("drivers"), dict) else {}

    normalized_drivers: dict[str, dict[str, Any]] = {}

    for duty_id, raw_meta in source_drivers.items():
        if not isinstance(raw_meta, dict):
            continue

        meta = dict(raw_meta)

        days = _normalize_days(meta.get("days"))
        daily_windows = meta.get("daily_windows") if isinstance(meta.get("daily_windows"), dict) else {}

        if not days and daily_windows:
            days = [day for day in WEEKDAYS if day in daily_windows]
        if not days:
            days = WEEKDAYS[:]

        elements = meta.get("elements")
        if not isinstance(elements, list):
            elements = []

        meta["duty_id"] = str(meta.get("duty_id", duty_id))
        meta["days"] = days
        meta["daily_windows"] = daily_windows
        meta["elements"] = elements
        meta["weekly_emergency_rest_quota"] = _coerce_int(meta.get("weekly_emergency_rest_quota", 2), 2)
        if "vehicle_type" not in meta:
            meta["vehicle_type"] = None

        normalized_drivers[str(duty_id)] = meta

    normalized_payload: Dict[str, Any] = {
        "schema_version": str(payload.get("schema_version", "2.0")),
        "drivers": normalized_drivers,
    }

    if "location_index_size" in payload:
        normalized_payload["location_index_size"] = payload["location_index_size"]

    if "producer" in payload:
        normalized_payload["producer"] = payload["producer"]

    return normalized_payload
