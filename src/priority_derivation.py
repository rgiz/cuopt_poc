from __future__ import annotations

from typing import Any, Dict


def clamp_priority(priority: int, default: int = 3) -> int:
    try:
        return max(1, min(5, int(priority)))
    except Exception:
        return int(default)


def normalize_priority_map(priority_map: Dict[str, Any] | None) -> Dict[str, int]:
    if not priority_map:
        return {}
    normalized: Dict[str, int] = {}
    for key, value in priority_map.items():
        normalized[str(key).upper().strip()] = clamp_priority(value)
    return normalized


def derive_priority(
    raw_priority: Any,
    load_type: Any,
    priority_map: Dict[str, Any] | None,
    default: int = 3,
) -> int:
    mapped = normalize_priority_map(priority_map)

    # Explicit numeric priority in input wins when valid.
    if raw_priority is not None:
        raw_str = str(raw_priority).strip()
        if raw_str and raw_str.lower() not in {"nan", "none", "null"}:
            try:
                return clamp_priority(int(float(raw_str)), default=default)
            except Exception:
                pass

    load_key = str(load_type or "").upper().strip()
    if load_key and load_key in mapped:
        return clamp_priority(mapped[load_key], default=default)

    if "DEFAULT" in mapped:
        return clamp_priority(mapped["DEFAULT"], default=default)

    return clamp_priority(default, default=default)
