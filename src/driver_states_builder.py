from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.rsl_helpers import find_day_cols, parse_hms_to_minutes, truthy
from src.priority_derivation import derive_priority

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
POST_MIDNIGHT_CARRY_CUTOFF_MINUTES = 6 * 60


def _normalize_element_type(value: Any) -> str:
    return (value or "").strip().upper()


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        as_str = str(value).strip()
        if not as_str or as_str.lower() in {"nan", "none", "null"}:
            return None
        return float(as_str)
    except Exception:
        return None


def _to_int_or_default(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        as_str = str(value).strip()
        if not as_str or as_str.lower() in {"nan", "none", "null"}:
            return default
        return int(float(as_str))
    except Exception:
        return default


def _canon_col_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _pick_col(columns: list[str], *candidates: str) -> str | None:
    by_canon = {_canon_col_name(col): col for col in columns}
    for candidate in candidates:
        found = by_canon.get(_canon_col_name(candidate))
        if found:
            return found
    return None


def _build_name_to_id_map(location_index_path: str | Path | None) -> dict[str, Any]:
    if not location_index_path:
        return {}

    path = Path(location_index_path)
    if not path.exists():
        return {}

    location_index = pd.read_csv(path)
    if not {"name", "center_id"}.issubset(location_index.columns):
        return {}

    return {
        str(name).upper().strip(): int(center_id)
        for name, center_id in zip(location_index["name"], location_index["center_id"])
        if str(name).strip()
    }


def build_driver_states(
    df: pd.DataFrame,
    location_index_path: str | Path | None = None,
    priority_map: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = ["Duty ID", "Element Type", "Commencement Time", "Ending Time"]
    for column in required:
        if column not in df.columns:
            raise SystemExit(f"Missing required column: {column}")

    day_cols = find_day_cols(df)
    if not day_cols:
        day_cols = WEEKDAYS[:]

    data = df.copy()
    data["start_min"] = data["Commencement Time"].apply(parse_hms_to_minutes)
    data["end_min"] = data["Ending Time"].apply(parse_hms_to_minutes)
    data["ET_NORM"] = data["Element Type"].apply(_normalize_element_type)

    name_to_id = _build_name_to_id_map(location_index_path)

    load_type_col = _pick_col(
        list(data.columns),
        "Due To Convey",
        "Due to Convey",
        "DueToConvey",
        "Service Type",
    )
    planz_col = _pick_col(
        list(data.columns),
        "PLANZ Code",
        "Planz Code",
        "planz_code",
    )
    priority_col = "Priority" if "Priority" in data.columns else None

    states: dict[str, Any] = {}
    qa_rows: list[dict[str, Any]] = []

    for duty_id, group in data.groupby("Duty ID", dropna=True):
        group = group.copy()

        start_rows = group[group["ET_NORM"] == "START FACILITY"]
        end_rows = group[group["ET_NORM"] == "END FACILITY"]

        if start_rows.empty:
            start_row = group.sort_values("start_min", ascending=True).iloc[0]
        else:
            start_row = start_rows.sort_values("start_min", ascending=True).iloc[0]

        if end_rows.empty:
            end_row = group.sort_values("end_min", ascending=False).iloc[0]
        else:
            end_row = end_rows.sort_values("end_min", ascending=False).iloc[0]

        start = float(start_row["start_min"])
        end = float(end_row["end_min"])
        nominal_end_min = _to_int_or_default(end_row.get("end_min"), 0)
        crosses_midnight = end < start
        if end < start:
            end += 24 * 60

        allowed_days: list[str] = []
        for day_column in day_cols:
            if day_column in start_row and truthy(start_row[day_column]):
                allowed_days.append(day_column)

        if not allowed_days:
            for day_column in day_cols:
                values = group[day_column] if day_column in group.columns else []
                if any(truthy(value) for value in values):
                    allowed_days.append(day_column)

        if not allowed_days:
            allowed_days = day_cols[:]

        home_center_id = None
        start_name = start_row.get("Mapped Name A")
        if isinstance(start_name, str) and start_name.strip():
            lookup = name_to_id.get(start_name.upper().strip())
            home_center_id = int(lookup) if lookup is not None else None

        grade = None
        if "Driver Grade" in group.columns and group["Driver Grade"].notna().any():
            grade = group["Driver Grade"].dropna().iloc[0]

        vehicle_type: Any = None
        if "Vehicle Type" in group.columns:
            vehicle_types = sorted(
                value
                for value in group["Vehicle Type"].dropna().unique()
                if value and str(value).strip().lower() != "no_data"
            )
            vehicle_type = vehicle_types[0] if vehicle_types else None

        daily_windows: dict[str, Any] = {
            day: {
                "start_min": int(round(start)),
                "end_min": int(round(end)),
                "crosses_midnight": bool(crosses_midnight),
            }
            for day in allowed_days
        }

        elements: list[dict[str, Any]] = []
        group_sorted = group.sort_values(["start_min", "end_min"], ascending=[True, True])
        for _, row in group_sorted.iterrows():
            element_type = _normalize_element_type(row.get("Element Type"))
            from_name = str(row.get("Mapped Name A", "") or "").upper().strip()
            to_name = str(row.get("Mapped Name B", "") or "").upper().strip()
            if not from_name:
                from_name = "NO_DATA"
            if not to_name:
                to_name = from_name

            from_id = name_to_id.get(from_name)
            to_id = name_to_id.get(to_name)

            start_min = _to_int_or_default(row.get("start_min"), 0)
            end_min = _to_int_or_default(row.get("end_min"), start_min)

            duration_min = end_min - start_min
            if duration_min < 0:
                duration_min += 24 * 60

            load_type = "NO_DATA"
            if load_type_col:
                raw_load_type = row.get(load_type_col)
                if raw_load_type is not None and str(raw_load_type).strip():
                    load_type = str(raw_load_type).upper().strip()
            if "TRAVEL" in element_type and load_type == "NO_DATA":
                load_type = "TRAVEL_NO_DATA"

            planz_code = "NO_DATA"
            if planz_col:
                raw_planz = row.get(planz_col)
                if raw_planz is not None and str(raw_planz).strip():
                    planz_code = str(raw_planz).upper().strip()

            raw_priority = row.get(priority_col) if priority_col else None
            priority = derive_priority(raw_priority, load_type, priority_map, default=3)

            element: dict[str, Any] = {
                "element_type": element_type,
                "is_travel": bool("TRAVEL" in element_type),
                "start": str(row.get("Commencement Time", "") or ""),
                "start_min": start_min,
                "end": str(row.get("Ending Time", "") or ""),
                "end_min": end_min,
                "from": from_name,
                "to": to_name,
                "from_id": int(from_id) if from_id is not None else None,
                "to_id": int(to_id) if to_id is not None else None,
                "miles": _to_float_or_none(row.get("Leg Mileage")),
                "duration_min": int(duration_min),
                "load_type": load_type,
                "planz_code": planz_code,
                "priority": int(priority),
            }

            for day_name in WEEKDAYS:
                if day_name in row:
                    element[day_name] = 1 if truthy(row.get(day_name)) else 0
                else:
                    element[day_name] = 1 if day_name in allowed_days else 0

            if crosses_midnight and start_min < POST_MIDNIGHT_CARRY_CUTOFF_MINUTES and start_min <= nominal_end_min:
                for i, day_name in enumerate(WEEKDAYS):
                    prev_day = WEEKDAYS[i - 1]
                    if int(element.get(day_name, 0) or 0) == 0 and int(element.get(prev_day, 0) or 0) == 1:
                        element[day_name] = 1
                element["overnight_day_carry"] = True

            elements.append(element)

        states[str(duty_id)] = {
            "duty_id": str(duty_id),
            "days": allowed_days,
            "daily_windows": daily_windows,
            "elements": elements,
            "weekly_emergency_rest_quota": 2,
            "home_center_id": home_center_id,
            "grade": grade,
            "vehicle_type": vehicle_type,
        }

        qa_rows.append(
            {
                "duty_id": duty_id,
                "window_start_min": start,
                "window_end_min": end,
                "allowed_days": ",".join(allowed_days),
                "home_center_id": home_center_id,
                "grade": grade,
                "vehicle_type": vehicle_type,
                "elements_count": len(elements),
                "crosses_midnight": bool(crosses_midnight),
            }
        )

    return states, pd.DataFrame(qa_rows)
