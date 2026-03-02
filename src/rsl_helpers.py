from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DAYSETS: list[list[str]] = [
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
]


def parse_hms_to_minutes(value) -> float:
    try:
        h, m, s = str(value).split(":")
        return int(h) * 60 + int(m) + int(s) / 60.0
    except Exception:
        return np.nan


def truthy(value) -> bool:
    if isinstance(value, (int, float)):
        return int(value) != 0
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"y", "yes", "true", "1", "t"}


def find_day_cols(df: pd.DataFrame, daysets: Iterable[list[str]] | None = None) -> list[str]:
    search_sets = list(daysets) if daysets is not None else DAYSETS
    for candidate in search_sets:
        if all(col in df.columns for col in candidate):
            return candidate

    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in search_sets:
        if all(day.lower() in lower_cols for day in candidate):
            return [lower_cols[day.lower()] for day in candidate]

    return []
