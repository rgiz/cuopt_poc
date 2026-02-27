from __future__ import annotations

import math

import numpy as np
import pandas as pd


def build_from_location_index(li: pd.DataFrame, mph: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    table = li.copy()
    table.columns = [column.lower() for column in table.columns]
    size = len(table["name"].astype(str).str.upper().tolist())
    distance = np.zeros((size, size), float)
    duration = np.zeros((size, size), float)

    have_lat_lon = {"lat", "lon"}.issubset(table.columns)

    def haversine_miles(a: pd.Series, b: pd.Series) -> float:
        radius = 3958.7613
        lat1, lon1, lat2, lon2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(h))

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if (
                have_lat_lon
                and pd.notna(table.iloc[i]["lat"])
                and pd.notna(table.iloc[i]["lon"])
                and pd.notna(table.iloc[j]["lat"])
                and pd.notna(table.iloc[j]["lon"])
            ):
                miles = haversine_miles(table.iloc[i], table.iloc[j])
            else:
                miles = 0.0

            distance[i, j] = miles
            duration[i, j] = miles / mph * 60.0

    return distance, duration
