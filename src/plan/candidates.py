# src/plan/candidates.py
from __future__ import annotations
from typing import Dict, Any, List, Set
import numpy as np

def _nearest_eta_in_route(route: list[dict], job_window: tuple[int,int]) -> int:
    """Return the ETA (minute-of-day) in the driver route closest to the job window start."""
    if not route:
        return 0
    target = job_window[0]
    etas = [int(step.get("eta_start", step.get("start_min", target))) for step in route]
    diffs = [abs(e - target) for e in etas]
    return etas[int(np.argmin(diffs))]

def shortlist(
    baseline: Dict[str, Any],
    job: Dict[str, Any],
    valid_location_ids: Set[str],
    matrices: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
) -> List[str]:
    """
    baseline: your baseline solution dict, expected to have {'routes': {driver_id: [steps...]}}
    job: {pickup_location_id, drop_location_id?, time_window:[s,e], service_time, priority}
    """
    cfg = cfg or {}
    k = int(cfg.get("shortlist_size", 20))

    # quick index access
    id_by_idx = matrices.get("index", None)
    time_mat = matrices["time"]

    # helper to map location_id -> matrix index
    if id_by_idx is not None and hasattr(id_by_idx, "set_index"):
        index_map = {str(r["location_id"]): int(r["matrix_index"]) for _, r in id_by_idx.iterrows()}
    elif "index_map" in baseline.get("problem", {}):
        index_map = baseline["problem"]["index_map"]
    else:
        # last resort, assume consecutive order
        index_map = {}

    pick = job["pickup_location_id"]
    if pick not in valid_location_ids or pick not in index_map:
        return []  # cannot evaluate without a valid pickup location

    pick_idx = index_map[pick]
    cand_scores: list[tuple[str, float]] = []

    for driver_id, route in baseline.get("routes", {}).items():
        # filter invalid routes
        if any((step.get("location_id") not in valid_location_ids) for step in route if "location_id" in step):
            continue

        # proximity score: min travel time from any location in route to the pickup
        if not route:
            continue
        # gather indices seen in route (fallback to skip if missing)
        idxs = []
        for step in route:
            lid = step.get("location_id")
            if lid in index_map:
                idxs.append(index_map[lid])
        if not idxs:
            continue
        # minimal travel time from route to pickup
        tmins = [time_mat[i, pick_idx] for i in idxs]
        best_t = float(np.min(tmins)) if tmins else 1e9

        # slack proxy: distance in time between job window start and nearest ETA
        eta = _nearest_eta_in_route(route, tuple(job["time_window"]))
        slack = max(0, job["time_window"][0] - eta)

        # lower score is better: weigh proximity more, then slack
        score = best_t - 0.25 * slack
        cand_scores.append((driver_id, float(score)))

    cand_scores.sort(key=lambda x: x[1])
    return [d for d, _ in cand_scores[:k]]
