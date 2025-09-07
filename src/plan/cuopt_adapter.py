# cuopt_adapter.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional, Sequence, List
import requests
from urllib.parse import urljoin

DEFAULT_TIMEOUT = float(os.getenv("CUOPT_SOLVER_TIMEOUT_SEC", "120"))

def _best_base(url: str) -> str:
    """
    Returns a base URL that works with this cuOpt server.
    Preference: <base>/v2 if /v2/health/live returns 200.
    Otherwise, use <base>.
    """
    base = url.rstrip("/")
    try:
        r = requests.get(urljoin(base + "/", "v2/health/live"), timeout=3)
        if r.status_code == 200:
            return base + "/v2"
    except Exception:
        pass
    return base

def _try_post(endpoints: Sequence[str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """
    Try several endpoints in order; return first successful JSON.
    Raise the last exception if all fail.
    """
    last_err: Optional[Exception] = None
    for ep in endpoints:
        try:
            resp = requests.post(ep, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err

def solve_with_cuopt(raw_base_url: Optional[str], payload: Dict[str, Any], timeout_sec: Optional[float] = None) -> Dict[str, Any]:
    """
    Call cuOpt server with a payload. We normalize base, then try common endpoints.
    """
    timeout = float(timeout_sec or DEFAULT_TIMEOUT)
    base_env = raw_base_url or os.getenv("CUOPT_URL", "http://cuopt:5000")
    base = _best_base(base_env)

    # Known solve endpoints across image versions:
    #  - POST <base>/solve
    #  - POST <base>/request
    candidates = [
        urljoin(base + "/", "solve"),
        urljoin(base + "/", "request"),
    ]
    return _try_post(candidates, payload, timeout)

def _collect_nodes(drivers: Dict[str, Any], extra_nodes: Optional[List[str]] = None) -> List[str]:
    """Collect unique location names from driver elements + any extras."""
    nodes = set()
    for meta in drivers.values():
        for e in meta.get("elements", []):
            if e.get("is_travel"):
                fr = e.get("from")
                to = e.get("to")
                if fr: nodes.add(str(fr).upper())
                if to: nodes.add(str(to).upper())
    if extra_nodes:
        for n in extra_nodes:
            if n:
                nodes.add(str(n).upper())
    return sorted(nodes)

def _dense_matrices_from_M(M: Dict[str, Any], nodes: List[str]) -> Dict[str, Any]:
    """
    Build dense distance/time matrices aligned to `nodes` using M['dist'], M['time'], M['loc2idx'].
    Unknown pairs become +inf (except diagonal 0).
    """
    loc2idx = M["loc2idx"]
    n = len(nodes)
    dist = [[0.0]*n for _ in range(n)]
    time = [[0.0]*n for _ in range(n)]
    for a, A in enumerate(nodes):
        i = loc2idx.get(A)
        for b, B in enumerate(nodes):
            j = loc2idx.get(B)
            if i is None or j is None:
                if a != b:
                    dist[a][b] = float("inf")
                    time[a][b] = float("inf")
            else:
                dist[a][b] = float(M["dist"][i, j])
                time[a][b] = float(M["time"][i, j])
    return {"nodes": nodes, "distance": dist, "time": time}

def build_cuopt_payload(
    DATA: Dict[str, Any],
    request_trip: Dict[str, Any],
    assignments_so_far: List[Dict[str, Any]],
    priorities: Dict[str, int],
    sla_windows: Dict[int, Dict[str, int]],
    M: Dict[str, Any],
    new_req_window: Optional[List[int]] = None,   # <-- pass [earliest, latest] from router
) -> Dict[str, Any]:
    """
    Map current in-memory state to a cuOpt-friendly payload:
      - vehicles: one per driver (start location + operating window)
      - tasks: each existing travel element (optional, prize by priority) + the NEW mandatory task
      - matrices: dense distance/time matrices over required nodes
    NOTE: Adjust the output shape to match your cuOpt REST API exactly.
    """
    # 1) Vehicles (drivers)
    ds = DATA.get("driver_states", {})
    drivers = ds.get("drivers", ds) if isinstance(ds, dict) else {}
    vehicles: List[Dict[str, Any]] = []
    for drv_id, meta in drivers.items():
        # Start location preference: explicit start_loc -> first element.from -> home_loc -> ""
        start_loc = str(
            meta.get("start_loc")
            or (meta.get("elements", [{}])[0].get("from") if meta.get("elements") else "")
            or meta.get("home_loc")
            or ""
        ).upper()
        earliest = int(meta.get("start_min", 0))
        latest   = int(meta.get("end_min", 24*60))
        vehicles.append({
            "id": drv_id,
            "start_location": start_loc,
            "time_window": [earliest, latest],
            # Optionally: "end_location": start_loc,
        })

    # 2) Tasks (existing + NEW)
    tasks: List[Dict[str, Any]] = []

    # Existing trips become optional tasks with prize keyed to priority.
    for drv_id, meta in drivers.items():
        for e in meta.get("elements", []):
            if not e.get("is_travel"):
                continue
            fr = str(e.get("from", "")).upper()
            to = str(e.get("to", "")).upper()
            if not fr or not to:
                continue
            prio = int(e.get("priority", 3))
            earliest = int(e.get("earliest", e.get("start_min", 0)))
            latest   = int(e.get("latest",   e.get("end_min",   24*60)))
            prize = 1000 if prio == 1 else 100 if prio == 2 else 10 if prio == 3 else 1
            tasks.append({
                "id": f"TASK:{drv_id}:{int(e.get('start_min', 0))}",
                "from": fr,
                "to": to,
                "time_window": [earliest, latest],
                "priority": prio,        # for your traceability
                "prize": prize,          # prize-collection style
                "mandatory": False,      # allow reallocation/drop if needed
            })

    # NEW request (mandatory)
    new_task = {
        "id": request_trip["id"],  # e.g., "NEW:A->B@<ts>"
        "from": str(request_trip["start_location"]).upper(),
        "to":   str(request_trip["end_location"]).upper(),
        "time_window": new_req_window or [0, 24*60],
        "priority": int(request_trip.get("priority", 3)),
        "prize": 10_000,           # very high prize to enforce serving this task
        "mandatory": True,
    }
    tasks.append(new_task)

    # 3) Matrices
    nodes = _collect_nodes(drivers, extra_nodes=[new_task["from"], new_task["to"]])
    matrices = _dense_matrices_from_M(M, nodes)

    # 4) Objective & options (tune to your cuOpt service schema)
    payload = {
        "vehicles": vehicles,
        "tasks": tasks,
        "matrices": matrices,
        "objective": {
            # Example: minimize composite 'cost' with weights;
            # your cuOpt service may expect different fields.
            "minimize": "cost",
            "weights": {"deadhead": 1.0, "overtime": 1.0, "admin": 1.0}
        },
        "options": {
            "return_multiple_solutions": True,
            "max_solutions": 5
        }
    }
    return payload

# ---- Optional: HTTP helpers to call cuOpt + parse response ----

def solve_with_cuopt(cuopt_url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """
    POST payload to cuOpt server. Adjust the endpoint path to match your container.
    """
    import requests
    url = cuopt_url.rstrip("/") + "/solve"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def extract_solutions_from_cuopt(raw: Dict[str, Any], max_solutions: int = 5) -> List[Dict[str, Any]]:
    """
    Turn cuOpt response into our neutral structure:
      { objective_value, assignments[], cascades[], details{} }
    Adjust to your cuOpt response shape (routes/steps field names).
    """
    sols = []
    many = raw.get("solutions") if isinstance(raw, dict) else None
    base = many if isinstance(many, list) else [raw]

    for sol in base[:max_solutions]:
        obj = float(sol.get("objective_value", 0.0))
        assignments: List[Dict[str, Any]] = []
        cascades: List[Dict[str, Any]] = []

        # Example traversal — adapt to your real schema:
        for route in sol.get("routes", []):
            drv_id = route.get("vehicle_id")
            for step in route.get("steps", []):
                tid = step.get("task_id")
                if not tid:
                    continue
                assignments.append({
                    "trip_id": tid,                       # e.g. "NEW:A->B@..." or "TASK:drv:time"
                    "type": "reassigned",                 # normalize; refine if cuOpt returns labels
                    "driver_id": drv_id,
                    "candidate_id": f"cuopt:{tid}",
                    "delay_minutes": step.get("delay_min"),
                    "deadhead_miles": step.get("deadhead_miles"),
                    "overtime_minutes": step.get("overtime_min"),
                    "miles_delta": step.get("delta_miles"),
                    "cost": step.get("cost"),
                    "cost_breakdown": step.get("cost_breakdown", {}),
                })

        sols.append({
            "objective_value": obj,
            "assignments": assignments,
            "cascades": cascades,   # populate if cuOpt emits displacement info
            "details": {"backend": "cuopt"},
        })

    return sols
