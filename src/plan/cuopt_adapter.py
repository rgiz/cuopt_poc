from __future__ import annotations
import os
from typing import Any, Dict, Optional, Sequence, List
import requests
import time
from urllib.parse import urljoin

DEFAULT_TIMEOUT = float(os.getenv("CUOPT_SOLVER_TIMEOUT_SEC", "120"))

def solve_with_cuopt(raw_base_url: Optional[str], payload: Dict[str, Any], timeout_sec: Optional[float] = None) -> Dict[str, Any]:
    """
    Call cuOpt server with a payload using the new async pattern for 25.10.0a.
    """
    timeout = float(timeout_sec or DEFAULT_TIMEOUT)
    base_env = raw_base_url or os.getenv("CUOPT_URL", "http://localhost:5000")
    base = base_env.rstrip("/")

    # Use correct cuOpt 25.x endpoint
    headers = {
        'Content-Type': 'application/json',
        'CLIENT-VERSION': 'custom'
    }
    
    try:
        # Step 1: Submit request to /cuopt/request (not /solve)
        response = requests.post(
            f"{base}/cuopt/request",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        initial = response.json()
        
        # Check for immediate response (rare in 25.x)
        if 'response' in initial and 'solver_response' in initial['response']:
            return initial['response']
            
        # Get request ID for polling
        request_id = initial.get('reqId')
        if not request_id:
            raise ValueError(f"No request ID in response: {initial}")
        
        # Step 2: Poll for results
        start_time = time.time()
        while time.time() - start_time < timeout:
            poll_response = requests.get(
                f"{base}/cuopt/requests/{request_id}",
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if poll_response.status_code == 200:
                result = poll_response.json()
                if 'response' in result and 'solver_response' in result['response']:
                    return result['response']
            elif poll_response.status_code == 404:
                # Still processing, continue polling
                pass
            else:
                poll_response.raise_for_status()
            
            time.sleep(1)  # Poll every second
            
        raise TimeoutError(f"cuOpt request {request_id} timed out after {timeout}s")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                "cuOpt endpoint not found. Ensure container is running with: "
                "docker run -d --gpus=1 -p 5000:5000 nvidia/cuopt:25.10.0a-cuda12.9-py3.12"
            )
        raise

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
    new_req_window: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Map current in-memory state to a cuOpt-friendly payload for 25.10.0a:
      - vehicles: one per driver (start location + operating window)
      - tasks: each existing travel element (optional, prize by priority) + the NEW mandatory task
      - matrices: dense distance/time matrices over required nodes
    """
    # 1) Vehicles (drivers)
    ds = DATA.get("driver_states", {})
    drivers = ds.get("drivers", ds) if isinstance(ds, dict) else {}
    vehicles = []
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
            "start_location": start_loc,
            "time_window": [earliest, latest],
        })

    # 2) Tasks (existing + NEW)
    tasks = []

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
                "location": fr,  # Using pickup location for simplicity
                "time_window": [earliest, latest],
                "priority": prio,
                "service_time": int(e.get("duration_min", 0)),
                "demand": 1,
            })

    # NEW request (mandatory)
    new_task = {
        "location": str(request_trip["start_location"]).upper(),
        "time_window": new_req_window or [0, 24*60],
        "priority": int(request_trip.get("priority", 3)),
        "service_time": int(request_trip.get("duration_minutes", 0)),
        "demand": 1,
    }
    tasks.append(new_task)

    # 3) Matrices
    nodes = _collect_nodes(drivers, extra_nodes=[new_task["location"]])
    matrices = _dense_matrices_from_M(M, nodes)

    # 4) Build payload for cuOpt 25.10.0a format
    payload = {
        "cost_matrix_data": {
            "data": {
                "0": matrices["distance"]
            }
        },
        "travel_time_matrix_data": {
            "data": {
                "0": matrices["time"]
            }
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0] for _ in vehicles],  # Simplified for now
            "vehicle_time_windows": [v["time_window"] for v in vehicles],
        },
        "task_data": {
            "task_locations": [t["location"] for t in tasks],
            "task_time_windows": [t["time_window"] for t in tasks],
            "service_times": [t["service_time"] for t in tasks],
            "demand": [[t["demand"]] for t in tasks],
        },
        "solver_config": {
            "time_limit": 60,
        }
    }
    return payload

def extract_solutions_from_cuopt(raw: Dict[str, Any], max_solutions: int = 5) -> List[Dict[str, Any]]:
    """
    Turn cuOpt response into our neutral structure:
      { objective_value, assignments[], cascades[], details{} }
    """
    sols = []
    
    # Handle single solution response
    if 'solver_response' in raw:
        obj = float(raw['solver_response'].get('solution_cost', 0.0))
        assignments = []
        
        # Extract vehicle routes
        vehicle_data = raw['solver_response'].get('vehicle_data', {})
        for vehicle_id, route_info in vehicle_data.items():
            route = route_info.get('route', [])
            for i, location in enumerate(route[1:-1], 1):  # Skip depot
                assignments.append({
                    "trip_id": f"task_{location}",
                    "type": "reassigned",
                    "driver_id": vehicle_id,
                    "cost": obj / len(vehicle_data) if vehicle_data else 0.0,
                })
        
        sols.append({
            "objective_value": obj,
            "assignments": assignments,
            "cascades": [],
            "details": {"backend": "cuopt"}
        })
    
    return sols[:max_solutions]

# # cuopt_adapter.py
# from __future__ import annotations
# import os
# from typing import Any, Dict, Optional, Sequence, List
# import requests
# from urllib.parse import urljoin

# DEFAULT_TIMEOUT = float(os.getenv("CUOPT_SOLVER_TIMEOUT_SEC", "120"))

# def _best_base(url: str) -> str:
#     """
#     Returns a base URL that works with this cuOpt server.
#     Preference: <base>/v2 if /v2/health/live returns 200.
#     Otherwise, use <base>.
#     """
#     base = url.rstrip("/")
#     try:
#         r = requests.get(urljoin(base + "/", "v2/health/live"), timeout=3)
#         if r.status_code == 200:
#             return base + "/v2"
#     except Exception:
#         pass
#     return base

# def _try_post(endpoints: Sequence[str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
#     """
#     Try several endpoints in order; return first successful JSON.
#     Raise the last exception if all fail.
#     """
#     last_err: Optional[Exception] = None
#     for ep in endpoints:
#         try:
#             resp = requests.post(ep, json=payload, timeout=timeout)
#             resp.raise_for_status()
#             return resp.json()
#         except Exception as e:
#             last_err = e
#     assert last_err is not None
#     raise last_err

# def solve_with_cuopt(raw_base_url: Optional[str], payload: Dict[str, Any], timeout_sec: Optional[float] = None) -> Dict[str, Any]:
#     """
#     Call cuOpt server with a payload. We normalize base, then try common endpoints.
#     """
#     timeout = float(timeout_sec or DEFAULT_TIMEOUT)
#     base_env = raw_base_url or os.getenv("CUOPT_URL", "http://cuopt:5000")
#     base = _best_base(base_env)

#     # Known solve endpoints across image versions:
#     #  - POST <base>/solve
#     #  - POST <base>/request
#     candidates = [
#         urljoin(base + "/", "solve"),
#         urljoin(base + "/", "request"),
#     ]
#     return _try_post(candidates, payload, timeout)

# def _collect_nodes(drivers: Dict[str, Any], extra_nodes: Optional[List[str]] = None) -> List[str]:
#     """Collect unique location names from driver elements + any extras."""
#     nodes = set()
#     for meta in drivers.values():
#         for e in meta.get("elements", []):
#             if e.get("is_travel"):
#                 fr = e.get("from")
#                 to = e.get("to")
#                 if fr: nodes.add(str(fr).upper())
#                 if to: nodes.add(str(to).upper())
#     if extra_nodes:
#         for n in extra_nodes:
#             if n:
#                 nodes.add(str(n).upper())
#     return sorted(nodes)

# def _dense_matrices_from_M(M: Dict[str, Any], nodes: List[str]) -> Dict[str, Any]:
#     """
#     Build dense distance/time matrices aligned to `nodes` using M['dist'], M['time'], M['loc2idx'].
#     Unknown pairs become +inf (except diagonal 0).
#     """
#     loc2idx = M["loc2idx"]
#     n = len(nodes)
#     dist = [[0.0]*n for _ in range(n)]
#     time = [[0.0]*n for _ in range(n)]
#     for a, A in enumerate(nodes):
#         i = loc2idx.get(A)
#         for b, B in enumerate(nodes):
#             j = loc2idx.get(B)
#             if i is None or j is None:
#                 if a != b:
#                     dist[a][b] = float("inf")
#                     time[a][b] = float("inf")
#             else:
#                 dist[a][b] = float(M["dist"][i, j])
#                 time[a][b] = float(M["time"][i, j])
#     return {"nodes": nodes, "distance": dist, "time": time}

# def build_cuopt_payload(
#     DATA: Dict[str, Any],
#     request_trip: Dict[str, Any],
#     assignments_so_far: List[Dict[str, Any]],
#     priorities: Dict[str, int],
#     sla_windows: Dict[int, Dict[str, int]],
#     M: Dict[str, Any],
#     new_req_window: Optional[List[int]] = None,   # <-- pass [earliest, latest] from router
# ) -> Dict[str, Any]:
#     """
#     Map current in-memory state to a cuOpt-friendly payload:
#       - vehicles: one per driver (start location + operating window)
#       - tasks: each existing travel element (optional, prize by priority) + the NEW mandatory task
#       - matrices: dense distance/time matrices over required nodes
#     NOTE: Adjust the output shape to match your cuOpt REST API exactly.
#     """
#     # 1) Vehicles (drivers)
#     ds = DATA.get("driver_states", {})
#     drivers = ds.get("drivers", ds) if isinstance(ds, dict) else {}
#     vehicles: List[Dict[str, Any]] = []
#     for drv_id, meta in drivers.items():
#         # Start location preference: explicit start_loc -> first element.from -> home_loc -> ""
#         start_loc = str(
#             meta.get("start_loc")
#             or (meta.get("elements", [{}])[0].get("from") if meta.get("elements") else "")
#             or meta.get("home_loc")
#             or ""
#         ).upper()
#         earliest = int(meta.get("start_min", 0))
#         latest   = int(meta.get("end_min", 24*60))
#         vehicles.append({
#             "id": drv_id,
#             "start_location": start_loc,
#             "time_window": [earliest, latest],
#             # Optionally: "end_location": start_loc,
#         })

#     # 2) Tasks (existing + NEW)
#     tasks: List[Dict[str, Any]] = []

#     # Existing trips become optional tasks with prize keyed to priority.
#     for drv_id, meta in drivers.items():
#         for e in meta.get("elements", []):
#             if not e.get("is_travel"):
#                 continue
#             fr = str(e.get("from", "")).upper()
#             to = str(e.get("to", "")).upper()
#             if not fr or not to:
#                 continue
#             prio = int(e.get("priority", 3))
#             earliest = int(e.get("earliest", e.get("start_min", 0)))
#             latest   = int(e.get("latest",   e.get("end_min",   24*60)))
#             prize = 1000 if prio == 1 else 100 if prio == 2 else 10 if prio == 3 else 1
#             tasks.append({
#                 "id": f"TASK:{drv_id}:{int(e.get('start_min', 0))}",
#                 "from": fr,
#                 "to": to,
#                 "time_window": [earliest, latest],
#                 "priority": prio,        # for your traceability
#                 "prize": prize,          # prize-collection style
#                 "mandatory": False,      # allow reallocation/drop if needed
#             })

#     # NEW request (mandatory)
#     new_task = {
#         "id": request_trip["id"],  # e.g., "NEW:A->B@<ts>"
#         "from": str(request_trip["start_location"]).upper(),
#         "to":   str(request_trip["end_location"]).upper(),
#         "time_window": new_req_window or [0, 24*60],
#         "priority": int(request_trip.get("priority", 3)),
#         "prize": 10_000,           # very high prize to enforce serving this task
#         "mandatory": True,
#     }
#     tasks.append(new_task)

#     # 3) Matrices
#     nodes = _collect_nodes(drivers, extra_nodes=[new_task["from"], new_task["to"]])
#     matrices = _dense_matrices_from_M(M, nodes)

#     # 4) Objective & options (tune to your cuOpt service schema)
#     payload = {
#         "vehicles": vehicles,
#         "tasks": tasks,
#         "matrices": matrices,
#         "objective": {
#             # Example: minimize composite 'cost' with weights;
#             # your cuOpt service may expect different fields.
#             "minimize": "cost",
#             "weights": {"deadhead": 1.0, "overtime": 1.0, "admin": 1.0}
#         },
#         "options": {
#             "return_multiple_solutions": True,
#             "max_solutions": 5
#         }
#     }
#     return payload

# # ---- Optional: HTTP helpers to call cuOpt + parse response ----

# # def solve_with_cuopt(cuopt_url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
# #     """
# #     POST payload to cuOpt server. Adjust the endpoint path to match your container.
# #     """
# #     import requests
# #     url = cuopt_url.rstrip("/") + "/solve"
# #     r = requests.post(url, json=payload, timeout=timeout)
# #     r.raise_for_status()
# #     return r.json()

# def extract_solutions_from_cuopt(raw: Dict[str, Any], max_solutions: int = 5) -> List[Dict[str, Any]]:
#     """
#     Turn cuOpt response into our neutral structure:
#       { objective_value, assignments[], cascades[], details{} }
#     Adjust to your cuOpt response shape (routes/steps field names).
#     """
#     sols = []
#     many = raw.get("solutions") if isinstance(raw, dict) else None
#     base = many if isinstance(many, list) else [raw]

#     for sol in base[:max_solutions]:
#         obj = float(sol.get("objective_value", 0.0))
#         assignments: List[Dict[str, Any]] = []
#         cascades: List[Dict[str, Any]] = []

#         # Example traversal — adapt to your real schema:
#         for route in sol.get("routes", []):
#             drv_id = route.get("vehicle_id")
#             for step in route.get("steps", []):
#                 tid = step.get("task_id")
#                 if not tid:
#                     continue
#                 assignments.append({
#                     "trip_id": tid,                       # e.g. "NEW:A->B@..." or "TASK:drv:time"
#                     "type": "reassigned",                 # normalize; refine if cuOpt returns labels
#                     "driver_id": drv_id,
#                     "candidate_id": f"cuopt:{tid}",
#                     "delay_minutes": step.get("delay_min"),
#                     "deadhead_miles": step.get("deadhead_miles"),
#                     "overtime_minutes": step.get("overtime_min"),
#                     "miles_delta": step.get("delta_miles"),
#                     "cost": step.get("cost"),
#                     "cost_breakdown": step.get("cost_breakdown", {}),
#                 })

#         sols.append({
#             "objective_value": obj,
#             "assignments": assignments,
#             "cascades": cascades,   # populate if cuOpt emits displacement info
#             "details": {"backend": "cuopt"},
#         })

#     return sols

# def solve_with_cuopt(payload: dict, timeout: int = 300) -> dict:
    """
    Send routing problem to cuOpt 25.10.0a using async pattern.
    """
    import requests
    import time
    
    # Use correct cuOpt 25.x endpoint
    base_url = "http://localhost:5000"
    
    # Step 1: Submit request
    headers = {
        'Content-Type': 'application/json',
        'CLIENT-VERSION': 'custom'
    }
    
    try:
        # POST to /cuopt/request, not /solve
        response = requests.post(
            f"{base_url}/cuopt/request",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        initial = response.json()
        
        # Check for immediate response (rare)
        if 'response' in initial:
            return initial['response']
            
        # Get request ID
        request_id = initial.get('reqId')
        if not request_id:
            raise ValueError(f"No request ID in response: {initial}")
        
        # Step 2: Poll for results
        start_time = time.time()
        while time.time() - start_time < timeout:
            poll_response = requests.get(
                f"{base_url}/cuopt/requests/{request_id}",
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if poll_response.status_code == 200:
                result = poll_response.json()
                if 'response' in result:
                    return result['response']
            
            time.sleep(1)  # Poll every second
            
        raise TimeoutError(f"cuOpt request {request_id} timed out")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                "cuOpt endpoint not found. Ensure container is running with: "
                "docker run -d --gpus=1 -p 5000:5000 nvidia/cuopt:25.10.0a-cuda12.9-py3.12"
            )
        raise