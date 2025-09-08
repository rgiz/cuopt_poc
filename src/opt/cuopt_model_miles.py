from __future__ import annotations
import os, time, json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin
import requests
try:
    import msgpack  # for octet-stream / msgpack responses
except Exception:
    msgpack = None

import requests

class CuOptSolution:
    def __init__(self, objective_value: float, assignments: List[Dict[str, Any]], details: Dict[str, Any]):
        self.objective_value = objective_value
        self.assignments = assignments
        self.details = details

class CuOptModel:
    def __init__(
        self,
        driver_states: Dict[str, Any],
        distance_miles_matrix,
        time_minutes_matrix,
        location_to_index: Dict[str, int],
        cost_config: Dict[str, float],
        server_url: str,
        max_solve_time_seconds: int = 30,
        num_workers: int = 16,
        force_greedy: bool = False,
        solve_path: Optional[str] = None,
    ):
        self.driver_states = driver_states
        self.distance = distance_miles_matrix
        self.time = time_minutes_matrix
        # normalize location keys to UPPER
        self.loc2idx = {str(k).upper(): int(v) for k, v in location_to_index.items()}
        self.cost = cost_config
        self.server_url = (server_url or "").rstrip("/") + "/"
        self.max_time = int(max_solve_time_seconds)
        self.num_workers = int(num_workers)

        self._force_greedy = bool(force_greedy or os.getenv("USE_GREEDY_ONLY", "").lower() == "true")
        self._solve_path = solve_path or os.getenv("CUOPT_SOLVE_PATH", "").strip() or None

        if not self._force_greedy and self.server_url and not self._solve_path:
            self._solve_path = self._discover_solve_path()

    def _request_and_poll(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 1) request
        url = urljoin(self.server_url, self._solve_path)
        r = requests.post(url, json=payload, headers={
            "Content-Type":"application/json", "Accept":"application/json"
        }, timeout=min(self.max_time, 10))
        r.raise_for_status()
        rid = r.json().get("reqId")
        if not rid:
            raise RuntimeError(f"Unexpected cuOpt /request response: ct={r.headers.get('content-type')}, body={r.text[:200]}")

        # 2) poll
        poll_paths = ["cuopt/get_result", "cuopt/result", "cuopt/status"]
        headers = {"Accept":"application/json"}
        deadline = time.time() + self.max_time
        last = None
        while time.time() < deadline:
            for p in poll_paths:
                u = urljoin(self.server_url, f"{p}?reqId={rid}")
                try:
                    st = requests.get(u, headers=headers, timeout=5)
                    last = (st.status_code, st.headers.get("content-type",""))
                    if st.status_code == 200 and "application/json" in st.headers.get("content-type",""):
                        return st.json()
                except Exception:
                    continue
            time.sleep(0.5)

        raise RuntimeError("cuOpt result polling exceeded max_solve_time_seconds")


    # ---------- discovery ----------
    def _discover_solve_path(self) -> Optional[str]:
        """
        Try likely endpoints. 404/connection error => not here; 200/202/400/415/422/500 => likely here.
        """
        candidates = [
            "vrp/solve",
            "cuopt/vrp/solve",
            "api/v1/vrp/solve",
            "solve",
            "optimize",
            "route/solve",
            "cuopt/request",   # async mode used by your image
        ]
        for path in candidates:
            url = urljoin(self.server_url, path)
            try:
                r = requests.post(url, json={}, timeout=3)
                if r.status_code in (200, 202, 400, 401, 403, 404, 415, 422, 500):
                    # Treat non-404 as “exists” (some will complain about empty body)
                    if r.status_code != 404:
                        return path
            except Exception:
                continue
        return None

    # ---------- low-level HTTP ----------

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
        url = urljoin(self.server_url, path)
        if params:
            url = url + ("?" + urlencode(params))
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"cuOpt returned non-JSON body (status {r.status_code})") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected cuOpt result type: {type(data).__name__}: {data}")
        return data

    # ---------- request/result mode ----------
    def _post_json(self, path: str, body: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        url = urljoin(self.server_url, path.lstrip("/"))
        r = requests.post(url, json=body,
                        headers={"Content-Type":"application/json","Accept":"application/json"},
                        timeout=timeout or self.max_time)
        r.raise_for_status()
        if "application/json" not in (r.headers.get("content-type","")):
            raise RuntimeError(f"Unexpected cuOpt response content-type: {r.headers.get('content-type')}; head={r.text[:120]!r}")
        return r.json()

    def _submit_request(self, payload: Dict[str, Any]) -> str:
        data = self._post_json(self._solve_path, payload, timeout=min(10, self.max_time))
        req_id = data.get("reqId") or data.get("requestId")
        if not req_id:
            raise RuntimeError(f"Missing reqId in /request response: keys={list(data.keys())}")
        return str(req_id)

    def _try_parse_body(self, resp: requests.Response) -> Dict[str, Any]:
        ct = resp.headers.get("content-type","")
        # JSON
        if "application/json" in ct:
            return resp.json()
        # MessagePack / octet-stream
        if ("application/octet-stream" in ct or "msgpack" in ct) and msgpack:
            try:
                return msgpack.unpackb(resp.content, raw=False)
            except Exception as e:
                return {"note":"msgpack-unpack-failed","error":str(e),"binary_len":len(resp.content)}
        # Plain text or unknown
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {"note":"non-json-text", "text_head": txt[:200], "content_type": ct}

    def _poll_result(self, req_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.max_time
        last = {"status_code": None, "content_type": None}

        # build all attempt combinations
        attempts = []
        for path in self._poll_paths:
            for method in self._poll_methods:
                for pkey in self._poll_params:
                    attempts.append((path, method, pkey))
            # also try path-param if requested
            if self._poll_use_path_param:
                attempts.append((path+"/{id}", "GET", None))
                attempts.append((path+"/{id}", "POST", None))

        headers = {"Accept":"application/json, application/octet-stream, */*"}
        while time.time() < deadline:
            for path, method, pkey in attempts:
                try:
                    url = urljoin(self.server_url, path.lstrip("/").replace("{id}", req_id))
                    if "{id}" in path:
                        # already embedded, no params
                        resp = requests.request(method, url, headers=headers, timeout=5)
                    elif method == "GET":
                        q = {pkey: req_id} if pkey else {}
                        resp = requests.get(url + (("?" + urlencode(q)) if q else ""), headers=headers, timeout=5)
                    else:
                        body = ({pkey: req_id} if pkey else {})
                        resp = requests.post(url, json=body, headers=headers, timeout=5)

                    last = {"status_code": resp.status_code, "content_type": resp.headers.get("content-type")}
                    if resp.status_code == 200:
                        return self._try_parse_body(resp)

                except Exception:
                    # ignore this attempt; move on
                    pass
            time.sleep(0.5)
        raise RuntimeError(f"cuOpt result polling exceeded max_solve_time_seconds; last={last}")


    def _parse_cuopt_solution(self, data: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Tolerant parser: try a few shapes.
        Expect something like:
          { "objective_value": 123, "assignments": [...] }
        or
          { "solution": { "objective": 123, "assignments": [...] } }
        or
          { "routes": [...], "objective": ... }
        """
        sol = data
        if "solution" in data and isinstance(data["solution"], dict):
            sol = data["solution"]

        # objective
        obj = None
        for k in ("objective_value", "objective", "cost", "score"):
            v = sol.get(k)
            if isinstance(v, (int, float)):
                obj = float(v)
                break
        if obj is None:
            obj = 0.0  # don’t crash the demo if objective missing

        # assignments
        assigns = None
        for k in ("assignments", "routes", "plan"):
            v = sol.get(k)
            if isinstance(v, list):
                assigns = v
                break
        if assigns is None:
            assigns = []

        return obj, assigns, {"raw": list(sol.keys())}

    # ---------- public solve ----------
def solve(self, disrupted_trips, candidates_per_trip, params=None) -> CuOptSolution:
    """
    Try cuOpt first (if available), otherwise greedy fallback.
    """
    payload = {
        "disrupted_trips": disrupted_trips,
        "candidates_per_trip": candidates_per_trip,
        "params": params or {},
    }

    note = None
    is_async = bool(self._solve_path and self._solve_path.strip("/").endswith("request"))

    if not self._force_greedy and self.server_url and self._solve_path:
        try:
            if is_async:
                req_id = self._submit_request(payload)
                data = self._poll_result(req_id)
                obj, assigns, meta = self._parse_cuopt_solution(data)
                return CuOptSolution(
                    objective_value=float(obj),
                    assignments=assigns,
                    details={"backend": "cuopt-async", "cuopt_solve_path": self._solve_path, **meta},
                )
            else:
                data = self._post_json(self._solve_path, payload, timeout=self.max_time + 5)
                obj, assigns, meta = self._parse_cuopt_solution(data)
                return CuOptSolution(
                    objective_value=float(obj),
                    assignments=assigns,
                    details={"backend": "cuopt", "cuopt_solve_path": self._solve_path, **meta},
                )
        except Exception as e:
            note = f"cuOpt HTTP failed: {e}"

    # ---------- greedy fallback ----------
    # obj, assigns = self._greedy_solve(disrupted_trips, candidates_per_trip)
    # return CuOptSolution(
    #     objective_value=float(obj),
    #     assignments=assigns,
    #     details={
    #         "backend": "greedy-miles+overtime",
    #         "note": note if note else "cuOpt disabled/undetected",
    #         "cuopt_solve_path": self._solve_path,
    #     },
    # )

    # # ---------- very simple greedy for demo ----------
    # def _greedy_solve(self, disrupted_trips, candidates_per_trip):
    #     total = 0.0
    #     all_assigns: List[Dict[str, Any]] = []
    #     for trip in disrupted_trips:
    #         tid = trip["id"]
    #         cands = candidates_per_trip.get(tid, []) or []
    #         # choose the feasible candidate with min 'est_cost', else outsource
    #         best = None
    #         for c in cands:
    #             if isinstance(c, dict):
    #                 feasible = c.get("feasible_hard", True)
    #                 if feasible and (best is None or c.get("est_cost", 1e18) < best.get("est_cost", 1e18)):
    #                     best = c
    #         if best:
    #             cost = float(best.get("est_cost", 0.0))
    #             total += cost
    #             all_assigns.append({
    #                 "trip_id": tid,
    #                 "type": "reassigned",
    #                 "driver_id": best.get("driver_id"),
    #                 "candidate_id": best.get("candidate_id"),
    #                 "delay_minutes": float(best.get("delay_minutes", 0.0)),
    #                 "uses_emergency_rest": bool(best.get("uses_emergency_rest", False)),
    #                 "deadhead_miles": float(best.get("deadhead_miles", 0.0)),
    #                 "overtime_minutes": float(best.get("overtime_minutes", 0.0)),
    #                 "miles_delta": float(best.get("miles_delta", 0.0)),
    #                 "cost": cost,
    #             })
    #         else:
    #             # outsource baseline = base + per mile, if provided
    #             base = float(self.cost.get("outsourcing_base_cost", 200.0))
    #             per_mi = float(self.cost.get("outsourcing_per_mile", self.cost.get("outsourcing_cost_per_mile", 2.0)))
    #             miles = float(trip.get("trip_miles", 0.0))
    #             cost = base + per_mi * miles
    #             total += cost
    #             all_assigns.append({
    #                 "trip_id": tid,
    #                 "type": "outsourced",
    #                 "driver_id": None,
    #                 "candidate_id": "OUTSOURCE",
    #                 "delay_minutes": 0.0,
    #                 "uses_emergency_rest": False,
    #                 "deadhead_miles": 0.0,
    #                 "overtime_minutes": 0.0,
    #                 "miles_delta": miles,
    #                 "cost": cost,
    #             })
    #     return total, all_assigns