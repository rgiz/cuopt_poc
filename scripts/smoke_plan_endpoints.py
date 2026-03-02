#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass
from importlib import reload
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class HttpResponse:
    status_code: int
    body: dict[str, Any]
    text: str


def _http_json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> HttpResponse:
    data: bytes | None = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(url=url, method=method, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            try:
                body = json.loads(text) if text else {}
            except Exception:
                body = {}
            return HttpResponse(status_code=int(response.status), body=body, text=text)
    except HTTPError as exc:
        text = exc.read().decode("utf-8") if exc.fp else ""
        try:
            body = json.loads(text) if text else {}
        except Exception:
            body = {}
        return HttpResponse(status_code=int(exc.code), body=body, text=text)
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


class LiveClient:
    def __init__(self, api_base_url: str, timeout: int):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> HttpResponse:
        return _http_json_request("GET", f"{self.api_base_url}{path}", timeout=self.timeout)

    def post(self, path: str, payload: dict[str, Any]) -> HttpResponse:
        return _http_json_request("POST", f"{self.api_base_url}{path}", payload=payload, timeout=self.timeout)


class InProcessClient:
    def __init__(self):
        from fastapi.testclient import TestClient
        import backend.main_miles as mm

        mm = reload(mm)
        self._client = TestClient(mm.app)

    def get(self, path: str) -> HttpResponse:
        response = self._client.get(path)
        try:
            body = response.json()
        except Exception:
            body = {}
        return HttpResponse(status_code=int(response.status_code), body=body, text=response.text)

    def post(self, path: str, payload: dict[str, Any]) -> HttpResponse:
        response = self._client.post(path, json=payload)
        try:
            body = response.json()
        except Exception:
            body = {}
        return HttpResponse(status_code=int(response.status_code), body=body, text=response.text)


def _require_keys(obj: dict[str, Any], keys: list[str], label: str) -> list[str]:
    missing = [key for key in keys if key not in obj]
    if missing:
        return [f"{label}: missing keys {missing}"]
    return []


def _pick_locations(client: LiveClient | InProcessClient, start_location: str | None, end_location: str | None) -> tuple[str, str]:
    if start_location and end_location and start_location != end_location:
        return start_location, end_location

    locations_resp = client.get("/plan/locations")
    if locations_resp.status_code != 200:
        raise RuntimeError(f"GET /plan/locations failed: {locations_resp.status_code} {locations_resp.text}")

    names = [str(item.get("name", "")).strip() for item in locations_resp.body.get("locations", [])]
    names = [name for name in names if name]
    if len(names) < 2:
        raise RuntimeError("Need at least two valid locations from /plan/locations")

    start = start_location or names[0]
    end = end_location or next((name for name in names if name != start), names[1])
    if start == end:
        raise RuntimeError("Unable to choose distinct start and end locations")
    return start, end


def run_smoke(
    *,
    api_base_url: str,
    in_process: bool,
    timeout_seconds: int,
    start_location: str | None,
    end_location: str | None,
    when_local: str,
    priority: int,
    mode: str,
    output_path: str | None,
) -> int:
    client: LiveClient | InProcessClient
    client = InProcessClient() if in_process else LiveClient(api_base_url=api_base_url, timeout=timeout_seconds)

    reload_resp = client.post("/admin/reload", payload={})
    if reload_resp.status_code not in {200, 204}:
        print(f"FAIL: POST /admin/reload returned {reload_resp.status_code}")
        return 2

    start, end = _pick_locations(client, start_location, end_location)

    cascades_payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": when_local,
        "priority": priority,
        "max_cascades": 2,
        "max_drivers_affected": 3,
    }

    multi_payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": when_local,
        "priority": priority,
        "top_n_per_step": 3,
        "max_cascades": 2,
        "max_drivers_affected": 3,
        "max_solutions": 5,
        "use_cuopt": False,
    }

    cascades_resp = client.post("/plan/solve_cascades", payload=cascades_payload)
    multi_resp = client.post("/plan/solve_multi", payload=multi_payload)

    errors: list[str] = []
    if cascades_resp.status_code != 200:
        errors.append(f"/plan/solve_cascades returned {cascades_resp.status_code}")
    if multi_resp.status_code != 200:
        errors.append(f"/plan/solve_multi returned {multi_resp.status_code}")

    cascades_details = cascades_resp.body.get("details", {}) if isinstance(cascades_resp.body, dict) else {}
    cascades_diag = cascades_details.get("cascade_diagnostics", {}) if isinstance(cascades_details, dict) else {}
    cascades_perf = cascades_details.get("performance", {}) if isinstance(cascades_details, dict) else {}

    multi_meta = multi_resp.body.get("meta", {}) if isinstance(multi_resp.body, dict) else {}
    multi_diag = multi_meta.get("cascade_diagnostics", {}) if isinstance(multi_meta, dict) else {}
    multi_perf = multi_meta.get("performance", {}) if isinstance(multi_meta, dict) else {}

    errors.extend(
        _require_keys(
            cascades_diag,
            [
                "candidates_total",
                "feasible_hard_count",
                "max_chain_depth",
                "avg_chain_depth",
                "unresolved_total",
                "reason_code_counts",
            ],
            "solve_cascades.cascade_diagnostics",
        )
    )
    errors.extend(
        _require_keys(
            cascades_perf,
            [
                "total_ms",
                "ensure_ready_ms",
                "candidate_generation_ms",
                "assignment_build_ms",
                "cascade_build_ms",
                "postprocess_ms",
            ],
            "solve_cascades.performance",
        )
    )
    errors.extend(
        _require_keys(
            multi_diag,
            [
                "candidates_total",
                "feasible_hard_count",
                "max_chain_depth",
                "avg_chain_depth",
                "unresolved_total",
                "reason_code_counts",
            ],
            "solve_multi.cascade_diagnostics",
        )
    )
    errors.extend(
        _require_keys(
            multi_perf,
            [
                "total_ms",
                "ensure_ready_ms",
                "candidate_generation_ms",
                "solution_build_ms",
            ],
            "solve_multi.performance",
        )
    )

    summary = {
        "api_base_url": api_base_url,
        "in_process": in_process,
        "route": {"start_location": start, "end_location": end},
        "inputs": {"when_local": when_local, "priority": priority, "mode": mode},
        "status_codes": {
            "solve_cascades": cascades_resp.status_code,
            "solve_multi": multi_resp.status_code,
        },
        "reassigned_counts": {
            "solve_cascades": sum(
                1
                for assignment in (cascades_resp.body.get("assignments", []) if isinstance(cascades_resp.body, dict) else [])
                if str(assignment.get("type", "")).lower() == "reassigned"
            ),
            "solve_multi": sum(
                1
                for solution in (multi_resp.body.get("solutions", []) if isinstance(multi_resp.body, dict) else [])
                for assignment in solution.get("assignments", [])
                if str(assignment.get("type", "")).lower() == "reassigned"
            ),
        },
        "errors": errors,
    }

    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote smoke report: {out}")

    if errors:
        print("FAIL: backend smoke checks failed")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("PASS: backend smoke checks passed")
    print(f"Route: {start} -> {end}")
    print(
        "Reassigned counts: "
        f"solve_cascades={summary['reassigned_counts']['solve_cascades']}, "
        f"solve_multi={summary['reassigned_counts']['solve_multi']}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend smoke test for planning endpoints")
    parser.add_argument("--api-base-url", default="http://localhost:8000", help="Base URL for live backend")
    parser.add_argument("--in-process", action="store_true", help="Use in-process FastAPI TestClient instead of live HTTP")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--start-location", default=None)
    parser.add_argument("--end-location", default=None)
    parser.add_argument("--when-local", default="2025-09-02T10:30")
    parser.add_argument("--priority", type=int, default=2)
    parser.add_argument("--mode", default="depart_after", choices=["depart_after", "arrive_before"])
    parser.add_argument("--output", default="artifacts/bench/smoke_report.json")
    args = parser.parse_args()

    code = run_smoke(
        api_base_url=args.api_base_url,
        in_process=bool(args.in_process),
        timeout_seconds=max(1, int(args.timeout_seconds)),
        start_location=args.start_location,
        end_location=args.end_location,
        when_local=args.when_local,
        priority=max(1, min(5, int(args.priority))),
        mode=args.mode,
        output_path=args.output,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
