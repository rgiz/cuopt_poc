#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    k = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return float(sorted(values)[k])


def _discover_locations(client: TestClient) -> list[str]:
    locations_resp = client.get("/plan/locations")
    locations_resp.raise_for_status()
    locations_payload = locations_resp.json()
    location_names = sorted(
        [str(item.get("name", "")).strip() for item in locations_payload.get("locations", []) if str(item.get("name", "")).strip()]
    )
    if len(location_names) < 2:
        raise RuntimeError("Need at least two locations from /plan/locations to run benchmark")
    return location_names


def _count_candidates(
    client: TestClient,
    *,
    start_location: str,
    end_location: str,
    when_local: str,
    priority: int,
    mode: str,
) -> int:
    payload = {
        "start_location": start_location,
        "end_location": end_location,
        "mode": mode,
        "when_local": when_local,
        "priority": priority,
        "max_cascades": 2,
        "max_drivers_affected": 3,
    }
    resp = client.post("/plan/solve_cascades", json=payload)
    resp.raise_for_status()
    body = resp.json()
    assignments = body.get("assignments", []) or []
    return sum(1 for assignment in assignments if str(assignment.get("type", "")).lower() == "reassigned")


def _find_candidate_route(
    client: TestClient,
    *,
    when_locals: list[str],
    priorities: list[int],
    mode: str,
    max_pairs: int,
    seed: int,
) -> tuple[str, str, int, str, int, int] | None:
    location_names = _discover_locations(client)
    if len(location_names) < 2:
        return None

    rnd = random.Random(seed)
    sampled_locations = location_names[:]
    rnd.shuffle(sampled_locations)
    sampled_locations = sampled_locations[: min(len(sampled_locations), 150)]

    combos = [(w, p) for w in when_locals for p in priorities]
    rnd.shuffle(combos)

    tried = 0
    for when_local, priority in combos:
        for i, start_location in enumerate(sampled_locations):
            for end_location in sampled_locations[i + 1 :]:
                tried += 1
                if tried > max_pairs:
                    return None
                try:
                    count = _count_candidates(
                        client,
                        start_location=start_location,
                        end_location=end_location,
                        when_local=when_local,
                        priority=priority,
                        mode=mode,
                    )
                except Exception:
                    continue
                if count > 0:
                    return start_location, end_location, count, when_local, priority, tried
    return None


def _run_profile(
    client: TestClient,
    *,
    label: str,
    iterations: int,
    start_location: str,
    end_location: str,
    when_local: str,
    priority: int,
    mode: str,
) -> dict[str, object]:
    initial_candidates = _count_candidates(
        client,
        start_location=start_location,
        end_location=end_location,
        when_local=when_local,
        priority=priority,
        mode=mode,
    )

    cascades_payload = {
        "start_location": start_location,
        "end_location": end_location,
        "when_local": when_local,
        "priority": priority,
        "mode": mode,
        "max_cascades": 2,
        "max_drivers_affected": 3,
    }

    multi_payload = {
        "start_location": start_location,
        "end_location": end_location,
        "mode": mode,
        "when_local": when_local,
        "priority": priority,
        "top_n_per_step": 3,
        "max_cascades": 2,
        "max_drivers_affected": 3,
        "max_solutions": 5,
        "use_cuopt": False,
    }

    cascades_ms: list[float] = []
    multi_ms: list[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        rc = client.post("/plan/solve_cascades", json=cascades_payload)
        rc.raise_for_status()
        cascades_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        rm = client.post("/plan/solve_multi", json=multi_payload)
        rm.raise_for_status()
        multi_ms.append((time.perf_counter() - t1) * 1000.0)

    print("Benchmark results (ms)")
    print(f"profile: {label}")
    print(f"route: {start_location} -> {end_location}")
    print(f"initial_candidate_count: {initial_candidates}")
    print(
        f"solve_cascades: n={len(cascades_ms)}, mean={statistics.mean(cascades_ms):.2f}, "
        f"p50={_percentile(cascades_ms, 50):.2f}, p95={_percentile(cascades_ms, 95):.2f}"
    )
    print(
        f"solve_multi:    n={len(multi_ms)}, mean={statistics.mean(multi_ms):.2f}, "
        f"p50={_percentile(multi_ms, 50):.2f}, p95={_percentile(multi_ms, 95):.2f}"
    )

    return {
        "profile": label,
        "route_start": start_location,
        "route_end": end_location,
        "when_local": when_local,
        "priority": int(priority),
        "mode": mode,
        "iterations": int(iterations),
        "initial_candidate_count": int(initial_candidates),
        "solve_cascades_mean_ms": round(statistics.mean(cascades_ms), 3),
        "solve_cascades_p50_ms": round(_percentile(cascades_ms, 50), 3),
        "solve_cascades_p95_ms": round(_percentile(cascades_ms, 95), 3),
        "solve_multi_mean_ms": round(statistics.mean(multi_ms), 3),
        "solve_multi_p50_ms": round(_percentile(multi_ms, 50), 3),
        "solve_multi_p95_ms": round(_percentile(multi_ms, 95), 3),
    }


def _write_outputs(results: list[dict[str, object]], json_out: str | None, csv_out: str | None) -> None:
    if json_out:
        json_path = Path(json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "profiles": results,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote_json: {json_path}")

    if csv_out:
        csv_path = Path(csv_out).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not results:
            csv_path.write_text("", encoding="utf-8")
            print(f"wrote_csv: {csv_path}")
            return
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"wrote_csv: {csv_path}")


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for token in _parse_csv_values(raw):
        try:
            values.append(int(token))
        except Exception:
            continue
    return values


def run_benchmark(
    iterations: int,
    *,
    start_location: str | None,
    end_location: str | None,
    when_local: str,
    priority: int,
    mode: str,
    include_candidate_profile: bool,
    search_max_pairs: int,
    search_seed: int,
    candidate_when_locals: list[str],
    candidate_priorities: list[int],
    json_out: str | None,
    csv_out: str | None,
) -> None:
    import backend.main_miles as mm

    mm = reload(mm)
    client = TestClient(mm.app)
    reload_resp = client.post("/admin/reload")
    if reload_resp.status_code not in {200, 204}:
        raise RuntimeError(f"/admin/reload failed: {reload_resp.status_code} {reload_resp.text}")

    discovered_locations = _discover_locations(client)
    selected_start = start_location or discovered_locations[0]
    selected_end = end_location or next((name for name in discovered_locations if name != selected_start), discovered_locations[1])

    results: list[dict[str, object]] = []

    results.append(_run_profile(
        client,
        label="primary",
        iterations=iterations,
        start_location=selected_start,
        end_location=selected_end,
        when_local=when_local,
        priority=priority,
        mode=mode,
    ))

    if include_candidate_profile:
        found = _find_candidate_route(
            client,
            when_locals=candidate_when_locals,
            priorities=candidate_priorities,
            mode=mode,
            max_pairs=search_max_pairs,
            seed=search_seed,
        )
        if found is None:
            print("Candidate profile search: no candidate-producing route found within search budget")
        else:
            cand_start, cand_end, count, cand_when_local, cand_priority, attempts = found
            print(
                f"Candidate profile search: found route with {count} candidates "
                f"after {attempts} attempts (when_local={cand_when_local}, priority={cand_priority})"
            )
            results.append(_run_profile(
                client,
                label="candidate-producing",
                iterations=iterations,
                start_location=cand_start,
                end_location=cand_end,
                when_local=cand_when_local,
                priority=cand_priority,
                mode=mode,
            ))

    _write_outputs(results, json_out=json_out, csv_out=csv_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark planning solve endpoints")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--start-location", type=str, default=None)
    parser.add_argument("--end-location", type=str, default=None)
    parser.add_argument("--when-local", type=str, default="2025-09-02T10:30")
    parser.add_argument("--priority", type=int, default=2)
    parser.add_argument("--mode", type=str, default="depart_after", choices=["depart_after", "arrive_before"])
    parser.add_argument("--include-candidate-profile", action="store_true")
    parser.add_argument("--search-max-pairs", type=int, default=300)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument(
        "--candidate-when-locals",
        type=str,
        default="2025-09-02T06:30,2025-09-02T10:30,2025-09-02T14:30,2025-09-02T18:30,2025-09-02T22:30",
        help="Comma-separated when_local values used during candidate-profile auto discovery",
    )
    parser.add_argument(
        "--candidate-priorities",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated priority values used during candidate-profile auto discovery",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output file path")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV output file path")
    args = parser.parse_args()

    candidate_when_locals = _parse_csv_values(args.candidate_when_locals)
    if not candidate_when_locals:
        candidate_when_locals = [args.when_local]

    candidate_priorities = [p for p in _parse_int_csv(args.candidate_priorities) if 1 <= p <= 5]
    if not candidate_priorities:
        candidate_priorities = [args.priority]

    run_benchmark(
        args.iterations,
        start_location=args.start_location,
        end_location=args.end_location,
        when_local=args.when_local,
        priority=args.priority,
        mode=args.mode,
        include_candidate_profile=args.include_candidate_profile,
        search_max_pairs=max(1, args.search_max_pairs),
        search_seed=args.search_seed,
        candidate_when_locals=candidate_when_locals,
        candidate_priorities=candidate_priorities,
        json_out=args.json_out,
        csv_out=args.csv_out,
    )


if __name__ == "__main__":
    main()
