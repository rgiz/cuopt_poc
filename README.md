# Dynamic Trip Rescheduling (Docker + cuOpt)

This repository provides a disruption-planning tool for adding a new trip into existing duty schedules and returning feasible reassignment options.

It includes:
- FastAPI backend for planning APIs
- Streamlit UI for operators
- NVIDIA cuOpt service for optimization (with fallback behavior when unavailable)
- Data pipeline scripts to build runtime artifacts from cleaned RSL inputs

## What the tool currently does

Given a new trip request (`start`, `end`, `when_local`, `mode`, `priority`), the tool:
- Loads prebuilt runtime artifacts (`driver_states.json`, location index, time/distance matrices)
- Finds candidate drivers and reconstructs before/after schedules
- Applies continuity and home-base return checks in candidate reconstruction paths
- Supports cascade-style reassignment outputs through solve endpoints
- Returns structured diagnostics for filtering/cascade behavior and runtime timing

UI behavior:
- Shows option list, map overlay, before/after duty tables, and cascade diagnostics
- Price/cost visibility is intentionally removed from UI display

## Architecture

- `backend/`: FastAPI app and settings/admin routes
- `src/plan/`: planning logic, candidate generation, cascade reconstruction, API models/router
- `src/`: shared data/build helpers
- `scripts/`: data pipeline and smoke/benchmark utilities
- `ui/`: Streamlit interface
- `data/private/active/`: active runtime dataset consumed by backend

## cuOpt in this tool

cuOpt is used as the optimizer in planning flows where available.

Runtime behavior:
- Backend checks cuOpt runtime availability (`client_imported`, `health_check`)
- Solve endpoints include cuOpt runtime status in responses
- If cuOpt is unavailable and strict mode is not enabled, workflows can fall back to non-cuOpt behavior
- If `REQUIRE_CUOPT=true`, solve endpoints fail fast with `503` when cuOpt runtime is unavailable

Main solve endpoints:
- `POST /plan/candidates`
- `POST /plan/solve_cascades`
- `POST /plan/solve_multi`

## Quick start (Docker)

Prerequisites:
- Docker Desktop
- NVIDIA GPU + compatible drivers/toolkit for cuOpt usage

Start services:

```bash
docker compose up -d cuopt api ui
```

Service URLs:
- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- cuOpt: `http://localhost:5000`

Health checks:

```bash
# API health
curl http://localhost:8000/health

# cuOpt selftest via API
curl http://localhost:8000/admin/cuopt_selftest
```

## Data pipeline (runtime artifacts)

Build/refresh active dataset artifacts:

```bash
python scripts/run_data_pipeline.py --data-dir /data
```

Expected active artifacts:
- `driver_states.json`
- `location_index.csv`
- `distance_miles_matrix.npz`
- `time_minutes_matrix.npz`

## Recommended operational workflow

1. `docker compose up -d cuopt api ui`
2. Verify `/health` and `/admin/cuopt_selftest`
3. Run smoke test:

```bash
python scripts/smoke_plan_endpoints.py --api-base-url http://localhost:8000 --output artifacts/bench/smoke_report.json
```

4. Use UI to evaluate options and schedule diffs

## Additional docs

- Runtime data contract: [RSL_SCHEMA.md](RSL_SCHEMA.md)
- cuOpt operational runbook: [docs/cuopt_runbook.md](docs/cuopt_runbook.md)
