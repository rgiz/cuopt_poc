# Refactor Guide (Feb 2026)

## Scope

This project now runs on a streamlined architecture with a single rich driver-state schema for planning.

- Canonical data contract: `RSL_SCHEMA.md`
- Primary objective achieved: modularized code paths with reduced duplication and cleaner runtime behavior.

## Current Architecture

### `backend/`

- FastAPI app assembly, data loading, route registration, and health/admin endpoints.
- Planning logic is delegated to `src/plan/*`.

### `src/plan/`

- `router.py`: endpoint orchestration (`/plan/candidates`, `/plan/solve_cascades`, `/plan/solve_multi`).
- `candidates.py` and `cascade_candidates.py`: candidate generation, filtering, and schedule reconstruction.
- `models.py`: contracts for request/response payloads.
- `geo.py`, `config.py`: geo lookups and policy/config loading.

### `src/`

- `driver_states_builder.py`: canonical rich `driver_states` generation.
- `driver_states_schema.py`: strict envelope normalization (`schema_version`, `drivers`).
- `rsl_helpers.py`: shared parsing/normalization helpers.
- `runtime.py`: logging and subprocess runtime utilities.

### `scripts/`

- Thin wrappers and orchestration only.
- Unified pipeline entrypoint: `scripts/run_data_pipeline.py`.
- `scripts/driver_states_prep.py` delegates to the shared builder.
- `scripts/pipeline.sh` delegates to the unified Python pipeline.

## Canonical Driver-State Contract

- Runtime planner contract is strict rich schema under envelope:
  - `{"schema_version": "2.0", "drivers": {...}}`
- Duty (driver-state) entries include:
  - duty-level fields (`duty_id`, `days`, `daily_windows`)
  - element-level schedule (`elements`)
  - planner metadata (`weekly_emergency_rest_quota`, `grade`, `vehicle_type`, etc.)
- Transition-only compatibility output fields were removed from canonical writer paths.

## Terminology

- Duty: a single planned duty record keyed by `duty_id`.
- Driver state: the runtime JSON representation of duties under `drivers`.
- Element: one schedule row within a duty (for example `START FACILITY`, `TRAVEL`, `END FACILITY`).
- Candidate: a feasible disruption-handling option produced by planner endpoints.

## Completed Refactor Outcomes

- Removed dead/commented blocks in key files (`ui/streamlit_app.py`, `scripts/build_dataset_from_rsl.py`).
- Extracted shared helper logic into `src/rsl_helpers.py`.
- Moved driver-state generation into reusable service code (`src/driver_states_builder.py`).
- Consolidated data pipeline entrypoint and script delegation.
- Added/maintained API compatibility features (`/plan/solve_multi`, outsourced fallback behavior).
- Standardized logging setup with targeted suppression of non-actionable cuOpt cleanup noise.

## Pipeline Usage

- Standard run:
  - `python scripts/run_data_pipeline.py --data-dir /data`
- Skip backend reload:
  - `python scripts/run_data_pipeline.py --data-dir /data --skip-reload`
- Make target:
  - `make pipeline DATA_DIR=/data`

## Logging Controls

- API log level: `LOG_LEVEL`
- API benign cuOpt noise suppression: `SUPPRESS_BENIGN_CUOPT_LOGS`
- cuOpt container verbosity: `CUOPT_SERVER_LOG_LEVEL`

## Validation Snapshot

- Unit tests: `13 passed`.
- Known non-blocking warning: Starlette lifespan deprecation warning from async-generator lifespan usage.

## Status

- Refactor program goals for modularity, DRY structure, and canonical schema enforcement are complete for this phase.
