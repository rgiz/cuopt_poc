# cuOpt Runbook (Windows + Docker)

## Purpose

Operational guide to run and verify cuOpt-backed planning in this repository.

## Key Principle

Use **Docker runtime** for cuOpt behavior validation.

- Local in-process Python on Windows 3.13 may fail to install NVIDIA parser wheels.
- cuOpt integration is reliable through the Docker `api` + `cuopt` services.

## Preflight

1. Start Docker Desktop.
2. Confirm WSL is healthy:
   - `wsl -l -v`
3. Confirm Docker engine is healthy:
   - `docker version`

## Start Services

From repository root:

- `docker compose up -d cuopt api ui`
- `docker compose ps`

Expected:

- `api` running on `localhost:8000`
- `cuopt` running on `localhost:5000`
- `ui` running on `localhost:8501`

## Health Checks

### 1) API-level cuOpt selftest (authoritative)

- `Invoke-WebRequest -UseBasicParsing http://localhost:8000/admin/cuopt_selftest`

Expected JSON fields:

- `ok: true`
- `status: 0`
- message indicating healthy cuOpt connectivity

### 2) Planning endpoint runtime status

`/plan/solve_cascades` and `/plan/solve_multi` now include:

- `details.cuopt_runtime` (or `meta.cuopt_runtime`)
- fields:
  - `client_imported`
  - `health_check`

Use these to verify whether responses are cuOpt-capable vs fallback-only.

## Smoke Test (Live API)

Run:

- `python scripts/smoke_plan_endpoints.py --api-base-url http://localhost:8000 --start-location "BIRMINGHAM MAIL CENTRE" --end-location "MIDLANDS SUPER HUB" --when-local "2025-09-02T10:30" --priority 3 --mode depart_after --output _tmp_cuopt_live_smoke.json`

Expected outcome for a healthy setup:

- status codes `200`
- reassigned counts > 0 for at least one solve path
- no validation errors in report

## Recommended Daily Workflow

1. `docker compose up -d cuopt api ui`
2. `/admin/cuopt_selftest`
3. Run smoke script in **live mode** (`--api-base-url`)
4. Exercise UI and capture scenario outputs

## Strict Mode (Require cuOpt)

Use strict mode in environments where fallback-to-outsourcing is not acceptable.

- Set env var: `REQUIRE_CUOPT=true`

Behavior:

- `/plan/solve_cascades` and `/plan/solve_multi` return `503` if cuOpt runtime is unavailable.
- `detail` payload includes `cuopt_runtime` with:
  - `client_imported`
  - `health_check`

Use this mode in staging/prod to fail fast on cuOpt outages.

## Troubleshooting

### Problem: always getting outsourced fallback

Check in order:

1. `/admin/cuopt_selftest` result
2. response `cuopt_runtime` status fields
3. `docker compose logs api --tail 200`
4. `docker compose logs cuopt --tail 200`

### Problem: cuOpt container unhealthy

- Inspect logs:
  - `docker compose logs cuopt --tail 300`
- Restart stack:
  - `docker compose restart cuopt api`
- Re-test selftest endpoint.

### Problem: local in-process tests show cuOpt unavailable

This is expected on Windows+Py3.13 with unavailable NVIDIA parser wheel.
Use Docker live mode for cuOpt validation.

## Notes

- Keep in-process tests for pure logic/regression checks.
- Use live Docker API tests for cuOpt performance/behavior checks.
