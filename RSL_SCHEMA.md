# RSL Schema and Planning Contract

This document defines the current source of truth for RSL-derived runtime data used by planning endpoints and UI outputs.

## Runtime Artifacts

The planner executes against derived artifacts, not raw RSL rows:

- `driver_states.json`
- `location_index.csv`
- `distance_miles_matrix.npz`
- `time_minutes_matrix.npz`

## Terminology

- Duty: a single planned duty record keyed by `duty_id`.
- Driver state: the runtime JSON representation of duties under `drivers`.
- Element: one schedule row within a duty (for example `START FACILITY`, `TRAVEL`, `END FACILITY`).
- Candidate: a feasible disruption-handling option produced by planner endpoints.

## End-to-End Runtime Flow

1. Cleaned RSL input is ingested.
2. Location index and time/distance matrices are generated.
3. Driver states are generated in canonical rich schema.
4. Backend loads artifacts at startup or via `/admin/reload`.
5. Planner endpoints (`/plan/candidates`, `/plan/solve_cascades`, `/plan/solve_multi`) consume this contract.

## Canonical `driver_states.json` Contract

Top-level envelope:

```json
{
  "schema_version": "2.0",
  "drivers": {
    "<duty_id>": {
      "duty_id": "<string>",
      "days": ["Mon", "Tue", "..."],
      "daily_windows": {
        "Mon": {
          "start_min": 1304,
          "end_min": 1685,
          "crosses_midnight": true
        }
      },
      "elements": [
        {
          "element_type": "START FACILITY|TRAVEL|LOAD(...)|UNLOAD(...)|END FACILITY|...",
          "is_travel": true,
          "start": "HH:MM:SS",
          "start_min": 1354,
          "end": "HH:MM:SS",
          "end_min": 1379,
          "from": "LOCATION NAME",
          "to": "LOCATION NAME",
          "from_id": 36,
          "to_id": 40,
          "miles": 13.0,
          "duration_min": 25,
          "load_type": "CONTAINER REPATRIATION|NO_DATA|...",
          "priority": 1,
          "Mon": 1,
          "Tue": 1,
          "Wed": 1,
          "Thu": 1,
          "Fri": 1,
          "Sat": 0,
          "Sun": 0
        }
      ],
      "weekly_emergency_rest_quota": 2,
      "grade": null,
      "vehicle_type": null,
      "home_center_id": null
    }
  }
}
```

## Field Semantics

### Duty-level

- `drivers.<duty_id>`: duty object keyed by Duty ID.
- `days`: active weekdays.
- `daily_windows.<day>.start_min` / `end_min`: duty window in minutes from midnight; values may exceed 1440 for cross-midnight duties.
- `daily_windows.<day>.crosses_midnight`: explicit cross-midnight indicator.

### Element-level

- `element_type`: normalized RSL element label.
- `is_travel`: travel-leg discriminator used by filtering and schedule reconstruction.
- `from`, `to`: normalized location names.
- `from_id`, `to_id`: matrix-space location indices from `location_index.csv`.
- `start_min`, `end_min`: element-local timing in minutes.
- `priority`: planner priority used in candidate scoring/displacement logic.
- weekday flags (`Mon..Sun`): per-element active-day flags.

## Operational Planning Rules

- Duties with invalid travel locations are excluded from candidate generation.
- Weekday filtering uses element/day flags plus duty-level windows.
- Cross-midnight windows are handled explicitly.
- RSL-aware reconstruction preserves `START FACILITY` / `END FACILITY` structure while inserting or appending disruption work.

## Producer Ownership

- `scripts/run_data_pipeline.py`: orchestrates artifact generation.
- `scripts/driver_states_prep.py`: emits canonical driver-state envelope.
- `src/driver_states_builder.py`: constructs rich duty windows and element records.

## Consumer Ownership

- `backend/main_miles.py`: artifact loading and normalization.
- `src/plan/candidates.py`: candidate filtering/scoring.
- `src/plan/cascade_candidates.py`: cascade candidate generation and schedule reconstruction.
- `src/plan/router.py`: API response orchestration.
- `ui/streamlit_app.py`: UI rendering of candidate and schedule outputs.

## Versioning Rule

- `schema_version` is required at envelope level.
- Any future breaking contract change must increment schema version and include an explicit migration/adapter path.
