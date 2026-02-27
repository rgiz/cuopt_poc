# Implementation Plan: Advanced Duty Insertion, Cascade Logic, and Constraint Enforcement

## Objective

Deliver a production-safe enhancement of planning behavior to support:

- As Directed spare-time utilization with continuity-aware insertion.
- Exact and partial route-overlap substitutions.
- Nearby-depot substitutions (pickup/drop neighborhood logic).
- Hard legal and home-base constraints as true feasibility gates.
- Multi-step cascading of displaced low-priority work.

This plan is staged to preserve service stability while improving solution quality.

## Inputs

- Gap analysis: GAP_TO_FEATURE_MATRIX.md
- Data contract: RSL_SCHEMA.md
- Runtime architecture: REFACTOR_GUIDE.md

---

## Delivery Strategy

- Implement in progressive phases with feature flags and scenario-based parity checks.
- Keep CPU as policy/validation orchestrator; use GPU/CUDA only for numeric hotspots and cuOpt solve stage.
- Ship observable increments with explicit acceptance gates before advancing.

---

## Phase 0 - Foundation and Guardrails (1-2 days)

## Goals

- Lock target behavior and prevent regression during iteration.

## Tasks

1. Add planning feature flags

- ENABLE_AS_DIRECTED_INSERTION
- ENABLE_PARTIAL_OVERLAP_INSERTION
- ENABLE_NEARBY_DEPOT_SUBSTITUTION
- ENABLE_STRICT_LEGAL_CONSTRAINTS
- ENABLE_TRUE_CASCADE

2. Establish scenario test pack (golden tests)

- As Directed insertion (with downstream commitment).
- Exact empty/low-priority replacement A->B.
- Pre-pickup overlap (A->C empty + C->B loaded).
- Post-drop overlap (A->C loaded + C->B empty).
- Nearby-neighbor substitution (A~C and B~D).
- Home-base return hard-fail case.
- Legal-hours hard-fail case.

3. Add explainability reason codes scaffold

- RULE\_\* reject/accept codes at each stage.

## Acceptance Gate

- Existing unit tests pass.
- New scenario tests are in place and initially marked expected-fail where feature is not yet implemented.

---

## Phase 1 - Candidate Taxonomy and Priority Policy (3-5 days)

## Goals

- Introduce deterministic candidate families and unified precedence rules.

## Tasks

1. Centralize precedence model

- New load may displace equal/lower-priority planned work.
- Empty/deadhead always displaceable.
- Higher-priority committed work is protected.

2. Implement candidate families

- exact_replacement
- as_directed_insertion
- overlap_prepickup
- overlap_postdrop
- nearby_neighbor_substitution
- append_fallback (lowest tier)

3. Add candidate-family metadata

- candidate_family
- displaced_element_ids
- continuity_anchor_before/after

## Primary Modules

- src/plan/candidates.py
- src/plan/models.py

## Acceptance Gate

- Scenario tests for exact and overlap families pass for candidate generation stage.
- Candidate ranking stable and policy-compliant.

---

## Phase 2 - Duty Continuity and Reconstruction Engine (3-5 days)

## Goals

- Guarantee physically and temporally valid before/after schedules for each candidate family.

## Tasks

1. Build continuity validator

- Time monotonicity.
- Location continuity between consecutive elements.
- Explicit waits allowed; impossible teleports disallowed.
- End-state verification against expected anchor.

2. Implement explicit As Directed reconstruction

- Remove current implicit fallback behavior.
- Construct insertion and rejoin segments with timing updates.

3. Implement overlap reconstruction rules

- prepickup: A->C empty, C->B loaded, rejoin.
- postdrop: A->C loaded, C->B empty, rejoin.

4. Add schedule diff annotations

- ADDED, REPLACED, SHIFTED, DISPLACED, REJOINED.

## Primary Modules

- src/plan/cascade_candidates.py
- src/plan/candidates.py

## Acceptance Gate

- All produced schedules pass continuity validator.
- UI-visible schedules are coherent and match expected scenarios.

---

## Phase 3 - Hard Constraint Engine (3-4 days)

## Goals

- Move hard constraints from coarse approximations to enforceable feasibility gates.

## Tasks

1. Implement legal constraint checks

- Driving minutes cap.
- Duty/on-shift cap.
- Required breaks/rest windows.
- Cross-midnight legal handling.

2. Enforce home-base return

- Candidate is hard-infeasible unless duty end-state returns to home base (unless explicitly configured otherwise).

3. Integrate constraint checks into feasible_hard

- Apply after reconstruction and before ranking/output.

## Primary Modules

- src/plan/candidates.py
- src/plan/cascade_candidates.py
- configuration/settings routing for policy parameters

## Acceptance Gate

- Hard-fail scenarios are rejected deterministically.
- No candidate marked feasible_hard violates legal or home-base constraints.

---

## Phase 4 - True Cascade Engine (4-6 days)

## Goals

- Reassign displaced low-priority work across other duties with bounded search.

## Tasks

1. Build displacement graph model

- Track displaced tasks, candidate receiving duties, and depth transitions.

2. Implement bounded cascade search

- Branch factor control.
- Max depth control.
- Cost/feasibility pruning.

3. Add cascade objective aggregation

- Total system cost + policy penalties + uncovered work penalties.

4. Extend API detail outputs

- cascade_chain with cause/effect by depth.
- structured diagnostics summaries for UI/ops consumers.

## Primary Modules

- src/plan/cascade_candidates.py
- src/plan/router.py
- src/plan/models.py

## Acceptance Gate

- Displaced low-priority tasks are reassigned when feasible.
- Multi-step cascades returned with valid chain provenance.
- `/plan/solve_cascades` and `/plan/solve_multi` include diagnostics aggregates without requiring reason-string parsing.

## Implemented Diagnostics Contract (Current State)

The API now emits structured cascade diagnostics to support UI rendering and operations analysis.

- `details.cascade_diagnostics` on `/plan/solve_cascades` (including outsourced fallback).
- `meta.cascade_diagnostics` on `/plan/solve_multi`.
- Per-cascade entries include:
  - `reason_code`
  - `reason_detail`
  - `assigned_steps`
  - `blocked_steps`
  - `depth`
- Aggregates currently include:
  - `candidates_total`
  - `feasible_hard_count`
  - `max_chain_depth`
  - `avg_chain_depth`
  - `unresolved_total`
  - `uncovered_p4_total`
  - `disposed_p5_total`
  - `reason_code_counts`

Validation status:

- Unit coverage for diagnostics parsing/aggregation is in place.
- Integration contract tests for `/plan/solve_cascades` and `/plan/solve_multi` payload shape are in place.

---

## Phase 5 - CUDA/GPU Acceleration (Optional but Recommended, 3-5 days)

## Goals

- Accelerate numeric hotspots without moving policy branching off CPU.

## Tasks

1. Profile baseline by stage

- filtering, expansion, scoring, reconstruction, solve.

2. GPU-enable numeric hotspots

- batched matrix reachability/time-slack computations.
- batched neighborhood checks.

3. Keep CPU for policy and legal validation

- avoid branch-heavy logic in kernels.

4. Add adaptive execution thresholds

- CPU path for small batches, GPU path for large batches.

5. Add parity and drift tests

- CPU/GPU output parity for same scenario set.

## Acceptance Gate

- Measured latency improvement on representative workloads.
- No feasibility/policy drift between CPU and GPU modes.

---

## Phase 6 - Rollout, Observability, and Cleanup (2-3 days)

## Goals

- Deploy safely, monitor behavior, and remove obsolete code paths.

## Tasks

1. Staged rollout

- Internal test mode -> canary -> full production enablement.

2. Observability

- per-rule reject counts.
- candidate family distribution.
- cascade depth distribution.
- feasible_hard failure reasons.
- solve latency by stage.

3. Post-rollout cleanup

- remove deprecated candidate branches.
- remove dead compatibility branches.
- align docs to final behavior and flags.

## Acceptance Gate

- KPI targets met (quality and latency).
- No critical regressions over agreed soak period.
- Redundant code paths removed.

---

## Testing Plan (Continuous Through All Phases)

1. Unit tests

- candidate family generation
- precedence policy
- continuity validator
- legal/home-base checks

2. Integration tests

- end-to-end endpoint behavior for representative scenarios

3. Golden scenario tests

- exact substitution, overlap variants, As Directed, nearby depots, cascades

4. Performance tests

- throughput and latency under realistic candidate volumes

5. Determinism tests

- repeatability under fixed seed and config

---

## Rollback Plan

- Feature flags allow immediate disablement by capability area.
- Keep baseline candidate path available until Phase 6 complete.
- Preserve old ranking path behind fallback switch during canary.

---

## Proposed Sequence of Implementation PRs

1. PR-1: Feature flags + scenario scaffolding + reason codes
2. PR-2: Candidate taxonomy + unified priority precedence
3. PR-3: Continuity validator + As Directed explicit reconstruction
4. PR-4: Overlap insertion/reconstruction families
5. PR-5: Hard legal + home-base enforcement
6. PR-6: True cascade displacement graph and search
7. PR-7: GPU numeric acceleration + parity tests
8. PR-8: Observability hardening + legacy cleanup

---

## Definition of Done (Program Level)

- All P0 scenarios pass with feasible, policy-compliant outputs.
- Hard constraints are enforced, not advisory.
- Cascading of displaced low-priority work is operational and bounded.
- Documentation and tests reflect final target behavior.
- Legacy redundant paths removed after successful soak.
