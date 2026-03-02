# Sprint-Ready Backlog

Source plans:

- IMPLEMENTATION_PLAN.md
- GAP_TO_FEATURE_MATRIX.md

## Planning Assumptions

- Team capacity baseline: 20-25 story points per sprint.
- Sprint length: 2 weeks.
- Sequence prioritizes P0 functional correctness before optimization.
- Story points are relative complexity (1, 2, 3, 5, 8, 13).

---

## Epic E1: Foundation, Flags, and Scenario Harness

## Goal

Establish safe delivery guardrails and measurable acceptance scenarios.

### Story E1-S1: Add feature flags for staged enablement (5 pts)

- Add runtime flags:
  - ENABLE_AS_DIRECTED_INSERTION
  - ENABLE_PARTIAL_OVERLAP_INSERTION
  - ENABLE_NEARBY_DEPOT_SUBSTITUTION
  - ENABLE_STRICT_LEGAL_CONSTRAINTS
  - ENABLE_TRUE_CASCADE
- Acceptance:
  - Flags are injectable via env/config.
  - Flags are visible in startup logs.
  - Disabled flags preserve current baseline path.
- Dependencies: none.

### Story E1-S2: Build golden scenario test harness (8 pts)

- Create deterministic fixtures and scenarios:
  - As Directed insertion with downstream commitment.
  - Exact empty/low-priority replacement.
  - Partial overlap pre-pickup and post-drop.
  - Nearby depot substitution.
  - Home-base and legal-hours fail scenarios.
- Acceptance:
  - Scenario tests execute in CI.
  - Clear expected outputs/reason codes are asserted.
- Dependencies: E1-S1.

### Story E1-S3: Introduce structured reason codes (3 pts)

- Add standardized rule-stage reason code taxonomy for accept/reject.
- Acceptance:
  - Candidate objects include reason_code and reason_detail.
  - Test coverage for reason code emission.
- Dependencies: E1-S1.

---

## Epic E2: Candidate Taxonomy and Priority Policy

## Goal

Implement required insertion families and unified displacement precedence.

### Story E2-S1: Centralize displacement precedence policy (5 pts)

- Implement one policy service/function for displacement eligibility.
- Rules:
  - empty/deadhead displaceable
  - equal/lower priority displaceable
  - higher-priority protected
- Acceptance:
  - Policy used by all candidate families.
  - Unit tests cover boundary conditions.
- Dependencies: E1-S3.

### Story E2-S2: Add candidate family: exact replacement (3 pts)

- Normalize and harden existing exact A->B replacement path.
- Acceptance:
  - Scenario pass for exact replacement with precedence checks.
- Dependencies: E2-S1.

### Story E2-S3: Add candidate family: overlap pre-pickup (8 pts)

- Planned A->B; new C->B modeled as A->C empty + C->B loaded + rejoin.
- Acceptance:
  - Candidate generated only if timing and continuity feasible.
  - Scenario pass with expected route transform.
- Dependencies: E2-S1, E1-S2.

### Story E2-S4: Add candidate family: overlap post-drop (8 pts)

- Planned A->B; new A->C modeled as A->C loaded + C->B empty + rejoin.
- Acceptance:
  - Candidate generated only if timing and continuity feasible.
  - Scenario pass with expected route transform.
- Dependencies: E2-S1, E1-S2.

### Story E2-S5: Add candidate family: nearby-neighbor substitution (8 pts)

- Add bilateral proximity logic for pickup and drop neighborhoods.
- Acceptance:
  - Supports A~C and B~D substitution path under constraints.
  - Rejects candidates that violate continuity/constraints.
- Dependencies: E1-S2.

---

## Epic E3: Continuity and Reconstruction Engine

## Goal

Guarantee physically and temporally valid schedules after transformation.

### Story E3-S1: Build continuity validator (8 pts)

- Validate:
  - monotonic timeline
  - location continuity between elements
  - allowed waits
  - no impossible jumps
- Acceptance:
  - All returned schedules pass validator.
  - Validator failures are observable with reason codes.
- Dependencies: E1-S3.

### Story E3-S2: Explicit As Directed reconstruction path (8 pts)

- Implement direct branch for as_directed_replacement.
- Preserve downstream commitments and rejoin logic.
- Acceptance:
  - As Directed scenario tests pass.
  - No fallback-to-append for this branch.
- Dependencies: E3-S1, E2-S1.

### Story E3-S3: Reconstruction for overlap families (8 pts)

- Implement schedule reconstruction for pre-pickup/post-drop and nearby-neighbor cases.
- Acceptance:
  - Reconstructed duties are valid and explainable.
  - Scenario tests pass with expected annotation codes.
- Dependencies: E3-S1, E2-S3, E2-S4, E2-S5.

---

## Epic E4: Hard Constraints (Legal + Home Base)

## Goal

Enforce true hard feasibility requirements.

### Story E4-S1: Implement legal-hours constraint service (13 pts)

- Include:
  - driving caps
  - duty caps
  - break/rest checks
  - cross-midnight correctness
- Acceptance:
  - Any rule breach forces feasible_hard=false.
  - Legal fail scenarios pass in tests.
- Dependencies: E3-S1.

### Story E4-S2: Enforce home-base end-state rule (5 pts)

- Hard-fail candidates that do not return to home base (unless explicitly configured).
- Acceptance:
  - Home-base fail scenario passes.
- Dependencies: E3-S1.

### Story E4-S3: Integrate hard constraints into ranking gate (3 pts)

- Ensure only hard-feasible candidates can rank as primary recommendations.
- Acceptance:
  - Ranking tests confirm infeasible candidates demoted/excluded.
- Dependencies: E4-S1, E4-S2.

---

## Epic E5: True Cascade Engine

## Goal

Reassign displaced low-priority work through bounded multi-step cascade search.

### Story E5-S1: Displacement graph model (8 pts)

- Represent displaced tasks, candidate receiving duties, and edge feasibility.
- Acceptance:
  - Graph builder unit-tested with deterministic fixtures.
- Dependencies: E2-S1, E3-S1.

### Story E5-S2: Bounded cascade search (13 pts)

- Add depth-limited, branch-limited search with pruning.
- Acceptance:
  - Multi-step cascades generated where feasible.
  - Runtime bounded by configuration.
- Dependencies: E5-S1.

### Story E5-S3: Cascade objective aggregation and output model (8 pts)

- Compute total system objective and chain provenance.
- Acceptance:
  - API outputs include chain details, displaced/reassigned trace.
- Dependencies: E5-S2.

### Story E5-S4: Diagnostics contract for solve endpoints (5 pts)

- Add structured diagnostics payloads for:
  - `/plan/solve_cascades` (`details.cascade_diagnostics`)
  - `/plan/solve_multi` (`meta.cascade_diagnostics` and per-solution diagnostics)
- Include unresolved-work aggregates and reason-code distribution.
- Add contract tests that verify diagnostics shape for both reassigned and fallback paths.
- Acceptance:
  - Diagnostics fields are present and stable on all solve responses.
  - Integration tests assert diagnostics contract shape.
- Dependencies: E5-S3.

---

## Epic E6: CUDA/GPU Acceleration and Performance

## Goal

Accelerate numeric-heavy planning stages safely.

### Story E6-S1: Stage-level performance profiling baseline (5 pts)

- Measure latency by stage (filtering, expansion, scoring, reconstruction, solve).
- Acceptance:
  - Baseline report committed and reproducible.
- Dependencies: E2-S5, E3-S3, E4-S3.

### Story E6-S2: GPU batched numeric feasibility path (8 pts)

- Accelerate matrix reachability/slack calculations with adaptive thresholds.
- Acceptance:
  - Measurable p95 reduction for high-volume workloads.
  - CPU parity tests pass.
- Dependencies: E6-S1.

### Story E6-S3: cuOpt batching and queue strategy tuning (8 pts)

- Optimize small/large job batching and queue behavior.
- Acceptance:
  - Throughput improved without correctness drift.
- Dependencies: E6-S1.

### Story E6-S4: CPU/GPU parity and determinism validation (5 pts)

- Compare outputs under fixed seeds/configs.
- Acceptance:
  - Policy-feasibility parity pass criteria met.
- Dependencies: E6-S2, E6-S3.

---

## Epic E7: Rollout, Observability, and Cleanup

## Goal

Deploy safely and remove obsolete paths after stability.

### Story E7-S1: Add production observability metrics (5 pts)

- Add counters/distributions:
  - reject reason codes
  - candidate families
  - cascade depth
  - solve latency by stage
- Acceptance:
  - Dashboard-ready metrics exposed.
- Dependencies: E1-S3.

### Story E7-S2: Canary rollout and acceptance checklist (3 pts)

- Define phased rollout with rollback triggers.
- Acceptance:
  - Runbook approved and executed in canary.
- Dependencies: E5-S3, E6-S4, E7-S1.

### Story E7-S3: Remove redundant legacy paths (5 pts)

- Delete deprecated candidate/reconstruction branches and stale compatibility logic.
- Acceptance:
  - No dead code remains for replaced paths.
  - Test suite and docs updated.
- Dependencies: E7-S2.

---

## Suggested Sprint Sequencing

## Sprint 1 (20-24 pts)

- E1-S1 (5)
- E1-S2 (8)
- E1-S3 (3)
- E2-S1 (5)
- E2-S2 (3)

## Sprint 2 (24-29 pts)

- E2-S3 (8)
- E2-S4 (8)
- E2-S5 (8)

## Sprint 3 (24 pts)

- E3-S1 (8)
- E3-S2 (8)
- E3-S3 (8)

## Sprint 4 (21 pts)

- E4-S1 (13)
- E4-S2 (5)
- E4-S3 (3)

## Sprint 5 (29 pts)

- E5-S1 (8)
- E5-S2 (13)
- E5-S3 (8)

## Sprint 5A (5 pts, completed increment)

- E5-S4 (5)

## Sprint 6 (26 pts)

- E6-S1 (5)
- E6-S2 (8)
- E6-S3 (8)
- E6-S4 (5)

## Sprint 7 (13 pts)

- E7-S1 (5)
- E7-S2 (3)
- E7-S3 (5)

---

## Program-Level Exit Criteria

- All P0 scenario tests pass.
- No hard-constraint violations in feasible_hard outputs.
- Cascading works with bounded depth and full traceability.
- Performance targets met for representative workloads.
- Deprecated redundant paths removed and docs fully aligned.
