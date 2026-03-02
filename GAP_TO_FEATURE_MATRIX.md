# Gap-to-Feature Matrix and CUDA Guidance

## Scope

This document maps required operational behavior to current implementation gaps and defines concrete feature targets. It also specifies how CUDA should be used for this planning use case and where CUDA has technical limits.

---

## Gap-to-Feature Matrix

| Feature Area                           | Required Behavior                                                                                                    | Current Behavior                                                                                                                                        | Gap    | Proposed Feature                                                                                          | Primary Files/Modules                                                                         | Priority | Acceptance Criteria                                                                                                |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------ |
| As Directed utilization                | Use As Directed blocks as real spare-time capacity when location/time fit and downstream commitments remain feasible | As Directed is partially detected; replacement strategy exists but reconstruction path does not explicitly implement as_directed_replacement end-to-end | High   | Add explicit As Directed insertion/replacement planner path with continuity checks before and after block | src/plan/candidates.py, src/plan/cascade_candidates.py                                        | P0       | Candidate appears for valid As Directed scenarios; reconstructed duty preserves chronology and next committed legs |
| Exact empty-to-loaded substitution     | If planned empty/low-priority A to B exists, replace with higher-priority new A to B service if timing works         | Exact empty leg replacement exists; priority handling is pattern-specific                                                                               | Medium | Unify displacement rule for empty and low-priority loads under one precedence model                       | src/plan/candidates.py, src/plan/cascade_candidates.py                                        | P0       | For A to B match, candidate generated only when precedence rule passes and time feasibility holds                  |
| Partial overlap insertion (pre-pickup) | Planned A to B, new C to B: allow A to C empty + C to B loaded + rejoin                                              | Not modeled as a first-class pattern                                                                                                                    | High   | Add pre-pickup overlap candidate family with continuity and legal checks                                  | src/plan/candidates.py, src/plan/cascade_candidates.py                                        | P0       | Valid C to B inserts returned when route/time feasible; no invalid duty breaks                                     |
| Partial overlap insertion (post-drop)  | Planned A to B, new A to C: allow A to C loaded + C to B empty + rejoin                                              | Not modeled as a first-class pattern                                                                                                                    | High   | Add post-drop overlap candidate family with continuity and legal checks                                   | src/plan/candidates.py, src/plan/cascade_candidates.py                                        | P0       | Valid A to C inserts returned when route/time feasible; downstream leg to B still achieved                         |
| Nearby-depot substitution              | Allow substitutions when A near C and B near D even if no direct calls at A/B                                        | Proximity checks exist but mostly around start-side filtering; no bilateral pickup/drop neighborhood substitution logic                                 | High   | Add bilateral neighborhood matching (pickup radius and drop radius) with route transformation rules       | src/plan/candidates.py, src/plan/geo.py, src/plan/cascade_candidates.py                       | P0       | Candidates returned for near-neighbor cases only when end-to-end feasibility passes                                |
| Duty continuity                        | Ensure modified duty remains physically and temporally consistent at every step                                      | Current templates do not comprehensively validate every handoff between consecutive elements                                                            | High   | Introduce continuity validator for full before/after element chain                                        | src/plan/cascade_candidates.py                                                                | P0       | Every returned schedule passes continuity validator (location continuity + monotonic time + allowed waits)         |
| Hard legal constraints                 | Enforce legal driving and duty constraints robustly, not only coarse max-duty approximations                         | Coarse max_duty_minutes style logic; not full legal regime modeling                                                                                     | High   | Add explicit legal constraint engine (driving minutes, duty minutes, breaks/rest windows, shift caps)     | src/plan/candidates.py, src/plan/cascade_candidates.py, config/settings                       | P0       | Candidates violating legal rules are never marked feasible_hard                                                    |
| Home-base return constraint            | Driver must end at home base after modified duty unless explicitly allowed otherwise                                 | Return-to-base is sometimes reconstructed but not guaranteed across all insertion families                                                              | High   | Add hard end-state constraint and fallback handling for non-return cases                                  | src/plan/cascade_candidates.py, src/plan/candidates.py                                        | P0       | All feasible_hard candidates satisfy end-at-home-base rule                                                         |
| Priority displacement policy           | Higher-priority new load may displace only equal/lower-priority planned work; protected work must not be displaced   | Priority checks exist in selected branches only                                                                                                         | High   | Centralize precedence policy and apply consistently to all candidate families                             | src/plan/candidates.py, src/plan/cascade_candidates.py                                        | P0       | Uniform policy applied regardless of candidate type                                                                |
| True cascading of displaced work       | Displaced low-priority jobs should be reassigned recursively/cascaded to other duties when possible                  | Current cascade path is largely single-driver optimization with limited displaced-work propagation                                                      | High   | Implement multi-driver cascade engine with bounded depth, branch factor, and objective tracking           | src/plan/cascade_candidates.py, src/plan/router.py                                            | P0       | Output includes valid multi-step cascades with displaced task reassignment records                                 |
| Candidate ranking quality              | Rank by hard feasibility, policy compliance, downstream impact, and true net cost/time                               | Current ranking is template-tier + estimated cost                                                                                                       | Medium | Multi-criteria ranking with policy penalties and continuity/legal confidence                              | src/plan/candidates.py, src/plan/models.py                                                    | P1       | Ranking aligns with defined objective and policy priorities                                                        |
| Data contract for priorities           | Priority mapping should be deterministic and aligned with load types/business rules                                  | Priority data source differs across generation paths                                                                                                    | Medium | Standardize single priority derivation path and validate at pipeline time                                 | src/driver_states_builder.py, scripts/build_dataset_from_rsl.py, scripts/run_data_pipeline.py | P1       | Identical input rows produce identical priorities across all pipeline paths                                        |
| Explainability                         | Provide clear reason why a candidate was generated/rejected                                                          | Reasons exist but are inconsistent and branch-specific                                                                                                  | Medium | Add structured reason codes for accept/reject by rule stage                                               | src/plan/candidates.py, src/plan/cascade_candidates.py                                        | P2       | API can report rule-stage outcomes for diagnostics and tuning                                                      |

---

## Implementation Status Snapshot (2026-02-25)

Status legend:

- Closed: implemented and covered by tests.
- Partial: implemented in core path but not yet complete for all required variants.
- Open: not yet implemented to required level.

| Feature Area                                     | Status  | Notes                                                                                                                                                                  |
| ------------------------------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| As Directed utilization                          | Closed  | Explicit reconstruction branch with continuity validation is implemented.                                                                                              |
| Exact empty-to-loaded substitution               | Closed  | Candidate + reconstruction paths implemented with displacement handling.                                                                                               |
| Partial overlap insertion (pre-pickup/post-drop) | Partial | Candidate generation implemented; full cascade reconstruction semantics for overlap families still limited.                                                            |
| Nearby-depot substitution                        | Partial | Bilateral candidate matching implemented; deeper reconstruction/cascade behavior for nearby substitutions still limited.                                               |
| Duty continuity                                  | Closed  | Continuity validator is active and tested for schedule reconstruction paths.                                                                                           |
| Hard legal constraints                           | Partial | Strict mode includes daily/continuous driving and Meal Relief break handling; full UK regime (weekly/two-week, rest compensation, WTD/night specifics) remains.        |
| Home-base return constraint                      | Partial | Strict enforcement implemented for candidate/cascade outputs, including secondary assignments; configurable exception policy and all edge paths are not fully modeled. |
| Priority displacement policy                     | Partial | Centralized in several major branches; still not fully unified across every insertion/reconstruction/cascade edge case.                                                |
| True cascading of displaced work                 | Partial | Bounded multi-driver propagation implemented (depth/driver limits, unresolved tracking), but objective-optimized graph search and full family coverage remain.         |
| Candidate ranking quality                        | Open    | Ranking is still mostly tier + estimated cost; multi-criteria policy/impact confidence scoring remains.                                                                |
| Data contract for priorities                     | Partial | Canonical schema path improved; full end-to-end deterministic priority derivation parity across all dataset paths remains to finalize.                                 |
| Explainability                                   | Closed  | Structured reason codes/details and API diagnostics contract are implemented and integration-tested.                                                                   |

### Remaining Gap Work (Highest Priority)

1. Complete legal constraint engine to full UK operational coverage (weekly/two-week caps, reduced/compensated rests, and working-time overlays).
2. Extend true-cascade displacement/reconstruction to all insertion families (overlap + nearby), not only exact replacement-driven displacement.
3. Upgrade ranking to multi-criteria scoring (hard-feasibility confidence, policy penalties, unresolved-work penalties, downstream impact).
4. Finalize priority derivation consistency checks across all pipeline entry points and add parity assertions.
5. Add performance/observability deliverables for cascade depth distributions, unresolved-work rates, and stage-level latency baselines.

---

## CUDA Strategy for This Use Case

## Where CUDA Should Be Used

1. Batch feasibility precomputation

- Run large batched matrix lookups and time-window reachability checks on GPU for many duties and candidate insertions.
- Best fit: dense operations over large arrays of times/distances.

2. Candidate expansion scoring at scale

- Evaluate many candidate families (exact, pre-pickup, post-drop, nearby-neighbor) in parallel where calculations are vectorizable.
- Best fit: arithmetic-heavy cost and slack computations.

3. Optimization stage (cuOpt)

- Keep cuOpt as the core solver for constrained routing subproblems that are naturally represented as vehicle-routing/assignment optimization.
- Use GPU for solving many candidate subproblems in batches when model size justifies it.

4. Neighborhood search acceleration

- GPU-accelerate nearest-neighbor and proximity filtering for pickup/drop endpoint neighborhoods.
- Best fit: distance threshold filtering and top-k neighbor retrieval over large location sets.

## Where CUDA Should Not Be Primary

1. Branch-heavy business-rule logic

- Complex if-else policy checks (priority precedence, protected loads, exception handling) are often CPU-friendlier due to divergence.

2. Small-problem orchestration

- For small candidate counts, kernel launch and host-device transfer overhead can dominate and make GPU slower than CPU.

3. Fine-grained schedule reconstruction

- Rebuilding one duty timeline with many conditional edits is often better kept on CPU unless heavily batched.

---

## Recommended Hybrid Architecture

1. CPU policy layer

- Build candidate families, apply policy guards, and orchestrate cascade tree search.

2. GPU numeric layer

- Batch distance/time/slack computations and neighborhood matching.

3. GPU optimization layer (cuOpt)

- Solve constrained subproblems for shortlisted candidates.

4. CPU validation layer

- Enforce hard legal rules and continuity validation before finalizing feasible_hard outputs.

This keeps policy determinism and explainability on CPU while using GPU where parallel numeric throughput is highest.

---

## CUDA Limitations in This Use Case

1. Transfer overhead

- Repeated host-device movement for small batches can erase gains.

2. Warp divergence

- Highly conditional route-policy logic causes poor GPU utilization.

3. VRAM limits

- Large matrices plus multiple concurrent candidate buffers can pressure GPU memory.

4. Determinism and reproducibility

- Parallel floating-point behavior and solver heuristics may produce slight run-to-run differences unless carefully controlled.

5. Debug complexity

- Hybrid CPU/GPU pipelines are harder to debug and profile than CPU-only logic.

6. Constraint expressiveness limits

- Some bespoke labor-rule constraints are awkward to encode directly in routing solvers and may require CPU-side post-validation.

7. Concurrency bottlenecks

- Running too many small cuOpt jobs concurrently can cause queueing/overhead; batching strategy matters.

8. Platform/ops dependencies

- CUDA stack health (driver, toolkit/container compatibility) is an operational dependency that can impact uptime.

---

## Implementation Guidance for CUDA Adoption

1. Start with profiling

- Measure CPU baseline by stage: filtering, candidate expansion, scoring, reconstruction, solve.

2. GPU-enable only hot numeric paths first

- Prioritize matrix-based reachability and batch cost/slack scoring.

3. Keep policy and legal validation deterministic on CPU

- Avoid embedding policy branching directly in kernels initially.

4. Introduce adaptive execution

- Route to CPU for small batch sizes and GPU for large batches using threshold heuristics.

5. Add observability

- Capture per-stage latency, GPU utilization, queue depth, and candidate acceptance/rejection counts.

6. Protect correctness with parity tests

- CPU vs GPU parity tests on the same scenarios before enabling production-by-default GPU paths.

---

## Suggested Next Deliverables

1. Formal candidate taxonomy spec

- Exact replacement, pre-pickup overlap, post-drop overlap, nearby-neighbor substitution, and cascade handoff semantics.

2. Hard-constraint specification

- Explicit legal constraints and home-base return rules with pass/fail definitions.

3. Test pack for real-world scenarios

- Add golden scenarios for As Directed usage, nearby depots, and multi-step cascades with priority displacement.

4. Performance benchmark plan

- Stage-level benchmark suite to justify GPU path activation thresholds.
