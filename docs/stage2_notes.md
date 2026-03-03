# Feasibility Study DRAFT: Create Spare Drivers at VOC + Time

## 1) Request Summary
- **Goal:** Create a target number of spare drivers at a selected VOC, day/time.
- **Inputs (proposed):**
  - `voc_location`
  - `when_local` (e.g., Thursday 21:00)
  - `target_available_drivers` (e.g., 5)
  - Per-priority delay/cancel policy sliders
- **Business success criteria:** UI can generate requested number of spare drivers by allocating existing resource within user-defined constraints

## 2) Current System Snapshot
- Current code paths focus on inserting **one new trip** into existing duties, then scoring candidates/cascades.
- Existing API request model is centered on `start_location -> end_location` plus timing mode and priority.
- Current cascade/cuOpt flow evaluates mostly single-driver candidates and returns schedule deltas.

## 3) CuOpt Doc Check - TBC

## 4) Feasibility Verdict
**Provisional:** Feasible, but as a **new optimization mode**, not a small tweak to current single-trip insertion.

## 5) What Already Helps
- Weekday activation and duty-window filtering exist.
- Priority exists on duty elements.
- Settings include cost/constraint structures that can be extended.
- UI already supports before/after schedule visualization patterns.

## 6) Gaps to Close
1. **Objective mismatch**
   - Current: minimize impact of adding one trip.
   - Needed: maximize released drivers at VOC/time under disruption constraints.
2. **Solver modeling depth**
   - Current payload in cascade path is minimal (small matrix/single vehicle-task style).
   - Needed: multi-duty/multi-driver redistribution model.
3. **Policy controls**
   - Needed: explicit per-priority late tolerance and cancellation rules mapped to optimization penalties/constraints.
4. **Legal compliance**
   - Needed: robust hours/break constraints and post-solve validation.
5. **Critical-duty protection**
   - Needed: skip duties with mostly/high P1 content.

## 7) Proposed Delivery Plan
### Phase 1: Policy-aware heuristic prototype
- New endpoint for availability request
- Candidate duty selection around VOC/time ±X
- Critical-duty skip logic
- Ranked options + disruption accounting

### Phase 2: cuOpt-backed re-optimization
- Multi-duty/multi-driver payload
- Priority-driven delay/cancel policy encoding
- Objective: meet target spare drivers with minimum weighted disruption

### Phase 3: Operational hardening
- Legal compliance validator
- Explainability output (what moved, delayed, canceled)
- Scenario regression tests

## 8) Risks
- Solver-model complexity and stability
- Data quality (duty windows, home base mapping, priority consistency)
- Operational acceptability of delays/cancellations
