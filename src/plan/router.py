from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Callable
from fastapi import APIRouter, HTTPException, Request
import sys
import traceback
import re
import time

from .models import (
    PlanRequest, PlanCandidatesResponse,
    PlanSolveCascadeRequest, PlanSolveCascadeResponse, AssignmentOut,
    PlanSolveMultiRequest, PlanSolveMultiResponse, PlanSolutionOut
)

from .config import load_priority_map, load_sla_windows

from .geo import build_loc_meta_from_locations_csv, enhanced_distance_time_lookup, get_location_coordinates, check_matrix_bidirectional, haversine_between_idx

from .candidates import weekday_from_local
from .cascade_candidates import (
    generate_cascade_candidates_with_schedules,  # Use this one
    CascadeCandidateOut
)


def _parse_reason_detail(detail: Optional[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not detail:
        return parsed
    for part in str(detail).split(";"):
        token = part.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed[key] = value.lower() == "true"
            continue
        try:
            parsed[key] = int(value)
            continue
        except Exception:
            pass
        try:
            parsed[key] = float(value)
            continue
        except Exception:
            pass
        parsed[key] = value
    return parsed


def _candidate_cascade_diag(candidate: Any) -> Dict[str, Any]:
    reason_diag = _parse_reason_detail(getattr(candidate, "reason_detail", None))
    return {
        "chain_depth": int(reason_diag.get("chain_depth", 0) or 0),
        "assigned_steps": int(reason_diag.get("assigned_steps", 0) or 0),
        "blocked_steps": int(reason_diag.get("blocked_steps", 0) or 0),
        "uncovered_p4": int(reason_diag.get("uncovered_p4", 0) or 0),
        "disposed_p5": int(reason_diag.get("disposed_p5", 0) or 0),
        "feasible_hard": bool(getattr(candidate, "feasible_hard", False)),
        "reason_code": getattr(candidate, "reason_code", None),
    }


def _summarize_cascade_candidates(candidates: List[Any]) -> Dict[str, Any]:
    if not candidates:
        return {
            "candidates_total": 0,
            "feasible_hard_count": 0,
            "max_chain_depth": 0,
            "avg_chain_depth": 0.0,
            "unresolved_total": 0,
            "uncovered_p4_total": 0,
            "disposed_p5_total": 0,
            "reason_code_counts": {},
        }

    diags = [_candidate_cascade_diag(c) for c in candidates]
    reason_code_counts: Dict[str, int] = {}
    for d in diags:
        rc = str(d.get("reason_code") or "UNKNOWN")
        reason_code_counts[rc] = reason_code_counts.get(rc, 0) + 1

    chain_depths = [int(d["chain_depth"]) for d in diags]
    uncovered_p4_total = int(sum(int(d["uncovered_p4"]) for d in diags))
    disposed_p5_total = int(sum(int(d["disposed_p5"]) for d in diags))

    return {
        "candidates_total": len(candidates),
        "feasible_hard_count": int(sum(1 for d in diags if d["feasible_hard"])),
        "max_chain_depth": max(chain_depths) if chain_depths else 0,
        "avg_chain_depth": float(sum(chain_depths) / len(chain_depths)) if chain_depths else 0.0,
        "unresolved_total": int(uncovered_p4_total + disposed_p5_total),
        "uncovered_p4_total": uncovered_p4_total,
        "disposed_p5_total": disposed_p5_total,
        "reason_code_counts": reason_code_counts,
    }


def _elapsed_ms(start_ts: float, end_ts: Optional[float] = None) -> float:
    end = time.perf_counter() if end_ts is None else end_ts
    return round((end - start_ts) * 1000.0, 3)

def create_router(
    get_data: Callable[[], Optional[Dict[str, Any]]],
    get_cost_config: Callable[[], Dict[str, float]],
    get_cuopt_url: Optional[Callable[[], str]] = None,
) -> APIRouter:
    """
    Factory that returns the /plan router. Uses callables to fetch the
    current in-memory DATA and cost config from the backend.
    `get_cuopt_url` is accepted for backward compatibility with older callers.
    """
    router = APIRouter(prefix="/plan", tags=["Plan"])

    PRIORITY_MAP: Dict[str, int] = load_priority_map()
    SLA_WINDOWS: Dict[int, Dict[str, int]] = load_sla_windows()

    # ------------------- Shared Helpers -------------------

    def ensure_ready():
        DATA = get_data()
        if DATA is None:
            raise HTTPException(
                status_code=503,
                detail="Private data not loaded. Upload/build and POST /admin/reload.",
            )
        M = {
            "dist": DATA["distance"],
            "time": DATA["time"],
            "loc2idx": DATA["location_to_index"],
        }
        loc_meta: Dict[str, Dict[str, Any]] = {}
        try:
            if DATA.get("locations_df") is not None:
                loc_meta = build_loc_meta_from_locations_csv(DATA["locations_df"])
        except Exception:
            pass
        return DATA, M, loc_meta

    def resolve_trip_metrics(req: PlanRequest, M: Dict[str, Any], loc_meta: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
        miles, minutes, _ = enhanced_distance_time_lookup(
            req.start_location,
            req.end_location,
            M,
            loc_meta,
        )
        trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else float(minutes)
        trip_miles = float(req.trip_miles) if req.trip_miles is not None else float(miles)
        return trip_minutes, trip_miles

    def build_outsourcing_assignment(trip_id: str, trip_miles: float, cfg: Dict[str, float]) -> AssignmentOut:
        outsourcing_base = float(cfg.get("outsourcing_base_cost", 200.0))
        outsourcing_miles = float(cfg.get("outsourcing_per_mile", 2.0)) * float(trip_miles)
        total_cost = outsourcing_base + outsourcing_miles
        return AssignmentOut(
            trip_id=trip_id,
            type="outsourced",
            cost=total_cost,
            miles_delta=float(trip_miles),
            cost_breakdown={
                "outsourcing_base": outsourcing_base,
                "outsourcing_miles": outsourcing_miles,
            },
        )


# ------------------- Endpoints -------------------

    @router.post("/candidates", response_model=PlanCandidatesResponse)
    def plan_candidates(req: PlanRequest):
        """
        UPDATED: Now returns schedule data for UI visualization
        """
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        
        # Use enhanced version that returns schedules
        weekday, trip_minutes, trip_miles, cands, schedules = generate_cascade_candidates_with_schedules(
            req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
            max_cascade_depth=2, 
            max_candidates=10
        )
        
        # IMPORTANT: PlanCandidatesResponse needs to be updated to include schedules
        # See Fix 2 below for the model update
        return PlanCandidatesResponse(
            weekday=weekday, 
            trip_minutes=trip_minutes, 
            trip_miles=trip_miles, 
            candidates=cands,
            schedules=schedules  # NEW: Add schedule data
        )

    @router.post("/solve_cascades", response_model=PlanSolveCascadeResponse)
    def plan_and_solve_cascades_enhanced(req: PlanSolveCascadeRequest, request: Request):
        """Enhanced cascade solving with proper UI schedule output"""
        t_total = time.perf_counter()

        t_ready = time.perf_counter()
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        ready_ms = _elapsed_ms(t_ready)

        generation_ms = 0.0
        assignment_build_ms = 0.0
        cascades_build_ms = 0.0
        postprocess_ms = 0.0
        
        # Check if user selected a specific candidate
        if req.preferred_driver_id:
            print(f"[solve_cascades] Preferred driver specified: {req.preferred_driver_id}")
            
            # Generate candidates but filter to only the preferred driver
            t_generation = time.perf_counter()
            weekday, trip_minutes, trip_miles, all_candidates, all_schedules = generate_cascade_candidates_with_schedules(
                req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
                max_cascade_depth=req.max_cascades,
                max_candidates=20,  # Generate more to find the preferred one
                preferred_driver_id=req.preferred_driver_id,
            )
            generation_ms += _elapsed_ms(t_generation)
            
            # Filter to only the preferred candidate
            t_post = time.perf_counter()
            candidates = [c for c in all_candidates if c.driver_id == req.preferred_driver_id]
            preferred_candidate_id = req.preferred_candidate_id
            if preferred_candidate_id:
                schedules = [
                    s for s in all_schedules
                    if str(s.get('candidate_id', '')).strip() == str(preferred_candidate_id).strip()
                ]
            else:
                schedules = [s for s in all_schedules if s.get('driver_id') == req.preferred_driver_id]
            
            if not candidates:
                # Fallback: if preferred not found, use all candidates
                print(f"[solve_cascades] Warning: Preferred driver {req.preferred_driver_id} not found in candidates")
                candidates = all_candidates[:1]  # Just take first
                schedules = all_schedules[:1]
            postprocess_ms += _elapsed_ms(t_post)
        else:
            # No preference - return all candidates
            t_generation = time.perf_counter()
            weekday, trip_minutes, trip_miles, candidates, schedules = generate_cascade_candidates_with_schedules(
                req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
                max_cascade_depth=req.max_cascades,
                max_candidates=req.max_cascades
            )
            generation_ms += _elapsed_ms(t_generation)
        
        if not candidates:
            t_fallback = time.perf_counter()
            trip_minutes, trip_miles = resolve_trip_metrics(req, M, LOC_META)
            trip_id = f"NEW-{req.start_location}→{req.end_location}@{req.when_local}"
            outsourced = build_outsourcing_assignment(trip_id, trip_miles, cfg)
            performance = {
                "total_ms": _elapsed_ms(t_total),
                "ensure_ready_ms": ready_ms,
                "candidate_generation_ms": generation_ms,
                "assignment_build_ms": 0.0,
                "cascade_build_ms": 0.0,
                "postprocess_ms": postprocess_ms + _elapsed_ms(t_fallback),
            }
            return PlanSolveCascadeResponse(
                weekday=weekday,
                trip_minutes=trip_minutes,
                trip_miles=trip_miles,
                objective_value=outsourced.cost,
                assignments=[outsourced],
                details={
                    "backend": "cascade-cuopt",
                    "error": "no_candidates",
                    "fallback": "outsourced",
                    "cascade_diagnostics": _summarize_cascade_candidates([]),
                    "performance": performance,
                },
                candidates_considered=0,
                cascades=[],
                schedules=[]
            )
        
        # Build trip_id from request
        trip_id = f"NEW-{req.start_location}→{req.end_location}@{req.when_local}"
        
        # Build assignments from filtered candidates
        t_assign = time.perf_counter()
        assignments = []
        total_cost = 0.0
        
        for candidate in candidates:
            assignment = AssignmentOut(
                trip_id=trip_id,
                type="reassigned",
                driver_id=candidate.driver_id,
                candidate_id=candidate.candidate_id,
                cost=candidate.est_cost,
                deadhead_miles=candidate.deadhead_miles,
                overtime_minutes=candidate.overtime_minutes,
                delay_minutes=candidate.delay_minutes,
                miles_delta=candidate.miles_delta,
                uses_emergency_rest=candidate.uses_emergency_rest
            )
            assignments.append(assignment)
            total_cost += candidate.est_cost
        assignment_build_ms += _elapsed_ms(t_assign)
        
        # Build cascade information
        t_cascades = time.perf_counter()
        cascades = []
        for candidate in candidates:
            cand_diag = _candidate_cascade_diag(candidate)
            cascades.append({
                "depth": cand_diag["chain_depth"],
                "displaced_by": "NEW_SERVICE",
                "driver_id": candidate.driver_id,
                "from": req.start_location,
                "to": req.end_location,
                "priority": req.priority,
                "reason": candidate.reason or "Enhanced cascade",
                "reason_code": candidate.reason_code,
                "reason_detail": candidate.reason_detail,
                "assigned_steps": cand_diag["assigned_steps"],
                "blocked_steps": cand_diag["blocked_steps"],
            })
        cascades_build_ms += _elapsed_ms(t_cascades)

        diagnostics = _summarize_cascade_candidates(candidates)
        performance = {
            "total_ms": _elapsed_ms(t_total),
            "ensure_ready_ms": ready_ms,
            "candidate_generation_ms": generation_ms,
            "assignment_build_ms": assignment_build_ms,
            "cascade_build_ms": cascades_build_ms,
            "postprocess_ms": postprocess_ms,
        }
        
        print(f"[solve_cascades] Returning {len(candidates)} candidates with {len(schedules)} schedules")
        
        return PlanSolveCascadeResponse(
            weekday=weekday,
            trip_minutes=trip_minutes,
            trip_miles=trip_miles,
            objective_value=total_cost,
            assignments=assignments,
            details={
                "backend": "cascade-cuopt-enhanced",
                "max_cascades": req.max_cascades,
                "drivers_touched": len(set(a.driver_id for a in assignments)),
                "preferred_driver": req.preferred_driver_id,
                "cascade_diagnostics": diagnostics,
                "performance": performance,
            },
            candidates_considered=len(candidates),
            cascades=cascades,
            schedules=schedules  # Only the filtered schedule(s)
        )

    @router.post("/solve_multi", response_model=PlanSolveMultiResponse)
    def plan_solve_multi(req: PlanSolveMultiRequest):
        t_total = time.perf_counter()

        t_ready = time.perf_counter()
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        ready_ms = _elapsed_ms(t_ready)

        cascade_req = PlanSolveCascadeRequest(
            start_location=req.start_location,
            end_location=req.end_location,
            mode=req.mode,
            when_local=req.when_local,
            priority=req.priority,
            trip_minutes=req.trip_minutes,
            trip_miles=req.trip_miles,
            top_n=max(1, req.top_n_per_step),
            max_cascades=req.max_cascades,
            max_drivers_affected=req.max_drivers_affected,
        )

        t_generation = time.perf_counter()
        weekday, trip_minutes, trip_miles, candidates, schedules = generate_cascade_candidates_with_schedules(
            cascade_req,
            DATA,
            M,
            cfg,
            LOC_META,
            SLA_WINDOWS,
            max_cascade_depth=req.max_cascades,
            max_candidates=max(1, req.max_solutions),
        )
        generation_ms = _elapsed_ms(t_generation)

        trip_minutes, trip_miles = resolve_trip_metrics(cascade_req, M, LOC_META)
        trip_id = f"NEW-{req.start_location}→{req.end_location}@{req.when_local}"

        solutions: List[PlanSolutionOut] = []

        if not candidates:
            t_build = time.perf_counter()
            outsourced = build_outsourcing_assignment(trip_id, trip_miles, cfg)
            solutions = [
                PlanSolutionOut(
                    rank=1,
                    objective_value=outsourced.cost,
                    drivers_touched=0,
                    assignments=[outsourced],
                    cascades=[],
                    schedules=[],
                    details={"backend": "cascade-cuopt", "fallback": "outsourced", "error": "no_candidates"},
                )
            ]
            build_ms = _elapsed_ms(t_build)
        else:
            t_build = time.perf_counter()
            for candidate in candidates[: max(1, req.max_solutions)]:
                assignment = AssignmentOut(
                    trip_id=trip_id,
                    type="reassigned",
                    driver_id=candidate.driver_id,
                    candidate_id=candidate.candidate_id,
                    cost=candidate.est_cost,
                    deadhead_miles=candidate.deadhead_miles,
                    overtime_minutes=candidate.overtime_minutes,
                    delay_minutes=candidate.delay_minutes,
                    miles_delta=candidate.miles_delta,
                    uses_emergency_rest=candidate.uses_emergency_rest,
                )
                candidate_schedules = [s for s in schedules if s.get("driver_id") == candidate.driver_id]
                solutions.append(
                    PlanSolutionOut(
                        rank=0,
                        objective_value=candidate.est_cost,
                        drivers_touched=1,
                        assignments=[assignment],
                        cascades=[
                            {
                                "depth": _candidate_cascade_diag(candidate)["chain_depth"],
                                "displaced_by": "NEW_SERVICE",
                                "driver_id": candidate.driver_id,
                                "from": req.start_location,
                                "to": req.end_location,
                                "priority": req.priority,
                                "reason": candidate.reason or "Enhanced cascade",
                                "reason_code": candidate.reason_code,
                                "reason_detail": candidate.reason_detail,
                            }
                        ],
                        schedules=candidate_schedules,
                        details={
                            "backend": "cascade-cuopt-enhanced",
                            "candidate_id": candidate.candidate_id,
                            "cascade_diagnostics": _summarize_cascade_candidates([candidate]),
                        },
                    )
                )

            solutions.sort(key=lambda s: s.objective_value)
            for idx, solution in enumerate(solutions, start=1):
                solution.rank = idx
            build_ms = _elapsed_ms(t_build)

        performance = {
            "total_ms": _elapsed_ms(t_total),
            "ensure_ready_ms": ready_ms,
            "candidate_generation_ms": generation_ms,
            "solution_build_ms": build_ms,
        }

        return PlanSolveMultiResponse(
            weekday=weekday,
            trip_minutes=trip_minutes,
            trip_miles=trip_miles,
            solutions=solutions,
            meta={
                "backend": "cascade-cuopt-enhanced",
                "solutions_returned": len(solutions),
                "cascade_diagnostics": _summarize_cascade_candidates(candidates),
                "performance": performance,
            },
        )

    @router.get("/priority_map")
    def get_priority_map():
        return PRIORITY_MAP

    @router.get("/locations")
    def list_locations():
        try:
            _DATA, M, _ = ensure_ready()
            df = _DATA.get("locations_df")
            if df is not None:
                df = df.dropna(subset=["Mapped Name A"]).copy()
                df["name"] = df["Mapped Name A"].astype(str).str.strip()
                df["postcode"] = df.get("Mapped Postcode A", None)
                locs = df[["name", "postcode"]].dropna().drop_duplicates().to_dict(orient="records")
                return {"locations": locs, "count": len(locs), "source": "memory"}
        except HTTPException:
            pass
        return {"locations": [], "count": 0, "source": "none"}

    return router

