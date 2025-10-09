from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Callable
from fastapi import APIRouter, HTTPException, Request
import sys
import traceback
import re

from .models import (
    PlanRequest, PlanCandidatesResponse,
    PlanSolveCascadeRequest, PlanSolveCascadeResponse, AssignmentOut
)

from .config import load_priority_map, load_sla_windows

from .geo import build_loc_meta_from_locations_csv, enhanced_distance_time_lookup, get_location_coordinates, check_matrix_bidirectional, get_location_coordinates, haversine_between_idx

from .candidates import weekday_from_local
from .cascade_candidates import (
    generate_cascade_candidates_with_schedules,  # Use this one
    CascadeCandidateOut
)

def create_router(
    get_data: Callable[[], Optional[Dict[str, Any]]],
    get_cost_config: Callable[[], Dict[str, float]],
) -> APIRouter:
    """
    Factory that returns the /plan router. Uses callables to fetch the
    current in-memory DATA and cost config from the backend.
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
        
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        
        # Check if user selected a specific candidate
        if req.preferred_driver_id:
            print(f"[solve_cascades] Preferred driver specified: {req.preferred_driver_id}")
            
            # Generate candidates but filter to only the preferred driver
            weekday, trip_minutes, trip_miles, all_candidates, all_schedules = generate_cascade_candidates_with_schedules(
                req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
                max_cascade_depth=req.max_cascades,
                max_candidates=20  # Generate more to find the preferred one
            )
            
            # Filter to only the preferred candidate
            candidates = [c for c in all_candidates if c.driver_id == req.preferred_driver_id]
            schedules = [s for s in all_schedules if s.get('driver_id') == req.preferred_driver_id]
            
            if not candidates:
                # Fallback: if preferred not found, use all candidates
                print(f"[solve_cascades] Warning: Preferred driver {req.preferred_driver_id} not found in candidates")
                candidates = all_candidates[:1]  # Just take first
                schedules = all_schedules[:1]
        else:
            # No preference - return all candidates
            weekday, trip_minutes, trip_miles, candidates, schedules = generate_cascade_candidates_with_schedules(
                req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
                max_cascade_depth=req.max_cascades,
                max_candidates=req.max_cascades
            )
        
        if not candidates:
            return PlanSolveCascadeResponse(
                weekday=weekday,
                trip_minutes=trip_minutes,
                trip_miles=trip_miles,
                objective_value=0.0,
                assignments=[],
                details={"backend": "cascade-cuopt", "error": "no_candidates"},
                candidates_considered=0,
                cascades=[],
                schedules=[]
            )
        
        # Build trip_id from request
        trip_id = f"NEW-{req.start_location}→{req.end_location}@{req.when_local}"
        
        # Build assignments from filtered candidates
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
        
        # Build cascade information
        cascades = []
        for candidate in candidates:
            cascades.append({
                "depth": 1,
                "displaced_by": "NEW_SERVICE",
                "driver_id": candidate.driver_id,
                "from": req.start_location,
                "to": req.end_location,
                "priority": req.priority,
                "reason": candidate.reason or "Enhanced cascade"
            })
        
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
                "preferred_driver": req.preferred_driver_id
            },
            candidates_considered=len(candidates),
            cascades=cascades,
            schedules=schedules  # Only the filtered schedule(s)
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

