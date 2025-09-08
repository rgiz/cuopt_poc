from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Callable
from fastapi import APIRouter, HTTPException, Request

from .models import (
    PlanRequest, PlanCandidatesResponse, CandidateOut,
    PlanSolveCascadeRequest, PlanSolveCascadeResponse, AssignmentOut,
    PlanSolveMultiRequest, PlanSolveMultiResponse, PlanSolutionOut, DriverScheduleOut
)
from .cuopt_adapter import build_cuopt_payload, solve_with_cuopt, extract_solutions_from_cuopt
from .config import load_priority_map, load_sla_windows
from .candidates import generate_candidates, weekday_from_local
from .geo import build_loc_meta_from_locations_csv


def create_router(
    get_data: Callable[[], Optional[Dict[str, Any]]],
    get_cost_config: Callable[[], Dict[str, float]],
    get_cuopt_url: Callable[[], str],
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

    def _find_leg_by_candidate_id(meta: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
        if "::swap_leg@" not in candidate_id:
            return None
        try:
            start_min = int(candidate_id.split("@", 1)[1])
        except Exception:
            return None
        for e in meta.get("elements", []):
            if e.get("is_travel") and int(e.get("start_min", -1)) == start_min:
                return e
        return None

    def _build_trip_from_leg(e: Dict[str, Any], M: Dict[str, Any]) -> Dict[str, Any]:
        start_loc = str(e.get("from", "")).upper()
        end_loc = str(e.get("to", "")).upper()
        i = M["loc2idx"].get(start_loc)
        j = M["loc2idx"].get(end_loc)
        dur = float(e.get("duration_min") or (M["time"][i, j] if i is not None and j is not None else 0.0))
        miles = float(e.get("miles") or (M["dist"][i, j] if i is not None and j is not None else 0.0))
        return {
            "id": f"CASCADE:{start_loc}->{end_loc}@{int(e.get('start_min', 0))}",
            "start_location": start_loc,
            "end_location": end_loc,
            "duration_minutes": dur,
            "trip_miles": miles,
        }

    def _compute_before_after_schedules(
        DATA: Dict[str, Any],
        assignments: List[AssignmentOut],
        cascades: List[Dict[str, Any]],
    ) -> List[DriverScheduleOut]:
        """
        Build per-driver schedules before/after:
        - 'before' is taken directly from DATA['driver_states'][driver]['elements']
        - We remove displaced legs (as indicated by cascades) from the original driver's 'after'
        - We add new/reassigned legs (those with trip_id starting with NEW:/CASCADE:) into the assigned driver's 'after'
        NOTE: This is still best-effort until cuOpt exact times are wired in.
        """
        ds = DATA.get("driver_states", {})
        drivers = ds.get("drivers", ds) if isinstance(ds, dict) else {}

        # Prepare 'before' and shallow-copy 'after'
        schedule_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for drv_id, meta in drivers.items():
            before = list(meta.get("elements", []))
            after = [dict(e) for e in before]
            schedule_map[drv_id] = {"before": before, "after": after}

        # Collect displaced legs from cascades (from/to pairs)
        displaced = []
        for c in cascades:
            # c has: driver_id, from, to, priority, displaced_by, depth
            d = c.get("driver_id")
            fr = str(c.get("from", "")).upper()
            to = str(c.get("to", "")).upper()
            if d and fr and to:
                displaced.append((d, fr, to))

        # Remove first matching travel leg from original driver's AFTER
        for (drv_id, fr, to) in displaced:
            m = schedule_map.get(drv_id)
            if not m:
                continue
            after = m["after"]
            for idx, e in enumerate(after):
                if e.get("is_travel") and str(e.get("from","")).upper() == fr and str(e.get("to","")).upper() == to:
                    del after[idx]
                    break

        # Add new assigned legs to target driver's AFTER
        def _leg_from_trip_id(trip_id: str) -> Dict[str, Any]:
            # e.g. NEW:A->B@1693650600 or CASCADE:A->B@12345 (we only trust from/to)
            leg: Dict[str, Any] = {"is_travel": True, "from": "", "to": "", "note": trip_id}
            core = trip_id.split("@", 1)[0]
            if ":" in core:
                core = core.split(":", 1)[1]
            if "->" in core:
                fr, to = core.split("->", 1)
                leg["from"] = str(fr).upper()
                leg["to"] = str(to).upper()
            return leg

        for a in assignments:
            # Only add concrete legs (the NEW/CASCADE trips we actually placed)
            if a.type in ("reassigned", "outsourced"):
                if isinstance(a.trip_id, str) and (a.trip_id.startswith("NEW:") or a.trip_id.startswith("CASCADE:")):
                    leg = _leg_from_trip_id(a.trip_id)
                    # Attach simple metrics if available
                    if a.delay_minutes is not None:
                        leg["delay_min"] = a.delay_minutes
                    if a.miles_delta is not None:
                        leg["miles"] = a.miles_delta
                    # Place on assigned driver if reassigned, otherwise mark as outsourced (no driver)
                    if a.type == "reassigned" and a.driver_id:
                        m = schedule_map.setdefault(a.driver_id, {"before": [], "after": []})
                        m["after"].append(leg)

        # Render as DriverScheduleOut list
        out: List[DriverScheduleOut] = []
        for drv_id, m in schedule_map.items():
            out.append(DriverScheduleOut(driver_id=drv_id, before=m["before"], after=m["after"]))
        return out

    def _bounds_from_when_local(when_local: str, minutes: float | None, mode: str, sla_windows: Dict[int, Dict[str, int]]) -> tuple[int, int]:
        """
        Convert ISO local like '2025-09-02T10:30' into [earliest, latest] minute bounds
        consistent with your weekday SLA window policy (SLA_WINDOWS), and the request's mode.

        - If mode == 'depart_after': earliest = when_min, latest = when_min + window_slack
        - If mode == 'arrive_before': earliest = when_min - window_slack, latest = when_min
        - If `minutes` (trip duration) is provided, we keep windows consistent with that.
        """
        # NOTE: you already have weekday_from_local(when_local); re-use it
        wk = weekday_from_local(when_local)  # 0..6
        # parse "YYYY-MM-DDTHH:MM"
        from datetime import datetime
        dt = datetime.strptime(when_local[:16], "%Y-%m-%dT%H:%M")
        when_min = dt.hour * 60 + dt.minute

        # default slack per weekday from SLA_WINDOWS, else 120 mins
        slack = int(SLA_WINDOWS.get(wk, {}).get("slack_min", 120))
        if mode == "arrive_before":
            earliest = max(0, when_min - slack)
            latest = when_min
        else:
            earliest = when_min
            latest = min(24 * 60, when_min + slack)

        # If we have a duration, keep window big enough to host it
        if minutes is not None:
            dur = int(round(float(minutes)))
            if mode == "arrive_before":
                earliest = min(earliest, latest - max(dur, 0))
            else:
                latest = max(latest, earliest + max(dur, 0))

        return int(earliest), int(latest)

    # ------------------- Endpoints -------------------

    @router.post("/candidates", response_model=PlanCandidatesResponse)
    def plan_candidates(req: PlanRequest):
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        weekday, trip_minutes, trip_miles, cands = generate_candidates(
            req, DATA, M, cfg, LOC_META, SLA_WINDOWS
        )
        return PlanCandidatesResponse(
            weekday=weekday, trip_minutes=trip_minutes, trip_miles=trip_miles, candidates=cands
        )

    @router.post("/solve_cascades", response_model=PlanSolveCascadeResponse)
    def plan_and_solve_cascades(req: PlanSolveCascadeRequest, request: Request):
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()

        deadhead_cpm = cfg.get("deadhead_cost_per_mile", cfg.get("deadhead_cost", 1.0))
        overtime_cpm = cfg.get("overtime_cost_per_minute", cfg.get("overtime_cost", 1.0))
        admin_cost = cfg.get("reassignment_admin_cost", 10.0)
        out_per_mile = cfg.get("outsourcing_per_mile", cfg.get("outsourcing_cost_per_mile", 2.0))

        i = M["loc2idx"][req.start_location.upper()]
        j = M["loc2idx"][req.end_location.upper()]
        trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else float(M["time"][i, j])
        trip_miles = float(req.trip_miles) if req.trip_miles is not None else float(M["dist"][i, j])

        root_trip = {
            "id": f"NEW:{req.start_location}->{req.end_location}@{req.when_local}",
            "start_location": req.start_location,
            "end_location": req.end_location,
            "duration_minutes": trip_minutes,
            "trip_miles": trip_miles,
        }

        cascades = []
        all_assignments = []
        total_obj = 0.0
        total_candidates_seen = 0
        affected_drivers: set[str] = set()
        visited: set[tuple[str, int]] = set()
        queue: List[tuple[Dict[str, Any], int, int]] = [(root_trip, req.priority, 0)]

        while queue:
            trip, prio, depth = queue.pop(0)
            if len(affected_drivers) >= req.max_drivers_affected:
                break

            cand_req = PlanRequest(
                start_location=trip["start_location"],
                end_location=trip["end_location"],
                mode=req.mode,
                when_local=req.when_local,
                priority=prio,
                top_n=50,
                trip_minutes=trip["duration_minutes"],
                trip_miles=trip["trip_miles"],
            )
            wk, tmn, tmi, cands = generate_candidates(cand_req, DATA, M, cfg, LOC_META, SLA_WINDOWS)
            total_candidates_seen += len(cands)

            chosen = cands[0] if cands else None
            if chosen is None:
                base = float(cfg.get("outsourcing_base_cost", 200.0))
                cost = base + trip["trip_miles"] * out_per_mile
                all_assignments.append(AssignmentOut(
                    trip_id=trip["id"], type="outsourced", driver_id=None, candidate_id="OUTSOURCE",
                    cost=cost,
                    cost_breakdown={
                        "outsourcing_base": base,
                        "outsourcing_miles": trip["trip_miles"] * out_per_mile
                    },
                    miles_delta=trip["trip_miles"]
                ))
                total_obj += cost
                continue

            affected_drivers.add(chosen.driver_id)
            bd = {"admin": admin_cost}
            if chosen.deadhead_miles:
                bd["deadhead"] = chosen.deadhead_miles * deadhead_cpm
            if chosen.overtime_minutes:
                bd["overtime"] = chosen.overtime_minutes * overtime_cpm
            cost = sum(bd.values())
            all_assignments.append(AssignmentOut(
                trip_id=trip["id"], type="reassigned", driver_id=chosen.driver_id,
                candidate_id=chosen.candidate_id, delay_minutes=chosen.delay_minutes,
                deadhead_miles=chosen.deadhead_miles, overtime_minutes=chosen.overtime_minutes,
                miles_delta=chosen.miles_delta, cost=cost, cost_breakdown=bd
            ))
            total_obj += cost

            if depth < req.max_cascades and "swap_leg@" in chosen.candidate_id:
                ds = DATA["driver_states"]
                drivers = ds["drivers"] if "drivers" in ds else ds
                m = drivers.get(chosen.driver_id, {})
                leg = _find_leg_by_candidate_id(m, chosen.candidate_id)
                if leg:
                    leg_pri = int(leg.get("priority", 3))
                    if leg_pri >= prio:
                        displaced_trip = _build_trip_from_leg(leg, M)
                        key = (chosen.driver_id, int(leg.get("start_min", -1)))
                        if key not in visited:
                            visited.add(key)
                            cascades.append({
                                "depth": depth + 1,
                                "displaced_by": chosen.candidate_id,
                                "driver_id": chosen.driver_id,
                                "from": displaced_trip["start_location"],
                                "to": displaced_trip["end_location"],
                                "priority": leg_pri
                            })
                            queue.append((displaced_trip, leg_pri, depth + 1))

        return PlanSolveCascadeResponse(
            weekday=weekday_from_local(req.when_local),
            trip_minutes=trip_minutes,
            trip_miles=trip_miles,
            objective_value=total_obj,
            assignments=all_assignments,
            details={
                "backend": "cascade-greedy",
                "max_cascades": req.max_cascades,
                "drivers_touched": len(affected_drivers)
            },
            candidates_considered=total_candidates_seen,
            cascades=cascades,
        )

    @router.post("/solve_multi", response_model=PlanSolveMultiResponse)
    def plan_solve_multi(req: PlanSolveMultiRequest, request: Request):
        """
        Branching search that explores top-N candidates per step to produce multiple
        feasible cascaded solutions. If req.use_cuopt is True, we refine/replace
        with cuOpt results (adapter stub provided).
        """
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()

        # Compute root trip stats
        i = M["loc2idx"][req.start_location.upper()]
        j = M["loc2idx"][req.end_location.upper()]
        trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else float(M["time"][i, j])
        trip_miles   = float(req.trip_miles)   if req.trip_miles   is not None else float(M["dist"][i, j])
        earliest, latest = _bounds_from_when_local(
            req.when_local,
            trip_minutes,
            req.mode,
            SLA_WINDOWS
        )

        # BFS/DFS over cascades, exploring the top-N (branch factor) each step
        # A state is a queue of "trips to place", accumulated assignments, cascades, affected drivers
        from collections import deque
        root_trip = {
            "id": f"NEW:{req.start_location}->{req.end_location}@{req.when_local}",
            "start_location": req.start_location,
            "end_location": req.end_location,
            "duration_minutes": trip_minutes,
            "trip_miles": trip_miles,
            "priority": req.priority,
        }

        deadhead_cpm = cfg.get("deadhead_cost_per_mile", cfg.get("deadhead_cost", 1.0))
        overtime_cpm = cfg.get("overtime_cost_per_minute", cfg.get("overtime_cost", 1.0))
        admin_cost   = cfg.get("reassignment_admin_cost", 10.0)
        out_per_mile = cfg.get("outsourcing_per_mile", cfg.get("outsourcing_cost_per_mile", 2.0))

        def cost_of_assignment(a: AssignmentOut) -> float:
            bd = a.cost_breakdown or {}
            if not bd:
                # derive quickly if missing
                c = admin_cost
                if a.deadhead_miles:
                    c += a.deadhead_miles * deadhead_cpm
                if a.overtime_minutes:
                    c += a.overtime_minutes * overtime_cpm
                return float(c)
            return float(sum(bd.values()))

        # state tuple: (queue_of_trips, assignments, cascades, affected_drivers_set, total_cost, candidates_seen)
        init = (deque([(root_trip, req.priority, 0)]), [], [], set(), 0.0, 0)
        solutions: List[PlanSolutionOut] = []

        while init and len(solutions) < req.max_solutions:
            queue, assignments, cascades, affected, total_cost, seen = init
            # If nothing left to place -> record a solution
            if not queue:
                # Build before/after schedules
                schedules = _compute_before_after_schedules(DATA, assignments, cascades)
                sol = PlanSolutionOut(
                    rank=0,
                    objective_value=float(total_cost),
                    drivers_touched=len(affected),
                    assignments=assignments,
                    cascades=cascades,
                    schedules=schedules,
                    details={"backend": "branch-greedy", "note": "heuristic multi"},
                )
                solutions.append(sol)
                break

            # Expand one layer (one trip)
            trip, prio, depth = queue.popleft()

            # Generate candidates for this trip
            cand_req = PlanRequest(
                start_location=trip["start_location"],
                end_location=trip["end_location"],
                mode=req.mode,
                when_local=req.when_local,
                priority=prio,
                top_n=max(3*req.top_n_per_step, 10),
                trip_minutes=trip.get("duration_minutes"),
                trip_miles=trip.get("trip_miles"),
            )
            wk, tmn, tmi, cands = generate_candidates(cand_req, DATA, M, cfg, LOC_META, SLA_WINDOWS)
            # Sort by our current greedy preference (generate_candidates likely returns already sorted best-first)
            topK = cands[: req.top_n_per_step]
            # Also include a fallback 'outsourced' branch (if allowed)
            if not topK:
                base = float(cfg.get("outsourcing_base_cost", 200.0))
                cost = base + float(trip["trip_miles"]) * out_per_mile
                new_assign = AssignmentOut(
                    trip_id=trip["id"], type="outsourced", driver_id=None, candidate_id="OUTSOURCE",
                    delay_minutes=0.0, uses_emergency_rest=False,
                    deadhead_miles=0.0, overtime_minutes=0.0, miles_delta=float(trip["trip_miles"]),
                    cost=float(cost),
                    cost_breakdown={"outsourcing_base": base, "outsourcing_miles": float(trip["trip_miles"] * out_per_mile)},
                )
                new_assignments = assignments + [new_assign]
                new_total = total_cost + float(cost)
                # No further cascade from outsourced
                if not queue:
                    schedules = _compute_before_after_schedules(DATA, new_assignments, cascades)
                    solutions.append(PlanSolutionOut(
                        rank=0,
                        objective_value=new_total,
                        drivers_touched=len(affected),
                        assignments=new_assignments,
                        cascades=cascades,
                        schedules=schedules,
                        details={"backend": "branch-greedy", "note": "outsourced leaf"},
                    ))
                    continue
                else:
                    # Continue search for remaining queue
                    init = (queue.copy(), new_assignments, list(cascades), set(affected), new_total, seen)
                    continue

            # Branch on topK candidates
            branches = []
            for chosen in topK:
                # Respect limit on affected drivers
                affected2 = set(affected)
                if chosen.driver_id:
                    affected2.add(chosen.driver_id)
                if len(affected2) > req.max_drivers_affected:
                    continue

                bd: Dict[str, float] = {"admin": float(admin_cost)}
                if chosen.deadhead_miles:
                    bd["deadhead"] = float(chosen.deadhead_miles * deadhead_cpm)
                if chosen.overtime_minutes:
                    bd["overtime"] = float(chosen.overtime_minutes * overtime_cpm)
                cost = float(sum(bd.values()))

                new_assign = AssignmentOut(
                    trip_id=trip["id"], type="reassigned", driver_id=chosen.driver_id,
                    candidate_id=chosen.candidate_id, delay_minutes=chosen.delay_minutes,
                    deadhead_miles=chosen.deadhead_miles, overtime_minutes=chosen.overtime_minutes,
                    miles_delta=chosen.miles_delta, cost=cost, cost_breakdown=bd
                )

                new_queue = deque(queue)
                new_casc = list(cascades)
                # If displacement -> enqueue the displaced trip
                if depth < req.max_cascades and "swap_leg@" in chosen.candidate_id:
                    ds = DATA["driver_states"]
                    drivers = ds["drivers"] if "drivers" in ds else ds
                    m = drivers.get(chosen.driver_id, {})
                    leg = _find_leg_by_candidate_id(m, chosen.candidate_id)
                    if leg:
                        leg_pri = int(leg.get("priority", 3))
                        if leg_pri >= prio:
                            displaced_trip = _build_trip_from_leg(leg, M)
                            new_casc.append({
                                "depth": depth + 1,
                                "displaced_by": chosen.candidate_id,
                                "driver_id": chosen.driver_id,
                                "from": displaced_trip["start_location"],
                                "to": displaced_trip["end_location"],
                                "priority": leg_pri
                            })
                            new_queue.append((displaced_trip, leg_pri, depth + 1))

                branches.append((
                    new_queue,
                    assignments + [new_assign],
                    new_casc,
                    affected2,
                    total_cost + cost,
                    seen + len(cands)
                ))

            # Choose next branch to expand (best-first by current cost and drivers touched)
            if not branches:
                # dead-end, continue loop to see if queue empties into solution
                continue
            branches.sort(key=lambda b: (b[4], len(b[3])))  # by new_total_cost then by drivers affected
            init = branches[0]  # best-first; for k-best, you could push more than one branch to a heap

        # Final sorting/ranking
        solutions.sort(key=lambda s: (s.objective_value, s.drivers_touched))
        for r, s in enumerate(solutions, 1):
            s.rank = r

        # Optional: refine with cuOpt if requested
        if req.use_cuopt and solutions:
            try:
                cuopt_url = get_cuopt_url()
                refined = []
                for s in solutions[: req.max_solutions]:
                    # Build one payload per candidate (you can also batch if your cuOpt supports)
                    payload = build_cuopt_payload(
                        DATA=DATA,
                        request_trip=root_trip,
                        assignments_so_far=[a.dict() for a in s.assignments],
                        priorities=PRIORITY_MAP,
                        sla_windows=SLA_WINDOWS,
                        M=M,
                        new_req_window=[earliest, latest],  # pass down
                    )
                    raw = solve_with_cuopt(cuopt_url, payload)
                    variants = extract_solutions_from_cuopt(raw, max_solutions=1)
                    if variants:
                        v = variants[0]
                        s.objective_value = float(v.get("objective_value", s.objective_value))
                        s.details["backend"] = "cuopt"
                # resort after refinement
                solutions.sort(key=lambda s: (s.objective_value, s.drivers_touched))
                for r, s in enumerate(solutions, 1):
                    s.rank = r
            except Exception as e:
                # Non-fatal: keep heuristic results
                pass

        return PlanSolveMultiResponse(
            weekday=weekday_from_local(req.when_local),
            trip_minutes=trip_minutes,
            trip_miles=trip_miles,
            solutions=solutions[: req.max_solutions],
            meta={"backend": "multi-solver", "note": "branching heuristic; cuOpt optional"},
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
