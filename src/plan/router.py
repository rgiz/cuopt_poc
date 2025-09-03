
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Callable
from fastapi import APIRouter, HTTPException, Request

from .models import (
    PlanRequest, PlanCandidatesResponse, CandidateOut,
    PlanSolveCascadeRequest, PlanSolveCascadeResponse, AssignmentOut
)
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

    # Load static config once (safe to do here)
    _PRIORITY_MAP = load_priority_map()
    _SLA_WINDOWS  = load_sla_windows()

    # ----- shared helpers (closure over get_data / get_cost_config) -----
    def ensure_ready():
        """
        Returns (DATA, M, LOC_META)
          - DATA: the backend data dict
          - M:    dict with keys: dist, time, loc2idx
          - LOC_META: optional metadata built from locations df (if present)
        """
        DATA = get_data()
        if DATA is None:
            raise HTTPException(
                status_code=503,
                detail="Private data not loaded. Upload/build and POST /admin/reload.",
            )
        M = {
            "dist":   DATA["distance"],
            "time":   DATA["time"],
            "loc2idx": DATA["location_to_index"],
        }

        # Try to build richer meta for geo filters if a locations df was loaded
        loc_meta: Dict[str, Dict[str, Any]] = {}
        try:
            if DATA.get("locations_df") is not None:
                loc_meta = build_loc_meta_from_locations_csv(DATA["locations_df"])
        except Exception:
            # best-effort only
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
        end_loc   = str(e.get("to", "")).upper()
        i = M["loc2idx"].get(start_loc)
        j = M["loc2idx"].get(end_loc)
        dur   = float(e.get("duration_min") or (M["time"][i, j] if i is not None and j is not None else 0.0))
        miles = float(e.get("miles")        or (M["dist"][i, j] if i is not None and j is not None else 0.0))
        return {
            "id": f"CASCADE:{start_loc}->{end_loc}@{int(e.get('start_min', 0))}",
            "start_location": start_loc,
            "end_location": end_loc,
            "duration_minutes": dur,
            "trip_miles": miles,
        }

    # ------------------- endpoints -------------------

    @router.post("/candidates", response_model=PlanCandidatesResponse)
    def plan_candidates(req: PlanRequest):
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        weekday, trip_minutes, trip_miles, cands = generate_candidates(
            req, DATA, M, cfg, LOC_META, _SLA_WINDOWS
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
        admin_cost   = cfg.get("reassignment_admin_cost", 10.0)
        out_per_mile = cfg.get("outsourcing_per_mile", cfg.get("outsourcing_cost_per_mile", 2.0))

        i = M["loc2idx"][req.start_location.upper()]
        j = M["loc2idx"][req.end_location.upper()]
        trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else float(M["time"][i, j])
        trip_miles   = float(req.trip_miles)   if req.trip_miles   is not None else float(M["dist"][i, j])

        root_trip = {
            "id": f"NEW:{req.start_location}->{req.end_location}@{req.when_local}",
            "start_location": req.start_location,
            "end_location": req.end_location,
            "duration_minutes": trip_minutes,
            "trip_miles": trip_miles,
        }

        cascades: List[Dict[str, Any]] = []
        all_assignments: List[AssignmentOut] = []
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
            wk, tmn, tmi, cands = generate_candidates(cand_req, DATA, M, cfg, LOC_META, _SLA_WINDOWS)
            total_candidates_seen += len(cands)

            chosen = cands[0] if cands else None
            if chosen is None:
                base = float(cfg.get("outsourcing_base_cost", 200.0))
                cost = base + trip["trip_miles"] * out_per_mile
                all_assignments.append(AssignmentOut(
                    trip_id=trip["id"], type="outsourced", driver_id=None, candidate_id="OUTSOURCE",
                    cost=float(cost),
                    cost_breakdown={"outsourcing_base": base, "outsourcing_miles": float(trip["trip_miles"] * out_per_mile)},
                    miles_delta=float(trip["trip_miles"])
                ))
                total_obj += float(cost)
                continue

            affected_drivers.add(chosen.driver_id)
            bd: Dict[str, float] = {"admin": float(admin_cost)}
            if chosen.deadhead_miles:
                bd["deadhead"] = float(chosen.deadhead_miles * deadhead_cpm)
            if chosen.overtime_minutes:
                bd["overtime"] = float(chosen.overtime_minutes * overtime_cpm)
            cost = float(sum(bd.values()))
            all_assignments.append(AssignmentOut(
                trip_id=trip["id"], type="reassigned", driver_id=chosen.driver_id,
                candidate_id=chosen.candidate_id, delay_minutes=chosen.delay_minutes,
                deadhead_miles=chosen.deadhead_miles, overtime_minutes=chosen.overtime_minutes,
                miles_delta=chosen.miles_delta, cost=cost, cost_breakdown=bd
            ))
            total_obj += cost

            # cascade if we displaced an equal/lower-priority leg
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
            trip_minutes=float(root_trip["duration_minutes"]),
            trip_miles=float(root_trip["trip_miles"]),
            objective_value=float(total_obj),
            assignments=all_assignments,
            details={"backend": "cascade-greedy", "max_cascades": req.max_cascades, "drivers_touched": len(affected_drivers)},
            candidates_considered=total_candidates_seen,
            cascades=cascades,
        )

    @router.post("/solve", response_model=Dict[str, Any])
    def plan_and_solve(req: PlanRequest, request: Request):
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()

        out_per_mile = cfg.get("outsourcing_per_mile", cfg.get("outsourcing_cost_per_mile", 2.0))
        admin_cost   = cfg.get("reassignment_admin_cost", 10.0)
        deadhead_cpm = cfg.get("deadhead_cost_per_mile", cfg.get("deadhead_cost", 1.0))
        overtime_cpm = cfg.get("overtime_cost_per_minute", cfg.get("overtime_cost", 1.0))

        wk, trip_minutes, trip_miles, cands = generate_candidates(
            req, DATA, M, cfg, LOC_META, _SLA_WINDOWS
        )
        if not cands:
            base = float(cfg.get("outsourcing_base_cost", 200.0))
            cost = base + trip_miles * out_per_mile
            return {
                "weekday": wk,
                "trip_minutes": trip_minutes,
                "trip_miles": trip_miles,
                "objective_value": float(cost),
                "assignments": [{
                    "trip_id": f"NEW:{req.start_location}->{req.end_location}@{req.when_local}",
                    "type": "outsourced",
                    "driver_id": None,
                    "candidate_id": "OUTSOURCE",
                    "delay_minutes": 0.0,
                    "uses_emergency_rest": False,
                    "deadhead_miles": 0.0,
                    "overtime_minutes": 0.0,
                    "miles_delta": float(trip_miles),
                    "cost": float(cost),
                    "cost_breakdown": {"outsourcing_base": base, "outsourcing_miles": float(trip_miles * out_per_mile)},
                }],
                "details": {"backend": "simple-greedy", "note": "No candidates, outsourced"},
                "candidates_considered": 0
            }

        chosen = cands[0]
        bd: Dict[str, float] = {"admin": float(admin_cost)}
        if chosen.deadhead_miles:
            bd["deadhead"] = float(chosen.deadhead_miles * deadhead_cpm)
        if chosen.overtime_minutes:
            bd["overtime"] = float(chosen.overtime_minutes * overtime_cpm)
        cost = float(sum(bd.values()))

        return {
            "weekday": wk,
            "trip_minutes": trip_minutes,
            "trip_miles": trip_miles,
            "objective_value": float(cost),
            "assignments": [{
                "trip_id": f"NEW:{req.start_location}->{req.end_location}@{req.when_local}",
                "type": "reassigned",
                "driver_id": chosen.driver_id,
                "candidate_id": chosen.candidate_id,
                "delay_minutes": chosen.delay_minutes,
                "uses_emergency_rest": False,
                "deadhead_miles": chosen.deadhead_miles,
                "overtime_minutes": chosen.overtime_minutes,
                "miles_delta": chosen.miles_delta,
                "cost": float(cost),
                "cost_breakdown": bd,
            }],
            "details": {"backend": "simple-greedy"},
            "candidates_considered": len(cands)
        }

    @router.get("/locations")
    def list_locations():
        """
        Return location names for UI autocompletion.
        1) Prefer in-memory mapping (DATA["location_to_index"])
        2) Else try CSVs in PRIVATE_DATA_DIR(/active) or /data
        """
        # 1) Try live memory first
        try:
            _DATA, M, _ = ensure_ready()
            names = sorted({str(k).strip() for k in M["loc2idx"].keys() if str(k).strip()})
            return {"names": names, "count": len(names), "source": "memory"}
        except HTTPException:
            pass

        # 2) CSV fallback
        import os
        import pandas as pd
        from pathlib import Path

        base = Path(os.getenv("PRIVATE_DATA_DIR", "/data")).resolve()
        dataset = base / "active" if (base / "active").exists() else base

        for fn in ("location_index.csv", "locations.csv"):
            p = dataset / fn
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    for col in ("name", "NAME", "site_name", "Site", "site", "location_id"):
                        if col in df.columns:
                            names = sorted({str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()})
                            return {"names": names, "count": len(names), "source": str(p)}
                except Exception as e:
                    return {"names": [], "count": 0, "source": str(p), "error": f"read-failed: {e}"}
        return {"names": [], "count": 0, "source": "none"}

    return router