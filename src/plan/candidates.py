from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from .models import PlanRequest, CandidateOut, WEEKDAYS
from .config import ENFORCE_SAME_ISLAND, USE_HAVERSINE_DEADHEAD, HAV_MAX_DEADHEAD_ONE_WAY_MI
from .geo import same_island_by_meta, haversine_between_idx

def minute_of_day_local(s: str) -> int:
    dt = datetime.fromisoformat(s) if "T" in s else datetime.strptime(s, "%Y-%m-%d %H:%M")
    if dt.tzinfo is None: dt = dt.replace(tzinfo=ZoneInfo("Europe/London"))
    else: dt = dt.astimezone(ZoneInfo("Europe/London"))
    return dt.hour * 60 + dt.minute

def weekday_from_local(s: str) -> str:
    dt = datetime.fromisoformat(s) if "T" in s else datetime.strptime(s, "%Y-%m-%d %H:%M")
    if dt.tzinfo is None: dt = dt.replace(tzinfo=ZoneInfo("Europe/London"))
    else: dt = dt.astimezone(ZoneInfo("Europe/London"))
    return WEEKDAYS[dt.weekday()]

def row_flag_true(v) -> bool:
    if v is None: return False
    try: return int(v) == 1
    except Exception: return str(v).strip().lower() in ("true","t","yes","y")

def element_active_on_weekday(e: dict, weekday: str) -> bool:
    if weekday in e:
        return row_flag_true(e.get(weekday))
    days_list = e.get("days")
    if isinstance(days_list, (list,tuple,set)):
        return weekday in {str(d).title()[:3] for d in days_list}
    wd = e.get("weekday")
    if isinstance(wd,str):
        return weekday == wd.title()[:3]
    return False

def idx_of(name: str, loc2idx: Dict[str,int]) -> int:
    key = str(name).upper().strip()
    if key not in loc2idx:
        raise ValueError(f"Unknown location '{name}'. Rebuild locations.")
    return int(loc2idx[key])

def minutes_between(i: int, j: int, tmat: np.ndarray) -> float:
    return max(0.0, float(tmat[i,j]))

def miles_between(i: int, j: int, dmat: np.ndarray) -> float:
    return max(0.0, float(dmat[i,j]))

def condensed_first_last_ids(elements: List[Dict[str,Any]], loc2idx: Dict[str,int]) -> Tuple[Optional[int], Optional[int]]:
    if not elements: return None, None
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    for e in elements:
        if start_id is None:
            if e.get("from_id") is not None: start_id = int(e["from_id"])
            else:
                nm = str(e.get("from","")).upper().strip()
                if nm in loc2idx: start_id = int(loc2idx[nm])
        if e.get("to_id") is not None: end_id = int(e["to_id"])
        else:
            nm2 = str(e.get("to","")).upper().strip()
            if nm2 in loc2idx: end_id = int(loc2idx[nm2])
    return start_id, end_id

def generate_candidates(
    req: PlanRequest,
    DATA: Dict[str,Any],
    matrices: Dict[str,Any],
    cost_cfg: Dict[str,float],
    loc_meta: Dict[str,Any],
    sla_windows: Dict[int, Dict[str,int]],
) -> Tuple[str, float, float, List[CandidateOut]]:
    # SLA & timing
    req_min  = minute_of_day_local(req.when_local)
    sla      = sla_windows.get(int(req.priority), {"early_min": 60, "late_min": 60})
    earliest = max(0, int(req_min) - int(sla["early_min"]))
    latest   = int(req_min) + int(sla["late_min"])

    same_island_required = bool(sla.get("enforce_same_island", ENFORCE_SAME_ISLAND))
    hav_max_dd_mi        = float(sla.get("hav_max_deadhead_one_way_mi", HAV_MAX_DEADHEAD_ONE_WAY_MI))

    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]

    weekday   = weekday_from_local(req.when_local)
    start_idx = idx_of(req.start_location, loc2idx)
    end_idx   = idx_of(req.end_location,   loc2idx)

    trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else minutes_between(start_idx, end_idx, Mtime)
    trip_miles   = float(req.trip_miles)   if req.trip_miles   is not None else miles_between(start_idx, end_idx, Mdist)

    # cost config
    deadhead_cpm = cost_cfg.get("deadhead_cost_per_mile", cost_cfg.get("deadhead_cost", 1.0))
    overtime_cpm = cost_cfg.get("overtime_cost_per_minute", cost_cfg.get("overtime_cost", 1.0))
    admin_cost   = cost_cfg.get("reassignment_admin_cost", 10.0)
    max_duty_min = int(cost_cfg.get("max_duty_minutes", 13*60))

    def passes_geo(from_idx: Optional[int], to_idx: int) -> bool:
        if from_idx is None: return True
        if same_island_required:
            # derive meta for both indices
            nm_i = next((k for k,v in loc2idx.items() if int(v)==int(from_idx)), None)
            nm_j = next((k for k,v in loc2idx.items() if int(v)==int(to_idx)), None)
            mi = loc_meta.get(nm_i.upper()) if nm_i else None
            mj = loc_meta.get(nm_j.upper()) if nm_j else None
            same = same_island_by_meta(mi, mj)
            if same is False: return False
        if USE_HAVERSINE_DEADHEAD:
            hdd = haversine_between_idx(from_idx, to_idx, loc2idx, loc_meta)
            if hdd is not None and hdd > hav_max_dd_mi: return False
        return True

    # helpers
    def _is_travel_leg(e: Dict[str,Any]) -> bool:
        if e.get("is_travel") is True: return True
        et = str(e.get("element_type","")).upper()
        return ("TRAVEL" in et) or ("LEG" in et and "TRAVEL" in et)

    def _is_empty_planz(e: Dict[str,Any]) -> bool:
        pc = str(e.get("planz_code", e.get("Planz Code",""))).strip().upper()
        return ("EMPTY" in pc) or ("TRAVEL_NO_DATA" in pc) or ("TRAVEL NO DATA" in pc) or bool(e.get("is_empty", False))

    def _same_loc(a: Optional[int], b: Optional[int]) -> bool:
        try: return (a is not None) and (b is not None) and (int(a)==int(b))
        except Exception: return False

    candidates: List[CandidateOut] = []

    # Tier 0 – take an existing EMPTY A->B
    def tier0_empty_A_to_B() -> List[CandidateOut]:
        ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
        out: List[CandidateOut] = []
        for duty_id, meta in drivers.items():
            elements_all = meta.get("elements", []) or []
            elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
            if not elements: continue

            first_id, _ = condensed_first_last_ids(elements, loc2idx)
            if not passes_geo(first_id, start_idx): continue

            for e in elements:
                if not _is_travel_leg(e):  continue
                if not _is_empty_planz(e): continue
                if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
                    continue
                s = e.get("start_min"); en = e.get("end_min")
                if s is None or en is None: continue
                ok_time = (
                    (req.mode == "depart_after"  and earliest <= s <= latest) or
                    (req.mode == "arrive_before" and earliest <= en <= latest)
                )
                if not ok_time: continue
                out.append(CandidateOut(
                    candidate_id=f"{duty_id}::take_empty@{int(s)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=True, est_cost=float(admin_cost),
                ))
        out.sort(key=lambda c: (int(c.candidate_id.split("@")[1]), c.driver_id))
        return out

    # Tier 1 – swap a departure from A (A->X) same/lower priority
    def tier1_swap_from_A() -> List[CandidateOut]:
        ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
        out: List[CandidateOut] = []
        for duty_id, meta in drivers.items():
            elements_all = meta.get("elements", []) or []
            elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
            if not elements: continue

            first_id, _ = condensed_first_last_ids(elements, loc2idx)
            if not passes_geo(first_id, start_idx): continue

            for e in elements:
                if not _is_travel_leg(e): continue
                if not _same_loc(e.get("from_id"), start_idx): continue
                s = e.get("start_min"); en = e.get("end_min")
                if s is None or en is None: continue
                ok_time = (
                    (req.mode == "depart_after"  and earliest <= s <= latest) or
                    (req.mode == "arrive_before" and earliest <= en <= latest)
                )
                if not ok_time: continue
                if int(e.get("priority", 3)) < int(req.priority): continue
                out.append(CandidateOut(
                    candidate_id=f"{duty_id}::swap_from_A@{int(s)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=True, est_cost=float(admin_cost),
                ))
        out.sort(key=lambda c: int(c.candidate_id.split("@")[1]))
        return out

    candidates.extend(tier0_empty_A_to_B())
    candidates.extend(tier1_swap_from_A())

    # Per-driver slack / exact A->B swap / append
    ds = DATA["driver_states"] or {}
    drivers = ds["drivers"] if isinstance(ds, dict) and "drivers" in ds else ds

    for duty_id, meta in drivers.items():
        elements_all = meta.get("elements", []) or []
        elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
        if not elements: continue

        daily_windows = meta.get("daily_windows", {})
        if weekday in daily_windows:
            win = daily_windows[weekday]
            start_min = int(win.get("start_min", 0)); end_min = int(win.get("end_min", 0))
            orig_length = max(0, end_min - start_min)
        else:
            durations = [int(e["duration_min"]) for e in elements if e.get("duration_min") is not None]
            orig_length = int(sum(durations)) if durations else 8*60

        first_id, last_id = condensed_first_last_ids(elements, loc2idx)

        # Slack (AS DIRECTED)
        if not passes_geo(first_id, start_idx):
            pass
        for e in elements:
            et = str(e.get("element_type","")).upper()
            if "AS DIRECTED" not in et: continue
            loc_id = e.get("from_id") or e.get("to_id")
            if loc_id is None:
                nm_from = str(e.get("from","")).upper().strip()
                nm_to   = str(e.get("to","")).upper().strip()
                if nm_from in loc2idx:   loc_id = loc2idx[nm_from]
                elif nm_to in loc2idx:   loc_id = loc2idx[nm_to]
            if loc_id is None: continue
            loc_id = int(loc_id)
            e_start = e.get("start_min"); e_dur = e.get("duration_min")
            if e_start is None or e_dur is None: continue
            e_end = e_start + e_dur
            fits_time = not (
                (req.mode == "depart_after"  and (e_end < earliest or e_start > latest)) or
                (req.mode == "arrive_before" and (e_end < earliest or e_start > req_min))
            )
            if not fits_time: continue
            dd_to_min   = minutes_between(loc_id, start_idx, Mtime)
            dd_back_min = minutes_between(end_idx,  loc_id, Mtime)
            budget      = dd_to_min + trip_minutes + dd_back_min
            if budget > e_dur: continue
            if not passes_geo(loc_id, start_idx): continue
            dd_mi = miles_between(loc_id, start_idx, Mdist) + miles_between(end_idx, loc_id, Mdist)
            est_cost = (dd_mi * deadhead_cpm) + admin_cost
            candidates.append(CandidateOut(
                candidate_id=f"{duty_id}::slack@{loc_id}",
                driver_id=str(duty_id), route_id=str(duty_id),
                deadhead_miles=float(dd_mi),
                deadhead_minutes=float(budget - trip_minutes),
                overtime_minutes=0.0,
                miles_delta=float(dd_mi + trip_miles),
                delay_minutes=0.0,
                uses_emergency_rest=False,
                feasible_hard=True,
                est_cost=float(est_cost),
            ))

        # Exact A->B swap
        if not passes_geo(first_id, start_idx):
            pass
        for e in elements:
            if not _is_travel_leg(e): continue
            if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
                continue
            e_start = e.get("start_min"); e_end = e.get("end_min")
            if e_start is None or e_end is None: continue
            ok_time = (
                (req.mode == "depart_after"  and (e_start >= earliest) and (e_start <= latest)) or
                (req.mode == "arrive_before" and (e_end   <= latest)   and (e_end   >= earliest))
            )
            if not ok_time: continue
            if int(e.get("priority", 3)) < int(req.priority): continue
            if not passes_geo(first_id, start_idx): continue
            candidates.append(CandidateOut(
                candidate_id=f"{duty_id}::swap_leg@{int(e_start)}",
                driver_id=str(duty_id), route_id=str(duty_id),
                deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
                feasible_hard=True, est_cost=float(admin_cost),
            ))

        # Append after last
        if last_id is not None:
            dd_min   = minutes_between(last_id, start_idx, Mtime)
            dd_mi    = miles_between(last_id, start_idx, Mdist)
            added    = dd_min + trip_minutes
            new_len  = orig_length + added
            feasible = (new_len <= max_duty_min) and passes_geo(last_id, start_idx)
            est_cost = (dd_mi * deadhead_cpm) + (added * overtime_cpm) + admin_cost
            candidates.append(CandidateOut(
                candidate_id=f"{duty_id}::append",
                driver_id=str(duty_id), route_id=str(duty_id),
                deadhead_miles=float(dd_mi),
                deadhead_minutes=float(dd_min),
                overtime_minutes=float(added),
                miles_delta=float(dd_mi + trip_miles),
                delay_minutes=0.0,
                uses_emergency_rest=False,
                feasible_hard=bool(feasible),
                est_cost=float(est_cost),
            ))

    def rank_key(c: CandidateOut):
        cid = c.candidate_id
        if "::take_empty@" in cid:   t = 0
        elif "::swap_leg@" in cid:   t = 1
        elif "::swap_from_A@" in cid:t = 2
        elif "::slack@" in cid:      t = 3
        else:                        t = 4
        return (not c.feasible_hard, t, c.est_cost)

    candidates.sort(key=rank_key)
    if req.top_n and req.top_n > 0:
        candidates = candidates[:req.top_n]

    return weekday, float(trip_minutes), float(trip_miles), candidates
