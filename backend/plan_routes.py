# backend/plan_routes.py
from __future__ import annotations

from curses import meta
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import traceback
import numpy as np
import json, os, math

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


# =========================
# Env + file helpers
# =========================
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

ENFORCE_SAME_ISLAND = _env_bool("ENFORCE_SAME_ISLAND", True)
USE_HAVERSINE_DEADHEAD = _env_bool("USE_HAVERSINE_DEADHEAD", True)
HAV_MAX_DEADHEAD_ONE_WAY_MI = _env_float("HAV_MAX_DEADHEAD_ONE_WAY_MI", 120.0)

def _dataset_dir() -> Path:
    base = Path(os.getenv("PRIVATE_DATA_DIR", "./data/private")).resolve()
    d = base / "active"
    return d if d.exists() else base

def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _load_priority_map() -> Dict[str,int]:
    mp = _load_json(_dataset_dir() / "priority_map.json", {})
    return {str(k).upper(): int(v) for k, v in mp.items()}

def _load_sla_windows() -> Dict[int, Dict[str,int]]:
    raw = _load_json(_dataset_dir() / "sla_windows.json", {})
    out: Dict[int, Dict[str,int]] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = {
                "early_min": int(v.get("early_min", 60)),
                "late_min":  int(v.get("late_min", 60)),
            }
        except Exception:
            continue
    if not out:
        out = {
            1: {"early_min":15, "late_min":30},
            2: {"early_min":30, "late_min":45},
            3: {"early_min":60, "late_min":60},
            4: {"early_min":90, "late_min":90},
            5: {"early_min":120,"late_min":120},
        }
    return out

def _minute_of_day_local(s: str) -> int:
    dt = datetime.fromisoformat(s) if "T" in s else datetime.strptime(s, "%Y-%m-%d %H:%M")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/London"))
    else:
        dt = dt.astimezone(ZoneInfo("Europe/London"))
    return dt.hour * 60 + dt.minute

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# =========================
# Pydantic models
# =========================
class PlanRequest(BaseModel):
    start_location: str = Field(..., description="Canonical site name (matches location_index.csv: name)")
    end_location:   str = Field(..., description="Canonical site name")
    mode:           str = Field(..., pattern="^(depart_after|arrive_before)$")
    when_local:     str = Field(..., description="Local datetime in Europe/London, e.g. 2025-08-18T21:30")
    priority:       int = Field(ge=1, le=5, default=3)
    trip_minutes: Optional[float] = Field(None, description="Override travel minutes; else use matrix")
    trip_miles:   Optional[float] = Field(None, description="Override miles; else use matrix")
    top_n:             int = Field(20, ge=1, le=200, description="Limit candidates returned")

class PlanSolveCascadeRequest(PlanRequest):
    max_cascades:           int = Field(2, ge=0, le=5)
    max_drivers_affected:   int = Field(5, ge=1, le=50)

class AssignmentOut(BaseModel):
    trip_id: str
    type: str
    driver_id: Optional[str] = None
    candidate_id: Optional[str] = None
    delay_minutes: float = 0.0
    uses_emergency_rest: bool = False
    deadhead_miles: float = 0.0
    overtime_minutes: float = 0.0
    miles_delta: float = 0.0
    cost: float = 0.0
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)

class PlanSolveCascadeResponse(BaseModel):
    weekday: str
    trip_minutes: float
    trip_miles: float
    objective_value: float
    assignments: List[AssignmentOut]
    details: Dict[str, Any]
    candidates_considered: int
    cascades: List[Dict[str, Any]]

class CandidateOut(BaseModel):
    candidate_id: str
    driver_id: str
    route_id: Optional[str] = None
    type: str = "reassigned"
    deadhead_miles: float = 0.0
    deadhead_minutes: float = 0.0
    overtime_minutes: float = 0.0
    miles_delta: float = 0.0
    delay_minutes: float = 0.0
    uses_emergency_rest: bool = False
    feasible_hard: bool = True
    est_cost: float

class PlanCandidatesResponse(BaseModel):
    weekday: str
    trip_minutes: float
    trip_miles: float
    candidates: List[CandidateOut]

class PlanSolveResponse(BaseModel):
    weekday: str
    trip_minutes: float
    trip_miles: float
    objective_value: float
    assignments: List[Dict[str, Any]]
    details: Dict[str, Any]
    candidates_considered: int


# =========================
# Small container for matrices
# =========================
@dataclass
class matrices:
    dist: np.ndarray    # miles
    time: np.ndarray    # minutes
    loc2idx: Dict[str, int]


# =========================
# Router factory
# =========================
def create_router(
    get_data: Callable[[], Optional[Dict[str, Any]]],
    get_cost_config: Callable[[], Dict[str, float]],
    get_cuopt_url: Callable[[], str],  # kept to preserve signature (unused here)
) -> APIRouter:

    router = APIRouter(prefix="/plan", tags=["Plan"])
    PRIORITY_MAP = _load_priority_map()
    SLA_WINDOWS  = _load_sla_windows()

    # -------- location meta (name -> lat/lon/postcode) --------
    def _try_load_locations_meta() -> Dict[str, Dict[str, Any]]:
        """
        Hardwired reader for data/private/active/locations.csv with columns:
        From Site, Mapped Name A, Mapped Postcode A, Lat_A, Long_A

        Produces LOC_META keyed by uppercased 'Mapped Name A'.
        Also adds an alias entry keyed by uppercased 'From Site' (if present).
        """
        import csv
        meta: Dict[str, Dict[str, Any]] = {}

        p = _dataset_dir() / "locations.csv"
        if not p.exists():
            return meta

        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Mapped Name A") or "").strip()
                lat_s = (row.get("Lat_A") or "").strip()
                lon_s = (row.get("Long_A") or "").strip()
                pc    = (row.get("Mapped Postcode A") or "").strip()
                if not name or not lat_s or not lon_s:
                    continue
                try:
                    lat = float(lat_s); lon = float(lon_s)
                except Exception:
                    continue

                key = name.upper()
                rec = {"lat": lat, "lon": lon}
                if pc:
                    rec["postcode"] = pc
                meta[key] = rec

                # Optional alias by “From Site” (e.g., "ABERDEEN MC") → same record
                alias = (row.get("From Site") or "").strip()
                if alias:
                    meta[alias.upper()] = rec

        return meta

    LOC_META = _try_load_locations_meta()

    # -------- data helpers (closure) --------
    def _ensure_ready() -> Tuple[Dict[str, Any], matrices]:
        DATA = get_data()
        if DATA is None:
            raise HTTPException(status_code=503, detail="Private data not loaded. Upload/build and POST /admin/reload.")
        return DATA, matrices(
            dist=DATA["distance"], time=DATA["time"], loc2idx=DATA["location_to_index"]
        )

    def _weekday_from_local(s: str) -> str:
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.strptime(s, "%Y-%m-%dT%H:%M")
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid when_local '{s}'. Use ISO like 2025-08-18T21:30")
        dt = dt.replace(tzinfo=ZoneInfo("Europe/London")) if dt.tzinfo is None else dt.astimezone(ZoneInfo("Europe/London"))
        return WEEKDAYS[dt.weekday()]

    def _idx_of(name: str, loc2idx: Dict[str,int]) -> int:
        key = str(name).upper().strip()
        if key not in loc2idx:
            raise HTTPException(status_code=400, detail=f"Unknown location '{name}'. Add/update locations and rebuild.")
        return int(loc2idx[key])

    def _minutes_between(i: int, j: int, tmat: np.ndarray) -> float:
        return max(0.0, float(tmat[i, j]))

    def _miles_between(i: int, j: int, dist: np.ndarray) -> float:
        return max(0.0, float(dist[i, j]))

    def _costs(cfg: Dict[str, float]) -> Tuple[float, float, float, float, float]:
        deadhead = cfg.get("deadhead_cost_per_mile", cfg.get("deadhead_cost", 1.0))
        overtime = cfg.get("overtime_cost_per_minute", cfg.get("overtime_cost", 1.0))
        admin    = cfg.get("reassignment_admin_cost", 10.0)
        delay    = cfg.get("delay_cost_per_minute", 10.0)
        outsource_per_mile = cfg.get("outsourcing_per_mile", cfg.get("outsourcing_cost_per_mile", 2.0))
        return deadhead, overtime, admin, delay, outsource_per_mile

    def _max_duty_minutes(cfg: Dict[str, float]) -> int:
        val = cfg.get("max_duty_minutes", None)
        try:
            return int(val) if val is not None else 13 * 60
        except Exception:
            return 13 * 60
        
    def _row_flag_true(v) -> bool:
        # Accept 1/0, True/False, "1"/"0"
        if v is None: 
            return False
        try:
            return int(v) == 1
        except Exception:
            return str(v).strip().lower() in ("true", "t", "yes", "y")

    def _element_active_on_weekday(e: dict, weekday: str) -> bool:
        """Return True iff this element row is scheduled on the given weekday (Mon..Sun)."""
        # Preferred: explicit boolean column like e["Mon"], e["Tue"], etc.
        if weekday in e:
            return _row_flag_true(e.get(weekday))
        # Common fallback: an array/list of weekday strings in the element
        days_list = e.get("days")
        if isinstance(days_list, (list, tuple, set)):
            return weekday in {str(d).title()[:3] for d in days_list}
        # Another fallback: single string like "Mon" on the element
        wd = e.get("weekday")
        if isinstance(wd, str):
            return weekday == wd.title()[:3]
        return False

    def _condensed_first_last_ids(elements: List[Dict[str, Any]], loc2idx: Dict[str,int]) -> Tuple[Optional[int], Optional[int]]:
        if not elements:
            return None, None
        start_id: Optional[int] = None
        end_id: Optional[int] = None
        for e in elements:
            if start_id is None:
                if e.get("from_id") is not None:
                    start_id = int(e["from_id"])
                else:
                    nm = str(e.get("from", "")).upper().strip()
                    if nm in loc2idx: start_id = int(loc2idx[nm])
            if e.get("to_id") is not None:
                end_id = int(e["to_id"])
            else:
                nm2 = str(e.get("to", "")).upper().strip()
                if nm2 in loc2idx: end_id = int(loc2idx[nm2])
        return start_id, end_id

    # -------- geo helpers --------
    def _idx_to_name(idx: int, loc2idx: Dict[str,int]) -> Optional[str]:
        for k, v in loc2idx.items():
            if int(v) == int(idx):
                return k
        return None

    def _meta_for_idx(idx: int, loc2idx: Dict[str,int]) -> Optional[Dict[str,Any]]:
        nm = _idx_to_name(idx, loc2idx)
        return LOC_META.get(nm) if nm else None

    def _postcode_is_NI(pc: Optional[str]) -> bool:
        return bool(pc) and pc.upper().strip().startswith("BT")  # Northern Ireland

    def _same_island_by_idx(i: int, j: int, loc2idx: Dict[str,int]) -> Optional[bool]:
        mi = _meta_for_idx(i, loc2idx)
        mj = _meta_for_idx(j, loc2idx)
        if not (mi and mj):
            return None
        # Prefer postcode classification if present
        if ("postcode" in mi) or ("postcode" in mj):
            return _postcode_is_NI(mi.get("postcode")) == _postcode_is_NI(mj.get("postcode"))
        # Fallback heuristics (very coarse NI box)
        def _is_NI(m): return (m.get("lon", 0) < -5.4 and m.get("lat", 0) > 54.0)
        return _is_NI(mi) == _is_NI(mj)

    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 3958.7613
        from math import radians, sin, cos, asin, sqrt
        dphi = radians(lat2 - lat1)
        dlmb = radians(lon2 - lon1)
        phi1 = radians(lat1); phi2 = radians(lat2)
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
        return 2 * R * asin(sqrt(a))

    def _haversine_between_idx(i: int, j: int, loc2idx: Dict[str,int]) -> Optional[float]:
        mi = _meta_for_idx(i, loc2idx)
        mj = _meta_for_idx(j, loc2idx)
        if not (mi and mj and "lat" in mi and "lon" in mi and "lat" in mj and "lon" in mj):
            return None
        return _haversine_miles(mi["lat"], mi["lon"], mj["lat"], mj["lon"])

    def _passes_geo_filters(from_idx: int, to_idx: int, loc2idx: Dict[str,int]) -> bool:
        if ENFORCE_SAME_ISLAND:
            same = _same_island_by_idx(from_idx, to_idx, loc2idx)
            if same is False:
                return False
        if USE_HAVERSINE_DEADHEAD:
            h = _haversine_between_idx(from_idx, to_idx, loc2idx)
            if h is not None and h > HAV_MAX_DEADHEAD_ONE_WAY_MI:
                return False
        return True
    
        # --- Normalize element keys (Planz Code variants etc.) ---
    
    def _norm_key_lookup(e: Dict[str,Any], *candidates: str, default=None):
        for k in candidates:
            if k in e: return e[k]
        # case-insensitive / space tolerant
        lower = {str(k).lower().replace(" ","_"): v for k,v in e.items()}
        for k in candidates:
            kk = k.lower().replace(" ","_")
            if kk in lower: return lower[kk]
        return default

    EMPTY_PLANZ_TOKENS = {"EMPTY", "TRAVEL_NO_DATA", "TRAVEL NO DATA"}

    def _is_empty_planz(e: Dict[str,Any]) -> bool:
        pc = _norm_key_lookup(e, "planz_code", "Planz Code", "planzCode", default="")
        pc = str(pc).upper()
        return any(tok in pc for tok in EMPTY_PLANZ_TOKENS)

    def _is_travel_leg(e: Dict[str,Any]) -> bool:
        # keep your existing flags but be permissive
        if e.get("is_travel") is True:
            return True
        et = str(e.get("element_type","")).upper()
        return ("TRAVEL" in et) or _is_empty_planz(e)

    def _same_loc(i: Optional[int], j: Optional[int]) -> bool:
        return (i is not None) and (j is not None) and (int(i) == int(j))

    # -------- find/cascade helpers --------
    def _find_leg_by_candidate_id(meta: Dict[str,Any], candidate_id: str) -> Optional[Dict[str,Any]]:
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

    def _build_trip_from_leg(e: Dict[str,Any], loc2idx: Dict[str,int], dist: np.ndarray, tmat: np.ndarray) -> Dict[str,Any]:
        start_loc = str(e.get("from","")).upper()
        end_loc   = str(e.get("to","")).upper()
        i = loc2idx.get(start_loc); j = loc2idx.get(end_loc)
        dur   = float(e.get("duration_min") or (tmat[i, j] if i is not None and j is not None else 0.0))
        miles = float(e.get("miles")        or (dist[i, j] if i is not None and j is not None else 0.0))
        return {
            "id": f"CASCADE:{start_loc}->{end_loc}@{int(e.get('start_min',0))}",
            "start_location": start_loc,
            "end_location": end_loc,
            "duration_minutes": dur,
            "trip_miles": miles,
        }

    # =========================
    # Candidate generation (single source of truth)
    # =========================

    def _generate_candidates_core(req: PlanRequest) -> Tuple[str, float, float, List[CandidateOut]]:
        # ---- tiny local helpers ----
        def _is_travel_leg(e: Dict[str, Any]) -> bool:
            if e.get("is_travel") is True:
                return True
            et = str(e.get("element_type", "")).strip().upper()
            return et.startswith("TRAVEL") or ("LEG" in et and "TRAVEL" in et)

        def _is_empty_planz(e: Dict[str, Any]) -> bool:
            pc = str(e.get("planz_code", e.get("Planz Code", ""))).strip().lower()
            if "empty" in pc or "travel_no_data" in pc:
                return True
            return bool(e.get("is_empty", False))

        def _same_loc(a: Optional[int], b: Optional[int]) -> bool:
            try:
                return (a is not None) and (b is not None) and (int(a) == int(b))
            except Exception:
                return False

        def _row_flag_true(v) -> bool:
            if v is None:
                return False
            try:
                return int(v) == 1
            except Exception:
                return str(v).strip().lower() in ("true", "t", "yes", "y")

        def _element_active_on_weekday(e: dict, weekday: str) -> bool:
            # Prefer explicit Mon..Sun flags on each row
            if weekday in e:
                return _row_flag_true(e.get(weekday))
            # Fallbacks if your driver_state rows are different
            days_list = e.get("days")
            if isinstance(days_list, (list, tuple, set)):
                return weekday in {str(d).title()[:3] for d in days_list}
            wd = e.get("weekday")
            if isinstance(wd, str):
                return weekday == wd.title()[:3]
            return False

        # ---- 1) SLA window & geo toggles ----
        req_min  = _minute_of_day_local(req.when_local)
        sla      = SLA_WINDOWS.get(int(req.priority), {"early_min": 60, "late_min": 60})
        earliest = max(0, int(req_min) - int(sla["early_min"]))
        latest   = int(req_min) + int(sla["late_min"])

        same_island_required = bool(sla.get("enforce_same_island", ENFORCE_SAME_ISLAND))
        hav_max_dd_mi        = float(sla.get("hav_max_deadhead_one_way_mi", HAV_MAX_DEADHEAD_ONE_WAY_MI))

        # ---- 2) Matrices + indices ----
        DATA, M   = _ensure_ready()
        weekday   = _weekday_from_local(req.when_local)  # "Mon".."Sun"
        start_idx = _idx_of(req.start_location, M.loc2idx)
        end_idx   = _idx_of(req.end_location,   M.loc2idx)

        # ---- 3) Trip metrics ----
        trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else _minutes_between(start_idx, end_idx, M.time)
        trip_miles   = float(req.trip_miles)   if req.trip_miles   is not None else _miles_between(start_idx, end_idx, M.dist)

        # ---- 4) Costs ----
        cfg = get_cost_config()
        deadhead_cost_per_mile, overtime_cost_per_minute, admin_cost, _delay_cost, _out_per_mile = _costs(cfg)
        max_duty_min = _max_duty_minutes(cfg)

        candidates: List[CandidateOut] = []

        # ---------- common geo check (haversine/same-island) ----------
        def _passes_geo_filters(from_idx: Optional[int], to_idx: int) -> bool:
            if from_idx is None:
                return True  # no opinion
            if same_island_required:
                si = _same_island_by_idx(from_idx, to_idx, M.loc2idx)
                if si is False:  # only block if we know it's different islands
                    return False
            if USE_HAVERSINE_DEADHEAD:
                hdd = _haversine_between_idx(from_idx, to_idx, M.loc2idx)
                if hdd is not None and hdd > hav_max_dd_mi:
                    return False
            return True

        # ---------- Tier 0: take an existing EMPTY A->B ----------
        def _tier0_empty_A_to_B() -> List[CandidateOut]:
            ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
            out: List[CandidateOut] = []
            for duty_id, meta in drivers.items():
                # filter to elements active on this weekday
                elements_all = meta.get("elements", []) or []
                elements = [e for e in elements_all if _element_active_on_weekday(e, weekday)]
                if not elements:
                    continue

                # soft geo gate using start-of-shift if available
                first_id, _ = _condensed_first_last_ids(elements, M.loc2idx)
                if not _passes_geo_filters(first_id, start_idx):
                    continue

                for e in elements:
                    if not _is_travel_leg(e):  continue
                    if not _is_empty_planz(e): continue
                    if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
                        continue

                    s = e.get("start_min"); en = e.get("end_min")
                    if s is None or en is None:
                        continue

                    ok_time = (
                        (req.mode == "depart_after"  and earliest <= s <= latest) or
                        (req.mode == "arrive_before" and earliest <= en <= latest)
                    )
                    if not ok_time:
                        continue

                    out.append(CandidateOut(
                        candidate_id=f"{duty_id}::take_empty@{int(s)}",
                        driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
                        deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                        miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
                        feasible_hard=True, est_cost=float(admin_cost),
                    ))
            out.sort(key=lambda c: (int(c.candidate_id.split("@")[1]), c.driver_id))
            return out

        # ---------- Tier 1: swap a departure from A (A->X) ----------
        def _tier1_swap_from_A() -> List[CandidateOut]:
            ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
            out: List[CandidateOut] = []
            for duty_id, meta in drivers.items():
                elements_all = meta.get("elements", []) or []
                elements = [e for e in elements_all if _element_active_on_weekday(e, weekday)]
                if not elements:
                    continue

                # soft geo gate using start-of-shift if available
                first_id, _ = _condensed_first_last_ids(elements, M.loc2idx)
                if not _passes_geo_filters(first_id, start_idx):
                    continue

                for e in elements:
                    if not _is_travel_leg(e): continue
                    if not _same_loc(e.get("from_id"), start_idx): continue

                    s = e.get("start_min"); en = e.get("end_min")
                    if s is None or en is None:
                        continue

                    ok_time = (
                        (req.mode == "depart_after"  and earliest <= s <= latest) or
                        (req.mode == "arrive_before" and earliest <= en <= latest)
                    )
                    if not ok_time:
                        continue

                    # do not displace higher-priority work
                    if int(e.get("priority", 3)) < int(req.priority):
                        continue

                    out.append(CandidateOut(
                        candidate_id=f"{duty_id}::swap_from_A@{int(s)}",
                        driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
                        deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                        miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
                        feasible_hard=True, est_cost=float(admin_cost),
                    ))
            out.sort(key=lambda c: int(c.candidate_id.split("@")[1]))
            return out

        # Add Tier 0/1 first (these are the “human” first checks)
        candidates.extend(_tier0_empty_A_to_B())
        candidates.extend(_tier1_swap_from_A())

        # ---------- Per-driver slack / exact A->B swap / append ----------
        ds = DATA["driver_states"] or {}
        drivers = ds["drivers"] if isinstance(ds, dict) and "drivers" in ds else ds

        for duty_id, meta in drivers.items():
            # filter elements to today
            elements_all = meta.get("elements", []) or []
            elements = [e for e in elements_all if _element_active_on_weekday(e, weekday)]
            if not elements:
                continue

            # duty baseline (today only)
            daily_windows = meta.get("daily_windows", {})
            if weekday in daily_windows:
                win = daily_windows[weekday]
                start_min = int(win.get("start_min", 0))
                end_min   = int(win.get("end_min", 0))
                orig_length = max(0, end_min - start_min)
            else:
                durations = [int(e["duration_min"]) for e in elements if e.get("duration_min") is not None]
                orig_length = int(sum(durations)) if durations else 8 * 60

            first_id, last_id = _condensed_first_last_ids(elements, M.loc2idx)

            # ---- Slack (AS DIRECTED) ----
            # soft geo gate using start-of-shift if available
            if not _passes_geo_filters(first_id, start_idx):
                pass  # still allow specific slack loc checks below
            for e in elements:
                et = str(e.get("element_type", "")).upper()
                if "AS DIRECTED" not in et:
                    continue

                # Resolve slack location id
                loc_id = e.get("from_id") or e.get("to_id")
                if loc_id is None:
                    nm_from = str(e.get("from", "")).upper().strip()
                    nm_to   = str(e.get("to", "")).upper().strip()
                    if nm_from in M.loc2idx:   loc_id = M.loc2idx[nm_from]
                    elif nm_to in M.loc2idx:   loc_id = M.loc2idx[nm_to]
                if loc_id is None:
                    continue
                loc_id = int(loc_id)

                e_start = e.get("start_min"); e_dur = e.get("duration_min")
                if e_start is None or e_dur is None:
                    continue
                e_end = e_start + e_dur

                fits_time = not (
                    (req.mode == "depart_after"  and (e_end < earliest or e_start > latest)) or
                    (req.mode == "arrive_before" and (e_end < earliest or e_start > req_min))
                )
                if not fits_time:
                    continue

                # minute budget inside slack
                dd_to_min   = _minutes_between(loc_id, start_idx, M.time)
                dd_back_min = _minutes_between(end_idx, loc_id, M.time)
                budget      = dd_to_min + trip_minutes + dd_back_min
                if budget > e_dur:
                    continue

                # Geo guards BEFORE miles
                if not _passes_geo_filters(loc_id, start_idx):
                    continue

                # miles (now safe to compute/use)
                dd_mi = _miles_between(loc_id, start_idx, M.dist) + _miles_between(end_idx, loc_id, M.dist)
                est_cost = (dd_mi * deadhead_cost_per_mile) + admin_cost

                candidates.append(CandidateOut(
                    candidate_id=f"{duty_id}::slack@{loc_id}",
                    driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
                    deadhead_miles=float(dd_mi),
                    deadhead_minutes=float(budget - trip_minutes),
                    overtime_minutes=0.0,             # fits inside slack
                    miles_delta=float(dd_mi + trip_miles),
                    delay_minutes=0.0,
                    uses_emergency_rest=False,
                    feasible_hard=True,
                    est_cost=float(est_cost),
                ))

            # ---- Exact A->B swap (equal/lower priority) ----
            # soft geo gate using start-of-shift if available
            if not _passes_geo_filters(first_id, start_idx):
                pass
            for e in elements:
                if not _is_travel_leg(e):
                    continue
                if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
                    continue

                e_start = e.get("start_min"); e_end = e.get("end_min")
                if e_start is None or e_end is None:
                    continue

                ok_time = (
                    (req.mode == "depart_after"  and (e_start >= earliest) and (e_start <= latest)) or
                    (req.mode == "arrive_before" and (e_end   <= latest)   and (e_end   >= earliest))
                )
                if not ok_time:
                    continue

                # don't displace higher priority
                if int(e.get("priority", 3)) < int(req.priority):
                    continue

                if not _passes_geo_filters(first_id, start_idx):
                    continue

                candidates.append(CandidateOut(
                    candidate_id=f"{duty_id}::swap_leg@{int(e_start)}",
                    driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=True, est_cost=float(admin_cost),
                ))

            # ---- Append after last if duty cap allows ----
            if last_id is not None:
                dd_min   = _minutes_between(last_id, start_idx, M.time)
                dd_mi    = _miles_between(last_id, start_idx, M.dist)
                added    = dd_min + trip_minutes
                new_len  = orig_length + added
                feasible = (new_len <= max_duty_min) and _passes_geo_filters(last_id, start_idx)

                est_cost = (dd_mi * deadhead_cost_per_mile) + (added * overtime_cost_per_minute) + admin_cost
                candidates.append(CandidateOut(
                    candidate_id=f"{duty_id}::append",
                    driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
                    deadhead_miles=float(dd_mi),
                    deadhead_minutes=float(dd_min),
                    overtime_minutes=float(added),
                    miles_delta=float(dd_mi + trip_miles),
                    delay_minutes=0.0,
                    uses_emergency_rest=False,
                    feasible_hard=bool(feasible),
                    est_cost=float(est_cost),
                ))

        # ---- ranking + cap ----
        def _rank_key(c: CandidateOut):
            cid = c.candidate_id
            if "::take_empty@" in cid:   t = 0
            elif "::swap_leg@" in cid:   t = 1
            elif "::swap_from_A@" in cid:t = 2
            elif "::slack@" in cid:      t = 3
            else:                        t = 4
            return (not c.feasible_hard, t, c.est_cost)

        candidates.sort(key=_rank_key)
        if req.top_n and req.top_n > 0:
            candidates = candidates[:req.top_n]

        return weekday, float(trip_minutes), float(trip_miles), candidates


    # def _generate_candidates_core(req: PlanRequest) -> Tuple[str, float, float, List[CandidateOut]]:
    #     # ---- tiny local helpers (no external dependencies) ----
    #     def _is_travel_leg(e: Dict[str, Any]) -> bool:
    #         if e.get("is_travel") is True:
    #             return True
    #         et = str(e.get("element_type", "")).strip().upper()
    #         return et.startswith("TRAVEL") or ("LEG" in et and "TRAVEL" in et)

    #     def _is_empty_planz(e: Dict[str, Any]) -> bool:
    #         pc = str(e.get("planz_code", e.get("Planz Code", ""))).strip().lower()
    #         if "empty" in pc or "travel_no_data" in pc:
    #             return True
    #         return bool(e.get("is_empty", False))

    #     def _same_loc(a: Optional[int], b: Optional[int]) -> bool:
    #         try:
    #             return (a is not None) and (b is not None) and (int(a) == int(b))
    #         except Exception:
    #             return False

    #     # ---- 1) SLA window & geo toggles (locals only) ----
    #     req_min  = _minute_of_day_local(req.when_local)
    #     sla      = SLA_WINDOWS.get(int(req.priority), {"early_min": 60, "late_min": 60})
    #     earliest = max(0, req_min - int(sla["early_min"]))
    #     latest   = req_min + int(sla["late_min"])

    #     same_island_required = bool(sla.get("enforce_same_island", ENFORCE_SAME_ISLAND))
    #     hav_max_dd_mi        = float(sla.get("hav_max_deadhead_one_way_mi", HAV_MAX_DEADHEAD_ONE_WAY_MI))

    #     # ---- 2) Matrices + indices ----
    #     DATA, M   = _ensure_ready()
    #     weekday   = _weekday_from_local(req.when_local)
    #     start_idx = _idx_of(req.start_location, M.loc2idx)
    #     end_idx   = _idx_of(req.end_location,   M.loc2idx)

    #     # ---- 3) Trip metrics ----
    #     trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else _minutes_between(start_idx, end_idx, M.time)
    #     trip_miles   = float(req.trip_miles)   if req.trip_miles   is not None else _miles_between(start_idx, end_idx, M.dist)

    #     # ---- 4) Costs ----
    #     cfg = get_cost_config()
    #     deadhead_cost_per_mile, overtime_cost_per_minute, admin_cost, _delay_cost, _out_per_mile = _costs(cfg)
    #     max_duty_min = _max_duty_minutes(cfg)

    #     candidates: List[CandidateOut] = []

    #     # ---------- Tier 0: take an existing EMPTY A->B ----------
    #     def _tier0_empty_A_to_B() -> List[CandidateOut]:
    #         ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
    #         out: List[CandidateOut] = []
    #         for duty_id, meta in drivers.items():
    #             if (set(meta.get("days", [])) or set(WEEKDAYS)) and weekday not in (set(meta.get("days", [])) or set(WEEKDAYS)):
    #                 continue
    #             for e in meta.get("elements", []):
    #                 if not _is_travel_leg(e):  continue
    #                 if not _is_empty_planz(e): continue
    #                 if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
    #                     continue
    #                 s = e.get("start_min"); en = e.get("end_min")
    #                 if s is None or en is None:
    #                     continue
    #                 ok_time = (
    #                     (req.mode == "depart_after"  and earliest <= s <= latest) or
    #                     (req.mode == "arrive_before" and earliest <= en <= latest)
    #                 )
    #                 if not ok_time:
    #                     continue
    #                 out.append(CandidateOut(
    #                     candidate_id=f"{duty_id}::take_empty@{int(s)}",
    #                     driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
    #                     deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
    #                     miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
    #                     feasible_hard=True, est_cost=float(admin_cost),
    #                 ))
    #         out.sort(key=lambda c: (int(c.candidate_id.split("@")[1]), c.driver_id))
    #         return out

    #     # ---------- Tier 1: swap a departure from A (A->X) of equal/lower priority ----------
    #     def _tier1_swap_from_A() -> List[CandidateOut]:
    #         ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
    #         out: List[CandidateOut] = []
    #         for duty_id, meta in drivers.items():
    #             active_days = set(meta.get("days", [])) or set(WEEKDAYS)
    #             if weekday not in active_days:
    #                 continue
    #             for e in meta.get("elements", []):
    #                 if not _is_travel_leg(e):                    continue
    #                 if not _same_loc(e.get("from_id"), start_idx): continue
    #                 s = e.get("start_min"); en = e.get("end_min")
    #                 if s is None or en is None:
    #                     continue
    #                 ok_time = (
    #                     (req.mode == "depart_after"  and earliest <= s <= latest) or
    #                     (req.mode == "arrive_before" and earliest <= en <= latest)
    #                 )
    #                 if not ok_time:
    #                     continue
    #                 if int(e.get("priority", 3)) < int(req.priority):
    #                     continue
    #                 # soft home-geo sanity
    #                 first_id, _ = _condensed_first_last_ids(meta.get("elements", []), M.loc2idx)
    #                 if first_id is not None:
    #                     if same_island_required:
    #                         same = _same_island_by_idx(first_id, start_idx, M.loc2idx)
    #                         if same is not True:
    #                             continue
    #                     hdd_home = _haversine_between_idx(first_id, start_idx, M.loc2idx)
    #                     if hdd_home is not None and hdd_home > hav_max_dd_mi:
    #                         continue
    #                 out.append(CandidateOut(
    #                     candidate_id=f"{duty_id}::swap_from_A@{int(s)}",
    #                     driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
    #                     deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
    #                     miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
    #                     feasible_hard=True, est_cost=float(admin_cost),
    #                 ))
    #         out.sort(key=lambda c: int(c.candidate_id.split("@")[1]))
    #         return out

    #     # add Tier 0/1
    #     candidates.extend(_tier0_empty_A_to_B())
    #     candidates.extend(_tier1_swap_from_A())

    #     # ---------- Per-driver slack / exact A->B swap / append ----------
    #     ds = DATA["driver_states"] or {}
    #     drivers = ds["drivers"] if isinstance(ds, dict) and "drivers" in ds else ds

    #     for duty_id, meta in drivers.items():
    #         active_days = set(meta.get("days", [])) or set(WEEKDAYS)
    #         if weekday not in active_days:
    #             continue

    #         # duty baseline
    #         daily_windows = meta.get("daily_windows", {})
    #         if weekday in daily_windows:
    #             win = daily_windows[weekday]
    #             start_min = int(win.get("start_min", 0)); end_min = int(win.get("end_min", 0))
    #             orig_length = max(0, end_min - start_min)
    #         else:
    #             durations = [int(e.get("duration_min")) for e in meta.get("elements", []) if e.get("duration_min") is not None]
    #             orig_length = int(sum(durations)) if durations else 8 * 60

    #         first_id, last_id = _condensed_first_last_ids(meta.get("elements", []), M.loc2idx)

    #         # ---- Slack (AS DIRECTED) ----
    #         for e in meta.get("elements", []):
    #             et = str(e.get("element_type", "")).upper()
    #             if "AS DIRECTED" not in et:
    #                 continue

    #             loc_id = e.get("from_id") or e.get("to_id")
    #             if loc_id is None:
    #                 nm_from = str(e.get("from", "")).upper().strip()
    #                 nm_to   = str(e.get("to", "")).upper().strip()
    #                 if nm_from in M.loc2idx:   loc_id = M.loc2idx[nm_from]
    #                 elif nm_to in M.loc2idx:   loc_id = M.loc2idx[nm_to]
    #             if loc_id is None:
    #                 continue
    #             loc_id = int(loc_id)

    #             e_start = e.get("start_min"); e_dur = e.get("duration_min")
    #             if e_start is None or e_dur is None:
    #                 continue
    #             e_end = e_start + e_dur

    #             fits_time = not (
    #                 (req.mode == "depart_after"  and (e_end < req_min or e_start > latest)) or
    #                 (req.mode == "arrive_before" and (e_end < earliest or e_start > req_min))
    #             )
    #             if not fits_time:
    #                 continue

    #             dd_to_min   = _minutes_between(loc_id, start_idx, M.time)
    #             dd_back_min = _minutes_between(end_idx, loc_id, M.time)
    #             budget      = dd_to_min + trip_minutes + dd_back_min
    #             if budget > e_dur:
    #                 continue

    #             # geo guards before miles
    #             if same_island_required:
    #                 same = _same_island_by_idx(first_id, start_idx, M.loc2idx)
    #                 if same is not True:
    #                     continue
    #             hdd = _haversine_between_idx(loc_id, start_idx, M.loc2idx)
    #             if hdd is not None and hdd > hav_max_dd_mi:
    #                 continue

    #             dd_mi   = _miles_between(loc_id, start_idx, M.dist) + _miles_between(end_idx, loc_id, M.dist)
    #             est_cost = (dd_mi * deadhead_cost_per_mile) + admin_cost

    #             candidates.append(CandidateOut(
    #                 candidate_id=f"{duty_id}::slack@{loc_id}",
    #                 driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
    #                 deadhead_miles=float(dd_mi),
    #                 deadhead_minutes=float(budget - trip_minutes),
    #                 overtime_minutes=0.0,
    #                 miles_delta=float(dd_mi + trip_miles),
    #                 delay_minutes=0.0,
    #                 uses_emergency_rest=False,
    #                 feasible_hard=True,
    #                 est_cost=float(est_cost),
    #             ))

    #         # ---- Exact A->B swap (equal/lower priority) ----
    #         for e in meta.get("elements", []):
    #             if not _is_travel_leg(e):
    #                 continue
    #             if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
    #                 continue

    #             e_start = e.get("start_min"); e_end = e.get("end_min")
    #             if e_start is None or e_end is None:
    #                 continue

    #             ok_time = (
    #                 (req.mode == "depart_after"  and (e_start >= req_min) and (e_start <= latest)) or
    #                 (req.mode == "arrive_before" and (e_end   <= req_min) and (e_end   >= earliest))
    #             )
    #             if not ok_time:
    #                 continue

    #             if int(e.get("priority", 3)) < int(req.priority):
    #                 continue

    #             if first_id is not None:
    #                 if same_island_required:
    #                     same = _same_island_by_idx(first_id, start_idx, M.loc2idx)
    #                     if same is not True:
    #                         continue
    #                 hdd_home = _haversine_between_idx(first_id, start_idx, M.loc2idx)
    #                 if hdd_home is not None and hdd_home > hav_max_dd_mi:
    #                     continue

    #             candidates.append(CandidateOut(
    #                 candidate_id=f"{duty_id}::swap_leg@{e_start}",
    #                 driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
    #                 deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
    #                 miles_delta=0.0, delay_minutes=0.0, uses_emergency_rest=False,
    #                 feasible_hard=True, est_cost=float(admin_cost),
    #             ))

    #         # ---- Append after last if duty cap allows ----
    #         if last_id is not None:
    #             dd_min   = _minutes_between(last_id, start_idx, M.time)
    #             dd_mi    = _miles_between(last_id, start_idx, M.dist)
    #             added    = dd_min + trip_minutes
    #             new_len  = orig_length + added
    #             feasible = new_len <= max_duty_min
    #             if same_island_required:
    #                 same = _same_island_by_idx(last_id, start_idx, M.loc2idx)
    #                 if same is False:
    #                     feasible = False
    #             hdd_home = _haversine_between_idx(last_id, start_idx, M.loc2idx)
    #             if hdd_home is not None and hdd_home > hav_max_dd_mi:
    #                 feasible = False

    #             est_cost = (dd_mi * deadhead_cost_per_mile) + (added * overtime_cost_per_minute) + admin_cost
    #             candidates.append(CandidateOut(
    #                 candidate_id=f"{duty_id}::append",
    #                 driver_id=str(duty_id), route_id=str(duty_id), type="reassigned",
    #                 deadhead_miles=float(dd_mi),
    #                 deadhead_minutes=float(dd_min),
    #                 overtime_minutes=float(added),
    #                 miles_delta=float(dd_mi + trip_miles),
    #                 delay_minutes=0.0,
    #                 uses_emergency_rest=False,
    #                 feasible_hard=bool(feasible),
    #                 est_cost=float(est_cost),
    #             ))

    #     # ---- ranking + cap ----
    #     def _rank_key(c: CandidateOut):
    #         cid = c.candidate_id
    #         if "::take_empty@" in cid:   t = 0
    #         elif "::swap_leg@" in cid:   t = 1
    #         elif "::swap_from_A@" in cid:t = 2
    #         elif "::slack@" in cid:      t = 3
    #         else:                        t = 4
    #         return (not c.feasible_hard, t, c.est_cost)

    #     candidates.sort(key=_rank_key)
    #     if req.top_n and req.top_n > 0:
    #         candidates = candidates[:req.top_n]

    #     return weekday, float(trip_minutes), float(trip_miles), candidates

    # =========================
    # Endpoints
    # =========================
    @router.post("/candidates", response_model=PlanCandidatesResponse)
    def plan_candidates(req: PlanRequest):
        weekday, trip_minutes, trip_miles, cands = _generate_candidates_core(req)
        return PlanCandidatesResponse(
            weekday=weekday, trip_minutes=trip_minutes, trip_miles=trip_miles, candidates=cands
        )

    @router.post("/solve_cascades", response_model=PlanSolveCascadeResponse)
    def plan_and_solve_cascades(req: PlanSolveCascadeRequest, request: Request):
        DATA, M = _ensure_ready()
        cfg = get_cost_config()
        deadhead_cost_per_mile, overtime_cpm, admin_cost, _delay_cpm, out_per_mile = _costs(cfg)

        i = _idx_of(req.start_location, M.loc2idx)
        j = _idx_of(req.end_location,   M.loc2idx)
        trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else _minutes_between(i, j, M.time)
        trip_miles   = float(req.trip_miles)   if req.trip_miles   is not None else _miles_between(i, j, M.dist)

        root_trip = {
            "id": f"NEW:{req.start_location}->{req.end_location}@{req.when_local}",
            "start_location": req.start_location,
            "end_location": req.end_location,
            "duration_minutes": trip_minutes,
            "trip_miles": trip_miles,
        }

        cascades: List[Dict[str,Any]] = []
        all_assignments: List[AssignmentOut] = []
        total_obj = 0.0
        total_candidates_seen = 0
        affected_drivers: set[str] = set()
        visited_cascade_keys: set[Tuple[str,int]] = set()  # (driver_id, start_min) to avoid dup loops

        queue: List[Tuple[Dict[str,Any], int, int]] = [(root_trip, req.priority, 0)]

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
            pc = plan_candidates(cand_req)
            total_candidates_seen += len(pc.candidates)

            chosen = pc.candidates[0] if pc.candidates else None
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
                bd["deadhead"] = float(chosen.deadhead_miles * deadhead_cost_per_mile)
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

            # Cascade if we stole an equal/lower-priority leg
            if depth < req.max_cascades and "swap_leg@" in chosen.candidate_id:
                ds = DATA["driver_states"]; drivers = ds["drivers"] if "drivers" in ds else ds
                m = drivers.get(chosen.driver_id, {})
                leg = _find_leg_by_candidate_id(m, chosen.candidate_id)
                if leg:
                    leg_pri = int(leg.get("priority", 3))
                    if leg_pri >= prio:
                        displaced_trip = _build_trip_from_leg(leg, M.loc2idx, M.dist, M.time)
                        key = (chosen.driver_id, int(leg.get("start_min", -1)))
                        if key not in visited_cascade_keys:
                            visited_cascade_keys.add(key)
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
            weekday=_weekday_from_local(req.when_local),
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
        """
        Simple wrapper: run candidate generation and pick the best candidate (no cascades).
        Returns an outsource if no candidates are available.
        """
        try:
            DATA, M = _ensure_ready()
            cfg = get_cost_config()
            deadhead_cost_per_mile, overtime_cpm, admin_cost, _delay_cpm, out_per_mile = _costs(cfg)

            weekday, trip_minutes, trip_miles, cands = _generate_candidates_core(req)

            if not cands:
                base = float(cfg.get("outsourcing_base_cost", 200.0))
                cost = base + trip_miles * out_per_mile
                return {
                    "weekday": weekday,
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
                bd["deadhead"] = float(chosen.deadhead_miles * deadhead_cost_per_mile)
            if chosen.overtime_minutes:
                bd["overtime"] = float(chosen.overtime_minutes * overtime_cpm)
            cost = float(sum(bd.values()))

            return {
                "weekday": weekday,
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

        except Exception as e:
            tb = traceback.format_exc(limit=8)
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb[:4000]})

    # -------- UI helpers --------
    @router.get("/locations")
    def list_locations():
        # Prefer live in-memory mapping; else try CSV
        try:
            _DATA, M = _ensure_ready()
            names = sorted({str(k).strip() for k in M.loc2idx.keys() if str(k).strip()})
            return {"names": names, "count": len(names), "source": "memory"}
        except HTTPException:
            pass

        import pandas as pd
        d = _dataset_dir()
        for fn in ("location_index.csv", "locations.csv"):
            p = d / fn
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    for col in ("name", "NAME", "site_name", "Site", "site"):
                        if col in df.columns:
                            names = sorted({str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()})
                            return {"names": names, "count": len(names), "source": str(p)}
                except Exception as e:
                    return {"names": [], "count": 0, "source": str(p), "error": f"read-failed: {e}"}
        return {"names": [], "count": 0, "source": "none"}

    @router.get("/priority_map")
    def get_priority_map():
        return _load_priority_map()

    return router
