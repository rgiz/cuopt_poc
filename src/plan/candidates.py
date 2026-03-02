from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from .models import PlanRequest, CandidateOut, WEEKDAYS
from .config import (
    ENFORCE_SAME_ISLAND,
    USE_HAVERSINE_DEADHEAD,
    HAV_MAX_DEADHEAD_ONE_WAY_MI,
    ENABLE_AS_DIRECTED_INSERTION,
    ENABLE_PARTIAL_OVERLAP_INSERTION,
    ENABLE_NEARBY_DEPOT_SUBSTITUTION,
    ENABLE_STRICT_LEGAL_CONSTRAINTS,
)
from .geo import same_island_by_meta, haversine_between_idx

# Constants for logistics timing
LOADING_TIME_MINUTES = 30
OFFLOADING_TIME_MINUTES = 20
TOTAL_SERVICE_TIME_MINUTES = LOADING_TIME_MINUTES + OFFLOADING_TIME_MINUTES
GEOGRAPHIC_PROXIMITY_MILES = 50  # Default - make configurable later

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

def get_driver_home_base(driver_meta: Dict[str, Any], loc2idx: Dict[str, int]) -> Optional[int]:
    """Extract driver's home base center_id from metadata."""
    # Try home_center_id first
    home_id = driver_meta.get("home_center_id")
    if home_id is not None and isinstance(home_id, (int, str)):
        try:
            return int(home_id)
        except (ValueError, TypeError):
            pass
    
    # Fallback: look for start_loc in elements or driver meta
    home_loc = driver_meta.get("home_loc")
    if home_loc and isinstance(home_loc, str):
        return loc2idx.get(home_loc.upper().strip())

    start_loc = driver_meta.get("start_loc")
    if start_loc and isinstance(start_loc, str):
        return loc2idx.get(start_loc.upper().strip())
    
    # Last resort: find START FACILITY in elements
    elements = driver_meta.get("elements", [])
    for e in elements:
        if "START FACILITY" in str(e.get("element_type", "")).upper():
            from_loc = str(e.get("from", "")).upper().strip()
            if from_loc in loc2idx:
                return loc2idx[from_loc]
    
    return None

def check_driver_duty_window_overlap(
    driver_meta: Dict[str, Any], 
    weekday: str, 
    req_start_min: int, 
    total_service_minutes: float
) -> bool:
    """
    Quick check: Does driver have enough time in their duty window for the full service?
    Includes A->B travel + loading/offloading time (50 mins total service overhead).
    """
    # Get daily window for this weekday
    daily_windows = driver_meta.get("daily_windows", {})
    if weekday in daily_windows:
        window = daily_windows[weekday]
        duty_start = int(window.get("start_min", 0))
        duty_end = int(window.get("end_min", 1440))
    else:
        duty_start = driver_meta.get("start_min")
        duty_end = driver_meta.get("end_min")

        if duty_start is None or duty_end is None:
            elements = [e for e in (driver_meta.get("elements", []) or []) if element_active_on_weekday(e, weekday)]
            mins = [e.get("start_min") for e in elements if e.get("start_min") is not None]
            maxs = [e.get("end_min") for e in elements if e.get("end_min") is not None]
            if mins and maxs:
                duty_start = min(mins)
                duty_end = max(maxs)

        if duty_start is None or duty_end is None:
            return True

        duty_start = int(duty_start)
        duty_end = int(duty_end)
    
    # Handle cross-midnight duties
    if duty_end < duty_start:
        duty_end += 24 * 60
    
    # For candidate generation we allow overtime beyond duty end and evaluate feasibility later.
    # We only require the requested service to start after duty start.
    return duty_start <= req_start_min

def check_geographic_proximity(
    driver_home_base: int, 
    collection_point: int, 
    max_miles: float,
    loc2idx: Dict[str, int], 
    loc_meta: Dict[str, Any]
) -> bool:
    """
    Check if driver's home base is within max_miles of collection point.
    Uses haversine_between_idx which does matrix-first, haversine-fallback.
    """
    try:
        distance = haversine_between_idx(driver_home_base, collection_point, loc2idx, loc_meta)
        if distance is None:
            return True  # If we can't compute distance, don't exclude
        return distance <= max_miles
    except Exception:
        return True  # Don't exclude on calculation errors

def compute_full_logistics_cost(
    driver_meta: Dict[str, Any],
    driver_home_base: int,
    collection_point: int,
    delivery_point: int,
    trip_minutes: float,
    trip_miles: float,
    cost_cfg: Dict[str, float],
    Mtime: np.ndarray,
    Mdist: np.ndarray,
    loc2idx: Dict[str, int],
    loc_meta: Dict[str, Any]
) -> Tuple[float, float, float, float, bool]:
    """
    Compute full logistics chain cost and timing.
    Returns: (total_cost, total_minutes, total_miles, overtime_minutes, is_feasible)
    """
    # Cost components
    deadhead_cpm = cost_cfg.get("deadhead_cost_per_mile", 1.0)
    overtime_cpm = cost_cfg.get("overtime_cost_per_minute", 1.0)
    admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
    max_duty_minutes = int(cost_cfg.get("max_duty_minutes", 13*60))
    
    # 1. Deadhead to collection point
    try:
        deadhead_to_minutes = haversine_between_idx(driver_home_base, collection_point, loc2idx, loc_meta)
        deadhead_to_miles = haversine_between_idx(driver_home_base, collection_point, loc2idx, loc_meta)
        if deadhead_to_minutes is None:
            deadhead_to_minutes = minutes_between(driver_home_base, collection_point, Mtime)
            deadhead_to_miles = miles_between(driver_home_base, collection_point, Mdist)
    except Exception:
        deadhead_to_minutes = minutes_between(driver_home_base, collection_point, Mtime)
        deadhead_to_miles = miles_between(driver_home_base, collection_point, Mdist)
    
    # 2. Loading time: 30 minutes
    loading_minutes = LOADING_TIME_MINUTES
    
    # 3. Service leg A->B
    service_minutes = trip_minutes
    service_miles = trip_miles
    
    # 4. Offloading time: 20 minutes  
    offloading_minutes = OFFLOADING_TIME_MINUTES
    
    # 5. Return deadhead to home base
    try:
        deadhead_return_minutes = haversine_between_idx(delivery_point, driver_home_base, loc2idx, loc_meta)
        deadhead_return_miles = haversine_between_idx(delivery_point, driver_home_base, loc2idx, loc_meta)
        if deadhead_return_minutes is None:
            deadhead_return_minutes = minutes_between(delivery_point, driver_home_base, Mtime)
            deadhead_return_miles = miles_between(delivery_point, driver_home_base, Mdist)
    except Exception:
        deadhead_return_minutes = minutes_between(delivery_point, driver_home_base, Mtime)
        deadhead_return_miles = miles_between(delivery_point, driver_home_base, Mdist)
    
    # Total time and distance
    total_minutes = (deadhead_to_minutes + loading_minutes + service_minutes + 
                    offloading_minutes + deadhead_return_minutes)
    total_miles = deadhead_to_miles + service_miles + deadhead_return_miles
    
    # Get driver's current duty length to compute overtime
    daily_windows = driver_meta.get("daily_windows", {})
    current_duty_minutes = 0
    if daily_windows:
        # Take first available window as baseline
        for window in daily_windows.values():
            start = int(window.get("start_min", 0))
            end = int(window.get("end_min", 1440))
            if end < start:
                end += 24 * 60
            current_duty_minutes = max(current_duty_minutes, end - start)
    
    new_duty_minutes = current_duty_minutes + total_minutes
    overtime_minutes = max(0.0, new_duty_minutes - current_duty_minutes) if current_duty_minutes > 0 else total_minutes
    is_feasible = new_duty_minutes <= max_duty_minutes
    
    # Calculate cost
    deadhead_cost = total_miles * deadhead_cpm
    overtime_cost = overtime_minutes * overtime_cpm if overtime_minutes > 0 else 0.0
    total_cost = admin_cost + deadhead_cost + overtime_cost
    
    return total_cost, total_minutes, total_miles, overtime_minutes, is_feasible

def generate_candidates(
    req: PlanRequest,
    DATA: Dict[str,Any],
    matrices: Dict[str,Any],
    cost_cfg: Dict[str,float],
    loc_meta: Dict[str,Any],
    sla_windows: Dict[int, Dict[str,int]],
    geographic_proximity_miles: float = GEOGRAPHIC_PROXIMITY_MILES,
) -> Tuple[str, float, float, List[CandidateOut]]:
    """
    Complete candidate generation with robust filtering pipeline:
    1. Day-of-week filter ✅
    2. Duty window overlap + service time check 🆕
    3. Geographic proximity check 🆕  
    4. Full logistics cost calculation for all candidate types 🆕
    """
    
    # Basic setup
    req_min = minute_of_day_local(req.when_local)
    weekday = weekday_from_local(req.when_local)
    
    # SLA timing
    sla = sla_windows.get(int(req.priority), {"early_min": 60, "late_min": 60})
    earliest = max(0, int(req_min) - int(sla["early_min"]))
    latest = int(req_min) + int(sla["late_min"])
    
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    start_idx = idx_of(req.start_location, loc2idx)
    end_idx = idx_of(req.end_location, loc2idx)
    
    trip_minutes = float(req.trip_minutes) if req.trip_minutes is not None else minutes_between(start_idx, end_idx, Mtime)
    trip_miles = float(req.trip_miles) if req.trip_miles is not None else miles_between(start_idx, end_idx, Mdist)
    
    candidates: List[CandidateOut] = []
    
    # Get driver data
    ds = DATA["driver_states"] or {}
    drivers = ds["drivers"] if isinstance(ds, dict) and "drivers" in ds else ds
    
    total_drivers = len(drivers)
    after_day_filter = 0
    after_window_filter = 0  
    after_geo_filter = 0
    after_geocoding_filter = 0
    after_full_calc = 0
    
    # Helper functions for candidate types
    def _is_travel_leg(e: Dict[str,Any]) -> bool:
        if e.get("is_travel") is True: return True
        et = str(e.get("element_type","")).upper()
        return ("TRAVEL" in et) or ("LEG" in et and "TRAVEL" in et)

    def _is_empty_planz(e: Dict[str,Any]) -> bool:
        pc = str(e.get("planz_code", e.get("Planz Code",""))).strip().upper()
        lt = str(e.get("load_type", "")).strip().upper()
        return (
            ("EMPTY" in pc)
            or ("DEADHEAD" in pc)
            or ("EMPTY" in lt)
            or ("DEADHEAD" in lt)
            or ("RETURN" in lt)
            or bool(e.get("is_empty", False))
        )

    def _same_loc(a: Optional[int], b: Optional[int]) -> bool:
        try: return (a is not None) and (b is not None) and (int(a)==int(b))
        except Exception: return False

    def _loc_matches(e: Dict[str, Any], idx_field: str, name_field: str, target_idx: int) -> bool:
        loc_id = e.get(idx_field)
        if _same_loc(loc_id, target_idx):
            return True
        loc_name = str(e.get(name_field, "")).upper().strip()
        if not loc_name:
            return False
        mapped = loc2idx.get(loc_name)
        return _same_loc(mapped, target_idx)

    def _priority_allows_displacement(e: Dict[str, Any], new_priority: int) -> bool:
        existing_priority = int(e.get("priority", 3))
        return existing_priority >= int(new_priority)

    def _is_meal_relief(e: Dict[str, Any]) -> bool:
        et = str(e.get("element_type", "")).upper()
        pc = str(e.get("planz_code", e.get("Planz Code", ""))).upper()
        return ("MEAL RELIEF" in et) or ("MEAL RELIEF" in pc)

    def _element_duration_minutes(e: Dict[str, Any]) -> float:
        if e.get("duration_min") is not None:
            try:
                return max(0.0, float(e.get("duration_min")))
            except Exception:
                pass
        s = e.get("start_min")
        en = e.get("end_min")
        if s is None or en is None:
            return 0.0
        try:
            return max(0.0, float(en) - float(s))
        except Exception:
            return 0.0

    def _planned_leg_drive_minutes(e: Dict[str, Any]) -> float:
        if not _is_travel_leg(e):
            return 0.0
        return _element_duration_minutes(e)

    def _legal_state_for_day(elements_day: List[Dict[str, Any]]) -> Tuple[float, float, bool]:
        sorted_elements = sorted(
            elements_day,
            key=lambda x: float(x.get("start_min", 10**9)) if x.get("start_min") is not None else float(10**9),
        )
        total_drive = 0.0
        trailing_since_break = 0.0
        has_qualifying_break = False
        legal_break_min = int(cost_cfg.get("legal_min_break_minutes", 45))

        for e in sorted_elements:
            dur = _element_duration_minutes(e)
            if dur <= 0:
                continue
            if _is_travel_leg(e):
                total_drive += dur
                trailing_since_break += dur
                continue
            if _is_meal_relief(e) and dur >= legal_break_min:
                trailing_since_break = 0.0
                has_qualifying_break = True

        return total_drive, trailing_since_break, has_qualifying_break

    def _driver_legal_history_minutes(driver_meta_local: Dict[str, Any]) -> Tuple[float, float, float]:
        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        week_drive = _as_float(
            driver_meta_local.get("week_drive_minutes", driver_meta_local.get("current_week_drive_minutes", 0.0)),
            0.0,
        )
        prev_week_drive = _as_float(
            driver_meta_local.get("previous_week_drive_minutes", driver_meta_local.get("last_week_drive_minutes", 0.0)),
            0.0,
        )
        week_work = _as_float(
            driver_meta_local.get("week_work_minutes", driver_meta_local.get("current_week_work_minutes", week_drive)),
            week_drive,
        )
        return week_drive, prev_week_drive, week_work

    def _weekly_rest_compliant(driver_meta_local: Dict[str, Any]) -> bool:
        if not ENABLE_STRICT_LEGAL_CONSTRAINTS:
            return True

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        periods_since_weekly_rest = _as_int(
            driver_meta_local.get(
                "consecutive_24h_periods_since_weekly_rest",
                driver_meta_local.get("periods_since_weekly_rest", 0),
            ),
            0,
        )
        max_periods_between_weekly_rests = _as_int(
            cost_cfg.get("legal_max_24h_periods_between_weekly_rests", 6),
            6,
        )
        if periods_since_weekly_rest > max_periods_between_weekly_rests:
            return False

        reduced_weekly_rests_used = _as_int(driver_meta_local.get("reduced_weekly_rests_used", 0), 0)
        max_reduced_weekly_rests = _as_int(
            cost_cfg.get("legal_max_reduced_weekly_rests_between_regular", 2),
            2,
        )
        if reduced_weekly_rests_used > max_reduced_weekly_rests:
            return False

        reduced_rest_compensation_due = _as_float(
            driver_meta_local.get("reduced_rest_compensation_minutes_due", 0.0),
            0.0,
        )
        allow_uncompensated_reduced_rest = bool(
            cost_cfg.get("legal_allow_uncompensated_reduced_rest", False)
        )
        if reduced_rest_compensation_due > 0 and not allow_uncompensated_reduced_rest:
            return False

        min_reduced_weekly_rest_minutes = _as_int(
            cost_cfg.get("legal_min_reduced_weekly_rest_minutes", 24 * 60),
            24 * 60,
        )
        last_weekly_rest_minutes = _as_float(
            driver_meta_local.get(
                "weekly_rest_last_duration_minutes",
                driver_meta_local.get("last_weekly_rest_minutes", 0.0),
            ),
            0.0,
        )
        if last_weekly_rest_minutes > 0 and last_weekly_rest_minutes < min_reduced_weekly_rest_minutes:
            return False

        return True

    def _strict_legal_feasible(
        existing_drive: float,
        trailing_since_break: float,
        additional_drive: float,
        week_drive_minutes: float,
        previous_week_drive_minutes: float,
        week_work_minutes: float,
        weekly_rest_ok: bool,
    ) -> bool:
        if not ENABLE_STRICT_LEGAL_CONSTRAINTS:
            return True

        if not weekly_rest_ok:
            return False

        max_daily = int(cost_cfg.get("legal_max_daily_driving_minutes", 9 * 60))
        max_daily_extended = int(cost_cfg.get("legal_extended_daily_driving_minutes", 10 * 60))
        allow_extended_daily = bool(cost_cfg.get("legal_allow_extended_daily_driving", False))
        max_continuous = int(cost_cfg.get("legal_max_continuous_driving_minutes", int(4.5 * 60)))

        allowed_daily = max_daily_extended if allow_extended_daily else max_daily
        if (float(existing_drive) + float(additional_drive)) > float(allowed_daily):
            return False

        if float(additional_drive) > float(max_continuous):
            return False

        if (float(trailing_since_break) + float(additional_drive)) > float(max_continuous):
            return False

        max_weekly_drive = int(cost_cfg.get("legal_max_weekly_driving_minutes", 56 * 60))
        max_fortnight_drive = int(cost_cfg.get("legal_max_fortnight_driving_minutes", 90 * 60))

        if (float(week_drive_minutes) + float(additional_drive)) > float(max_weekly_drive):
            return False

        if (float(previous_week_drive_minutes) + float(week_drive_minutes) + float(additional_drive)) > float(max_fortnight_drive):
            return False

        enforce_weekly_work_cap = bool(cost_cfg.get("legal_enforce_weekly_work_minutes", False))
        if enforce_weekly_work_cap:
            max_weekly_work = int(cost_cfg.get("legal_max_weekly_work_minutes", 60 * 60))
            if (float(week_work_minutes) + float(additional_drive)) > float(max_weekly_work):
                return False

        return True

    def _is_within_radius(from_idx: int, to_idx: int, max_miles: float) -> bool:
        if from_idx == to_idx:
            return True
        miles = miles_between(int(from_idx), int(to_idx), Mdist)
        if USE_HAVERSINE_DEADHEAD and loc_meta:
            hv = haversine_between_idx(int(from_idx), int(to_idx), loc2idx, loc_meta)
            if hv is not None:
                miles = float(hv)
        return float(miles) <= float(max_miles)

    def _same_island_or_unknown(from_idx: int, to_idx: int) -> bool:
        if not ENFORCE_SAME_ISLAND:
            return True
        try:
            from_name = next((k for k, v in loc2idx.items() if int(v) == int(from_idx)), None)
            to_name = next((k for k, v in loc2idx.items() if int(v) == int(to_idx)), None)
            if not from_name or not to_name:
                return True
            from_meta = loc_meta.get(str(from_name).upper()) if loc_meta else None
            to_meta = loc_meta.get(str(to_name).upper()) if loc_meta else None
            same = same_island_by_meta(from_meta, to_meta)
            return True if same is None else bool(same)
        except Exception:
            return True
    
    for duty_id, driver_meta in drivers.items():
        
        elements_all = driver_meta.get("elements", []) or []
        elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
        if not elements:
            continue
        after_day_filter += 1
        
        # NEW FILTER 1.5: Check for invalid locations in driver's duty
        has_invalid_locations = False
        if loc_meta:
            for e in elements:
                for field in ["from", "to"]:
                    loc_name = str(e.get(field, "")).upper().strip()
                    if loc_name and loc_name != "NO_DATA" and loc_name in loc2idx:
                        # Check if this location has valid coordinates in loc_meta
                        loc_meta_entry = loc_meta.get(loc_name)
                        if not loc_meta_entry or not (loc_meta_entry.get("lat") and loc_meta_entry.get("lon")):
                            has_invalid_locations = True
                            break
                if has_invalid_locations:
                    break

        if has_invalid_locations:
            continue  # Skip this entire driver
        after_geocoding_filter += 1  # Add this counter
        
        # FILTER 2: Duty window overlap + basic service time check 🆕
        if not check_driver_duty_window_overlap(driver_meta, weekday, req_min, trip_minutes):
            continue
        after_window_filter += 1
        
        # FILTER 3: Geographic proximity check 🆕
        driver_home_base = get_driver_home_base(driver_meta, loc2idx)
        if driver_home_base is None:
            continue  # Can't determine home base
            
        if not check_geographic_proximity(driver_home_base, start_idx, geographic_proximity_miles, loc2idx, loc_meta):
            continue
        after_geo_filter += 1

        existing_drive_minutes, trailing_drive_since_break, _ = _legal_state_for_day(elements)
        week_drive_minutes, previous_week_drive_minutes, week_work_minutes = _driver_legal_history_minutes(driver_meta)
        weekly_rest_ok = _weekly_rest_compliant(driver_meta)
        
        # FILTER 4: Generate all candidate types with full logistics calculation 🆕
        
        # TYPE 1: Take existing EMPTY A->B legs
        for e in elements:
            if not _is_travel_leg(e) or not _is_empty_planz(e):
                continue
            if not (_loc_matches(e, "from_id", "from", start_idx) and _loc_matches(e, "to_id", "to", end_idx)):
                continue
            s = e.get("start_min"); en = e.get("end_min")
            if s is None or en is None: continue
            
            ok_time = (
                (req.mode == "depart_after"  and earliest <= s <= latest) or
                (req.mode == "arrive_before" and earliest <= en <= latest)
            )
            if not ok_time: continue
            
            # For empty legs, minimal cost (no deadhead since route matches exactly)
            try:
                admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
                planned_leg_drive = _planned_leg_drive_minutes(e)
                additional_drive = max(0.0, float(trip_minutes) - float(planned_leg_drive))
                legal_ok = _strict_legal_feasible(
                    existing_drive_minutes,
                    trailing_drive_since_break,
                    additional_drive,
                    week_drive_minutes,
                    previous_week_drive_minutes,
                    week_work_minutes,
                    weekly_rest_ok,
                )
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::take_empty@{int(s)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=float(trip_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=bool(legal_ok), est_cost=float(admin_cost),
                    reason="Take existing empty A→B leg",
                    reason_code="RULE_EMPTY_EXACT_REPLACE",
                    reason_detail="Exact A->B empty/deadhead leg replaced by requested service",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Empty leg calculation failed for {duty_id}: {e}")
                continue
        
        # TYPE 2: Swap departure from A (same/lower priority)
        if ENABLE_PARTIAL_OVERLAP_INSERTION or ENABLE_NEARBY_DEPOT_SUBSTITUTION:
            for e in elements:
                if not _is_travel_leg(e) or not _loc_matches(e, "from_id", "from", start_idx):
                    continue
                s = e.get("start_min"); en = e.get("end_min")
                if s is None or en is None: continue
                
                ok_time = (
                    (req.mode == "depart_after"  and earliest <= s <= latest) or
                    (req.mode == "arrive_before" and earliest <= en <= latest)
                )
                if not ok_time: continue
                if int(e.get("priority", 3)) < int(req.priority): continue
                
                try:
                    admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
                    planned_leg_drive = _planned_leg_drive_minutes(e)
                    additional_drive = max(0.0, float(trip_minutes) - float(planned_leg_drive))
                    legal_ok = _strict_legal_feasible(
                        existing_drive_minutes,
                        trailing_drive_since_break,
                        additional_drive,
                        week_drive_minutes,
                        previous_week_drive_minutes,
                        week_work_minutes,
                        weekly_rest_ok,
                    )
                    candidate = CandidateOut(
                        candidate_id=f"{duty_id}::swap_from_A@{int(s)}",
                        driver_id=str(duty_id), route_id=str(duty_id),
                        deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                        miles_delta=float(trip_miles), delay_minutes=0.0, uses_emergency_rest=False,
                        feasible_hard=bool(legal_ok), est_cost=float(admin_cost),
                        reason="Swap existing departure from A",
                        reason_code="RULE_SWAP_DEPARTURE_FROM_A",
                        reason_detail="Departure from matching origin can be reassigned within time and priority bounds",
                    )
                    candidates.append(candidate)
                    after_full_calc += 1
                except Exception as e:
                    print(f"[warn] Swap calculation failed for {duty_id}: {e}")
                    continue
        
        # TYPE 3: Use slack time (AS DIRECTED)
        if ENABLE_AS_DIRECTED_INSERTION:
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
                
                # Check if slack time is sufficient for full logistics chain
                dd_to_min = minutes_between(loc_id, start_idx, Mtime)
                dd_back_min = minutes_between(end_idx, loc_id, Mtime)
                budget = dd_to_min + LOADING_TIME_MINUTES + trip_minutes + OFFLOADING_TIME_MINUTES + dd_back_min
                if budget > e_dur: continue
                
                try:
                    total_cost, total_minutes, total_miles, overtime_minutes, is_feasible = compute_full_logistics_cost(
                        driver_meta, loc_id, start_idx, end_idx, 
                        trip_minutes, trip_miles, cost_cfg, Mtime, Mdist, loc2idx, loc_meta
                    )
                    additional_drive = max(0.0, float(trip_minutes) + float(max(0.0, total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES)))
                    legal_ok = _strict_legal_feasible(
                        existing_drive_minutes,
                        trailing_drive_since_break,
                        additional_drive,
                        week_drive_minutes,
                        previous_week_drive_minutes,
                        week_work_minutes,
                        weekly_rest_ok,
                    )
                    
                    candidate = CandidateOut(
                        candidate_id=f"{duty_id}::slack@{loc_id}",
                        driver_id=str(duty_id), route_id=str(duty_id),
                        deadhead_miles=float(total_miles - trip_miles),
                        deadhead_minutes=float(total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES),
                        overtime_minutes=float(overtime_minutes),
                        miles_delta=float(total_miles), delay_minutes=0.0, uses_emergency_rest=False,
                        feasible_hard=bool(is_feasible and legal_ok), est_cost=float(total_cost),
                        reason="Use AS DIRECTED slack time",
                        reason_code="RULE_AS_DIRECTED_SLACK",
                        reason_detail="AS DIRECTED time can absorb pickup, service, and return repositioning",
                    )
                    candidates.append(candidate)
                    after_full_calc += 1
                except Exception as e:
                    print(f"[warn] Slack calculation failed for {duty_id}: {e}")
                    continue
        
        # TYPE 4: Partial overlap families around an existing A->B travel leg
        if ENABLE_PARTIAL_OVERLAP_INSERTION:
            for e in elements:
                if not _is_travel_leg(e):
                    continue

                leg_from = e.get("from_id")
                leg_to = e.get("to_id")
                if leg_from is None:
                    leg_from_name = str(e.get("from", "")).upper().strip()
                    leg_from = loc2idx.get(leg_from_name)
                if leg_to is None:
                    leg_to_name = str(e.get("to", "")).upper().strip()
                    leg_to = loc2idx.get(leg_to_name)
                if leg_from is None or leg_to is None:
                    continue

                e_start = e.get("start_min")
                e_end = e.get("end_min")
                if e_start is None or e_end is None:
                    continue

                if not _priority_allows_displacement(e, int(req.priority)):
                    continue

                # 4A. Pre-pickup overlap: planned A->B, new C->B (A->C empty + C->B loaded)
                if int(leg_to) == int(end_idx) and int(leg_from) != int(start_idx):
                    deadhead_to_pickup = minutes_between(int(leg_from), int(start_idx), Mtime)
                    loaded_trip = float(trip_minutes)
                    overlap_window = deadhead_to_pickup + loaded_trip

                    if overlap_window <= max(0.0, float(e_end - e_start)) and earliest <= e_start <= latest:
                        try:
                            total_cost, total_minutes, total_miles, overtime_minutes, is_feasible = compute_full_logistics_cost(
                                driver_meta,
                                int(leg_from),
                                start_idx,
                                end_idx,
                                trip_minutes,
                                trip_miles,
                                cost_cfg,
                                Mtime,
                                Mdist,
                                loc2idx,
                                loc_meta,
                            )
                            planned_leg_drive = _planned_leg_drive_minutes(e)
                            additional_drive = max(
                                0.0,
                                float(trip_minutes) + float(max(0.0, total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES)) - float(planned_leg_drive),
                            )
                            legal_ok = _strict_legal_feasible(
                                existing_drive_minutes,
                                trailing_drive_since_break,
                                additional_drive,
                                week_drive_minutes,
                                previous_week_drive_minutes,
                                week_work_minutes,
                                weekly_rest_ok,
                            )
                            candidates.append(
                                CandidateOut(
                                    candidate_id=f"{duty_id}::overlap_prepickup@{int(e_start)}",
                                    driver_id=str(duty_id),
                                    route_id=str(duty_id),
                                    deadhead_miles=float(max(0.0, total_miles - trip_miles)),
                                    deadhead_minutes=float(max(0.0, total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES)),
                                    overtime_minutes=float(overtime_minutes),
                                    miles_delta=float(total_miles),
                                    delay_minutes=0.0,
                                    uses_emergency_rest=False,
                                    feasible_hard=bool(is_feasible and legal_ok),
                                    est_cost=float(total_cost),
                                    reason="Pre-pickup overlap insertion (A→C empty, C→B loaded)",
                                    reason_code="RULE_OVERLAP_PREPICKUP",
                                    reason_detail="Existing leg endpoint aligns with requested destination; insertion can rejoin leg endpoint",
                                )
                            )
                            after_full_calc += 1
                        except Exception as exc:
                            print(f"[warn] Prepickup overlap failed for {duty_id}: {exc}")

                # 4B. Post-drop overlap: planned A->B, new A->C (A->C loaded + C->B empty)
                if int(leg_from) == int(start_idx) and int(leg_to) != int(end_idx):
                    deadhead_to_rejoin = minutes_between(int(end_idx), int(leg_to), Mtime)
                    loaded_trip = float(trip_minutes)
                    overlap_window = loaded_trip + deadhead_to_rejoin

                    if overlap_window <= max(0.0, float(e_end - e_start)) and earliest <= e_start <= latest:
                        try:
                            total_cost, total_minutes, total_miles, overtime_minutes, is_feasible = compute_full_logistics_cost(
                                driver_meta,
                                int(leg_from),
                                start_idx,
                                end_idx,
                                trip_minutes,
                                trip_miles,
                                cost_cfg,
                                Mtime,
                                Mdist,
                                loc2idx,
                                loc_meta,
                            )
                            planned_leg_drive = _planned_leg_drive_minutes(e)
                            additional_drive = max(
                                0.0,
                                float(trip_minutes) + float(max(0.0, total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES)) - float(planned_leg_drive),
                            )
                            legal_ok = _strict_legal_feasible(
                                existing_drive_minutes,
                                trailing_drive_since_break,
                                additional_drive,
                                week_drive_minutes,
                                previous_week_drive_minutes,
                                week_work_minutes,
                                weekly_rest_ok,
                            )
                            candidates.append(
                                CandidateOut(
                                    candidate_id=f"{duty_id}::overlap_postdrop@{int(e_start)}",
                                    driver_id=str(duty_id),
                                    route_id=str(duty_id),
                                    deadhead_miles=float(max(0.0, total_miles - trip_miles)),
                                    deadhead_minutes=float(max(0.0, total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES)),
                                    overtime_minutes=float(overtime_minutes),
                                    miles_delta=float(total_miles),
                                    delay_minutes=0.0,
                                    uses_emergency_rest=False,
                                    feasible_hard=bool(is_feasible and legal_ok),
                                    est_cost=float(total_cost),
                                    reason="Post-drop overlap insertion (A→C loaded, C→B empty)",
                                    reason_code="RULE_OVERLAP_POSTDROP",
                                    reason_detail="Existing leg origin aligns with requested origin; insertion can rejoin leg destination",
                                )
                            )
                            after_full_calc += 1
                        except Exception as exc:
                            print(f"[warn] Postdrop overlap failed for {duty_id}: {exc}")

        # TYPE 5: Nearby bilateral depot substitution (C~A and D~B)
        if ENABLE_NEARBY_DEPOT_SUBSTITUTION:
            for e in elements:
                if not _is_travel_leg(e):
                    continue

                leg_from = e.get("from_id")
                leg_to = e.get("to_id")
                if leg_from is None:
                    leg_from_name = str(e.get("from", "")).upper().strip()
                    leg_from = loc2idx.get(leg_from_name)
                if leg_to is None:
                    leg_to_name = str(e.get("to", "")).upper().strip()
                    leg_to = loc2idx.get(leg_to_name)
                if leg_from is None or leg_to is None:
                    continue

                if int(leg_from) == int(start_idx) and int(leg_to) == int(end_idx):
                    continue
                if not _priority_allows_displacement(e, int(req.priority)):
                    continue

                if not _is_within_radius(int(leg_from), int(start_idx), geographic_proximity_miles):
                    continue
                if not _is_within_radius(int(leg_to), int(end_idx), geographic_proximity_miles):
                    continue
                if not _same_island_or_unknown(int(leg_from), int(start_idx)):
                    continue
                if not _same_island_or_unknown(int(leg_to), int(end_idx)):
                    continue

                e_start = e.get("start_min")
                e_end = e.get("end_min")
                if e_start is None or e_end is None:
                    continue

                ok_time = (
                    (req.mode == "depart_after" and earliest <= e_start <= latest) or
                    (req.mode == "arrive_before" and earliest <= e_end <= latest)
                )
                if not ok_time:
                    continue

                pre_reposition_min = minutes_between(int(leg_from), int(start_idx), Mtime)
                post_reposition_min = minutes_between(int(end_idx), int(leg_to), Mtime)
                pre_reposition_mi = miles_between(int(leg_from), int(start_idx), Mdist)
                post_reposition_mi = miles_between(int(end_idx), int(leg_to), Mdist)

                if USE_HAVERSINE_DEADHEAD and loc_meta:
                    hv_pre_mi = haversine_between_idx(int(leg_from), int(start_idx), loc2idx, loc_meta)
                    hv_post_mi = haversine_between_idx(int(end_idx), int(leg_to), loc2idx, loc_meta)
                    if hv_pre_mi is not None:
                        pre_reposition_mi = float(hv_pre_mi)
                    if hv_post_mi is not None:
                        post_reposition_mi = float(hv_post_mi)

                extra_deadhead_min = float(pre_reposition_min + post_reposition_min)
                extra_deadhead_mi = float(pre_reposition_mi + post_reposition_mi)

                deadhead_cpm = float(cost_cfg.get("deadhead_cost_per_mile", 1.0))
                admin_cost = float(cost_cfg.get("reassignment_admin_cost", 10.0))
                est_cost = admin_cost + extra_deadhead_mi * deadhead_cpm
                planned_leg_drive = _planned_leg_drive_minutes(e)
                additional_drive = max(0.0, float(trip_minutes) + float(extra_deadhead_min) - float(planned_leg_drive))
                legal_ok = _strict_legal_feasible(
                    existing_drive_minutes,
                    trailing_drive_since_break,
                    additional_drive,
                    week_drive_minutes,
                    previous_week_drive_minutes,
                    week_work_minutes,
                    weekly_rest_ok,
                )

                candidates.append(
                    CandidateOut(
                        candidate_id=f"{duty_id}::nearby_substitution@{int(e_start)}",
                        driver_id=str(duty_id),
                        route_id=str(duty_id),
                        deadhead_miles=extra_deadhead_mi,
                        deadhead_minutes=extra_deadhead_min,
                        overtime_minutes=0.0,
                        miles_delta=float(trip_miles + extra_deadhead_mi),
                        delay_minutes=0.0,
                        uses_emergency_rest=False,
                        feasible_hard=bool(legal_ok),
                        est_cost=float(est_cost),
                        reason="Nearby bilateral substitution",
                        reason_code="RULE_NEARBY_BILATERAL_SUBSTITUTION",
                        reason_detail="Planned leg endpoints are within pickup/drop neighborhood radii of requested endpoints",
                    )
                )
                after_full_calc += 1

        # TYPE 6: Exact A->B swap (higher priority swapping lower)
        for e in elements:
            if not _is_travel_leg(e): continue
            if not (_loc_matches(e, "from_id", "from", start_idx) and _loc_matches(e, "to_id", "to", end_idx)):
                continue
            
            e_start = e.get("start_min"); e_end = e.get("end_min")
            if e_start is None or e_end is None: continue
            
            ok_time = (
                (req.mode == "depart_after"  and (e_start >= earliest) and (e_start <= latest)) or
                (req.mode == "arrive_before" and (e_end   <= latest)   and (e_end   >= earliest))
            )
            if not ok_time: continue
            if int(e.get("priority", 3)) < int(req.priority): continue
            
            try:
                admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
                planned_leg_drive = _planned_leg_drive_minutes(e)
                additional_drive = max(0.0, float(trip_minutes) - float(planned_leg_drive))
                legal_ok = _strict_legal_feasible(
                    existing_drive_minutes,
                    trailing_drive_since_break,
                    additional_drive,
                    week_drive_minutes,
                    previous_week_drive_minutes,
                    week_work_minutes,
                    weekly_rest_ok,
                )
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::swap_leg@{int(e_start)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=float(trip_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=bool(legal_ok), est_cost=float(admin_cost),
                    reason="Exact A→B leg swap",
                    reason_code="RULE_EXACT_LEG_SWAP",
                    reason_detail="Exact A->B travel leg swapped under priority/time constraints",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Exact swap calculation failed for {duty_id}: {e}")
                continue
        
        # TYPE 7: Append after last element (with full logistics chain)
        # Find last element's location
        last_loc_id = None
        if elements:
            for e in reversed(elements):  # Start from end
                if _is_travel_leg(e):
                    last_loc_id = e.get("to_id")
                    if last_loc_id is None:
                        to_name = str(e.get("to", "")).upper().strip()
                        if to_name in loc2idx:
                            last_loc_id = loc2idx[to_name]
                    if last_loc_id is not None:
                        break
            
            # If no travel leg found, try other elements
            if last_loc_id is None:
                for e in reversed(elements):
                    for field in ["to_id", "from_id"]:
                        if e.get(field) is not None:
                            last_loc_id = e.get(field)
                            break
                    if last_loc_id is None:
                        for field in ["to", "from"]:
                            loc_name = str(e.get(field, "")).upper().strip()
                            if loc_name in loc2idx:
                                last_loc_id = loc2idx[loc_name]
                                break
                    if last_loc_id is not None:
                        break
        
        if last_loc_id is not None:
            try:
                total_cost, total_minutes, total_miles, overtime_minutes, is_feasible = compute_full_logistics_cost(
                    driver_meta, int(last_loc_id), start_idx, end_idx, 
                    trip_minutes, trip_miles, cost_cfg, Mtime, Mdist, loc2idx, loc_meta
                )
                additional_drive = max(0.0, float(trip_minutes) + float(max(0.0, total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES)))
                legal_ok = _strict_legal_feasible(
                    existing_drive_minutes,
                    trailing_drive_since_break,
                    additional_drive,
                    week_drive_minutes,
                    previous_week_drive_minutes,
                    week_work_minutes,
                    weekly_rest_ok,
                )
                
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::append",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=float(total_miles - trip_miles),
                    deadhead_minutes=float(total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES),
                    overtime_minutes=float(overtime_minutes),
                    miles_delta=float(total_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=bool(is_feasible and legal_ok), est_cost=float(total_cost),
                    reason="Append after last duty element",
                    reason_code="RULE_APPEND_AFTER_LAST",
                    reason_detail="No direct replacement path; appended after last known duty position",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Append calculation failed for {duty_id}: {e}")
                continue
    
    # Sort by tier priority, then feasibility, then multi-criteria score
    def candidate_rank_score(c: CandidateOut, tier: int) -> float:
        deadhead_weight = float(cost_cfg.get("rank_deadhead_miles_weight", 1.0))
        deadhead_minutes_weight = float(cost_cfg.get("rank_deadhead_minutes_weight", 0.15))
        overtime_weight = float(cost_cfg.get("rank_overtime_minutes_weight", 2.0))
        delay_weight = float(cost_cfg.get("rank_delay_minutes_weight", 1.0))

        family_penalties = {
            0: float(cost_cfg.get("rank_penalty_take_empty", 0.0)),
            1: float(cost_cfg.get("rank_penalty_swap_leg", 1.0)),
            2: float(cost_cfg.get("rank_penalty_swap_from_a", 2.0)),
            3: float(cost_cfg.get("rank_penalty_slack", 4.0)),
            4: float(cost_cfg.get("rank_penalty_overlap_prepickup", 6.0)),
            5: float(cost_cfg.get("rank_penalty_overlap_postdrop", 6.0)),
            6: float(cost_cfg.get("rank_penalty_nearby_substitution", 8.0)),
            7: float(cost_cfg.get("rank_penalty_append", 30.0)),
        }

        return (
            float(c.est_cost)
            + float(c.deadhead_miles) * deadhead_weight
            + float(c.deadhead_minutes) * deadhead_minutes_weight
            + float(c.overtime_minutes) * overtime_weight
            + float(c.delay_minutes) * delay_weight
            + float(family_penalties.get(tier, float(cost_cfg.get("rank_penalty_unknown", 15.0))))
        )

    def rank_key(c: CandidateOut):
        cid = c.candidate_id
        if "::take_empty@" in cid:     tier = 0  # Highest priority
        elif "::swap_leg@" in cid:     tier = 1
        elif "::swap_from_A@" in cid:  tier = 2
        elif "::slack@" in cid:        tier = 3
        elif "::overlap_prepickup@" in cid: tier = 4
        elif "::overlap_postdrop@" in cid: tier = 5
        elif "::nearby_substitution@" in cid: tier = 6
        elif "::append" in cid:        tier = 7  # Lowest priority
        else:                          tier = 8
        return (not c.feasible_hard, tier, candidate_rank_score(c, tier), c.est_cost)
    
    candidates.sort(key=rank_key)
    
    # Apply top_n limit if specified
    if req.top_n and req.top_n > 0:
        candidates = candidates[:req.top_n]
    
    # Debug info
    print(f"[debug] Candidate filtering: {total_drivers} → {after_day_filter} → {after_window_filter} → {after_geo_filter} → {after_full_calc} candidates")
    
    return weekday, float(trip_minutes), float(trip_miles), candidates
