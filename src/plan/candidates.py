from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from .models import PlanRequest, CandidateOut, WEEKDAYS
from .config import ENFORCE_SAME_ISLAND, USE_HAVERSINE_DEADHEAD, HAV_MAX_DEADHEAD_ONE_WAY_MI
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
    if weekday not in daily_windows:
        return False
        
    window = daily_windows[weekday]
    duty_start = int(window.get("start_min", 0))
    duty_end = int(window.get("end_min", 1440))
    
    # Handle cross-midnight duties
    if duty_end < duty_start:
        duty_end += 24 * 60
    
    # Check if we have enough time for: request start + service + loading/offloading
    service_end_time = req_start_min + total_service_minutes + TOTAL_SERVICE_TIME_MINUTES
    
    # Must fit within duty window
    return duty_start <= req_start_min and service_end_time <= duty_end

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
        return ("EMPTY" in pc) or ("TRAVEL_NO_DATA" in pc) or ("TRAVEL NO DATA" in pc) or bool(e.get("is_empty", False))

    def _same_loc(a: Optional[int], b: Optional[int]) -> bool:
        try: return (a is not None) and (b is not None) and (int(a)==int(b))
        except Exception: return False
    
    for duty_id, driver_meta in drivers.items():
        
        elements_all = driver_meta.get("elements", []) or []
        elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
        if not elements:
            continue
        after_day_filter += 1
        
        # NEW FILTER 1.5: Check for invalid locations in driver's duty
        has_invalid_locations = False
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
        
        # FILTER 4: Generate all candidate types with full logistics calculation 🆕
        
        # TYPE 1: Take existing EMPTY A->B legs
        for e in elements:
            if not _is_travel_leg(e) or not _is_empty_planz(e):
                continue
            if not (_same_loc(e.get("from_id"), start_idx) and _same_loc(e.get("to_id"), end_idx)):
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
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::take_empty@{int(s)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=float(trip_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=True, est_cost=float(admin_cost),
                    reason="Take existing empty A→B leg",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Empty leg calculation failed for {duty_id}: {e}")
                continue
        
        # TYPE 2: Swap departure from A (same/lower priority)
        for e in elements:
            if not _is_travel_leg(e) or not _same_loc(e.get("from_id"), start_idx):
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
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::swap_from_A@{int(s)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=float(trip_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=True, est_cost=float(admin_cost),
                    reason="Swap existing departure from A",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Swap calculation failed for {duty_id}: {e}")
                continue
        
        # TYPE 3: Use slack time (AS DIRECTED)
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
                
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::slack@{loc_id}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=float(total_miles - trip_miles),
                    deadhead_minutes=float(total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES),
                    overtime_minutes=float(overtime_minutes),
                    miles_delta=float(total_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=bool(is_feasible), est_cost=float(total_cost),
                    reason="Use AS DIRECTED slack time",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Slack calculation failed for {duty_id}: {e}")
                continue
        
        # TYPE 4: Exact A->B swap (higher priority swapping lower)
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
            
            try:
                admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::swap_leg@{int(e_start)}",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=0.0, deadhead_minutes=0.0, overtime_minutes=0.0,
                    miles_delta=float(trip_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=True, est_cost=float(admin_cost),
                    reason="Exact A→B leg swap",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Exact swap calculation failed for {duty_id}: {e}")
                continue
        
        # TYPE 5: Append after last element (with full logistics chain)
        # Find last element's location
        last_loc_id = None
        if elements:
            for e in reversed(elements):  # Start from end
                if _is_travel_leg(e):
                    last_loc_id = e.get("to_id")
                    if last_loc_id is not None:
                        break
            
            # If no travel leg found, try other elements
            if last_loc_id is None:
                for e in reversed(elements):
                    for field in ["to_id", "from_id"]:
                        if e.get(field) is not None:
                            last_loc_id = e.get(field)
                            break
                    if last_loc_id is not None:
                        break
        
        if last_loc_id is not None:
            try:
                total_cost, total_minutes, total_miles, overtime_minutes, is_feasible = compute_full_logistics_cost(
                    driver_meta, int(last_loc_id), start_idx, end_idx, 
                    trip_minutes, trip_miles, cost_cfg, Mtime, Mdist, loc2idx, loc_meta
                )
                
                candidate = CandidateOut(
                    candidate_id=f"{duty_id}::append",
                    driver_id=str(duty_id), route_id=str(duty_id),
                    deadhead_miles=float(total_miles - trip_miles),
                    deadhead_minutes=float(total_minutes - trip_minutes - TOTAL_SERVICE_TIME_MINUTES),
                    overtime_minutes=float(overtime_minutes),
                    miles_delta=float(total_miles), delay_minutes=0.0, uses_emergency_rest=False,
                    feasible_hard=bool(is_feasible), est_cost=float(total_cost),
                    reason="Append after last duty element",
                )
                candidates.append(candidate)
                after_full_calc += 1
            except Exception as e:
                print(f"[warn] Append calculation failed for {duty_id}: {e}")
                continue
    
    # Sort by tier priority, then feasibility, then cost
    def rank_key(c: CandidateOut):
        cid = c.candidate_id
        if "::take_empty@" in cid:     tier = 0  # Highest priority
        elif "::swap_leg@" in cid:     tier = 1
        elif "::swap_from_A@" in cid:  tier = 2
        elif "::slack@" in cid:        tier = 3
        elif "::append" in cid:        tier = 4  # Lowest priority
        else:                          tier = 5
        return (not c.feasible_hard, tier, c.est_cost)
    
    candidates.sort(key=rank_key)
    
    # Apply top_n limit if specified
    if req.top_n and req.top_n > 0:
        candidates = candidates[:req.top_n]
    
    # Debug info
    print(f"[debug] Candidate filtering: {total_drivers} → {after_day_filter} → {after_window_filter} → {after_geo_filter} → {after_full_calc} candidates")
    
    return weekday, float(trip_minutes), float(trip_miles), candidates
