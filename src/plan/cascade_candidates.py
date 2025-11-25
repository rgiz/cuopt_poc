"""
cascade_candidates.py

Single-driver cascade candidate generation using cuOpt optimization.

This module generates driver candidates for new service requests by:
1. Filtering drivers based on location, time windows, and calling points
2. Running single-driver cuOpt optimization for each viable candidate
3. Building before/after schedules for UI visualization

Multi-driver cascade logic has been archived - see _archive/multi_driver_cascade_wip.py
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import time

from .models import PlanRequest, CandidateOut
from .candidates import minute_of_day_local, weekday_from_local, element_active_on_weekday, idx_of
from .config import load_sla_windows

try:
    from cuopt_sh_client import CuOptServiceSelfHostClient
    CUOPT_CLIENT_AVAILABLE = True
    print("[cascade] cuopt-sh-client imported successfully")
except ImportError as e:
    print(f"[cascade] cuopt-sh-client not available: {e}")
    CUOPT_CLIENT_AVAILABLE = False

LOADING_TIME_MINUTES = 30
OFFLOADING_TIME_MINUTES = 20

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CascadeCandidateOut:
    """Result from cascade evaluation with schedule details"""
    candidate_id: str
    primary_driver_id: str
    total_system_cost: float
    drivers_affected: int
    cascade_chain: List[Any]
    before_after_schedules: Dict[str, Dict[str, Any]]
    is_fully_feasible: bool
    uncovered_p4_tasks: List[Dict[str, Any]]
    disposed_p5_tasks: List[Dict[str, Any]]
    
    def to_candidate_out(self) -> CandidateOut:
        """Convert to API response format"""
        return CandidateOut(
            candidate_id=self.primary_driver_id,
            driver_id=self.primary_driver_id,
            type="cascade_optimized",
            est_cost=self.total_system_cost,
            deadhead_miles=0.0,
            delay_minutes=0.0,
            overtime_minutes=0.0,
            uses_emergency_rest=False,
            miles_delta=0.0,
            feasible_hard=self.is_fully_feasible,
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def generate_cascade_candidates_with_schedules(
    req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    loc_meta: Dict[str, Any],
    sla_windows: Dict[int, Dict[str, int]],
    max_cascade_depth: int = 2,
    max_candidates: int = 10,
) -> Tuple[str, float, float, List[CandidateOut], List[Dict[str, Any]]]:
    """
    Generate candidates using single-driver cuOpt optimization.
    
    Returns:
        (weekday, trip_minutes, trip_miles, candidates, schedules)
    """
    weekday = weekday_from_local(req.when_local)
    req_min = minute_of_day_local(req.when_local)
    
    # Test cuOpt connectivity
    cuopt_working = _test_official_cuopt_client()
    status = "OFFICIAL_CLIENT" if cuopt_working else "FALLBACK"
    print(f"[cascade] Generating candidates for {req.start_location}→{req.end_location} (cuOpt: {status})")
    
    try:
        # Filter viable drivers
        base_drivers = _enhanced_driver_filtering(
            req, DATA, matrices, loc_meta, weekday, req_min, sla_windows,
            calling_point_proximity_miles=50,
            max_drivers=20
        )
        
        if not base_drivers:
            print("[cascade] No viable drivers found")
            return weekday, 0.0, 0.0, [], []
        
        # Evaluate each driver with cuOpt (parallel)
        cascade_solutions = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    _evaluate_single_driver_with_cuopt,
                    driver_id=driver_id,
                    new_trip_req=req,
                    DATA=DATA,
                    matrices=matrices,
                    cost_cfg=cost_cfg,
                    weekday=weekday
                ): driver_id 
                for driver_id in base_drivers[:req.top_n]
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result and result.is_fully_feasible:
                    cascade_solutions.append(result)
        
        # Convert to API format
        final_candidates = [
            _enhanced_to_candidate_out(sol) 
            for sol in cascade_solutions[:max_candidates]
        ]
        
        # Build schedules for UI
        schedules = _compute_cascade_schedules(DATA, cascade_solutions[:max_candidates])
        
        # Calculate trip metrics
        if cascade_solutions:
            Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
            start_idx = idx_of(req.start_location, loc2idx)
            end_idx = idx_of(req.end_location, loc2idx)
            trip_minutes = float(Mtime[start_idx, end_idx])
            trip_miles = float(Mdist[start_idx, end_idx])
        else:
            trip_minutes, trip_miles = 0.0, 0.0
        
        print(f"[cascade] Generated {len(final_candidates)} candidates with {len(schedules)} schedules")
        return weekday, trip_minutes, trip_miles, final_candidates, schedules
        
    except Exception as e:
        print(f"[cascade] Generation failed: {e}")
        return weekday, 0.0, 0.0, [], []


# ============================================================================
# DRIVER FILTERING
# ============================================================================

def _enhanced_driver_filtering(
    req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    loc_meta: Dict[str, Any],
    weekday: str,
    req_min: int,
    sla_windows: Dict[int, Dict[str, int]],
    calling_point_proximity_miles: float = 50,
    max_drivers: int = 100
) -> List[str]:
    """Enhanced filtering with calling points proximity logic"""
    
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    start_idx = idx_of(req.start_location, loc2idx)
    
    # Service time window with SLA flexibility
    sla = sla_windows.get(int(req.priority), {"early_min": 60, "late_min": 60})
    earliest_service = req_min - sla["early_min"]
    latest_service = req_min + sla["late_min"]
    service_time_window = (earliest_service, latest_service)
    
    ds = DATA["driver_states"] or {}
    drivers = ds["drivers"] if isinstance(ds, dict) and "drivers" in ds else ds
    
    viable_candidates = []
    
    total_drivers = len(drivers)
    after_day_filter = 0
    after_nan_filter = 0  # NEW
    after_window_filter = 0
    after_calling_points_filter = 0
    
    # INVALID location names to exclude
    INVALID_LOCATIONS = {"NAN", "NO_DATA", "UNKNOWN", "NONE", "", "NULL", "N/A"}
    
    for duty_id, driver_meta in drivers.items():
        
        # Filter 1: Day-of-week
        elements_all = driver_meta.get("elements", []) or []
        elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
        if not elements:
            continue
        after_day_filter += 1
        
        # ✅ Filter 1.5: HARD FILTER - Exclude duties with NaN/invalid locations
        has_invalid_location = False
        for e in elements:
            if not e.get("is_travel", False):
                continue
            
            from_loc = str(e.get("from", "")).upper().strip()
            to_loc = str(e.get("to", "")).upper().strip()
            
            # Check if either location is invalid
            if from_loc in INVALID_LOCATIONS or to_loc in INVALID_LOCATIONS:
                has_invalid_location = True
                print(f"[filter] Driver {duty_id} EXCLUDED: has invalid location (from={from_loc}, to={to_loc})")
                break
        
        if has_invalid_location:
            continue
        after_nan_filter += 1
        
        # Filter 2: Cross-midnight duty window check
        if not _check_cross_midnight_duty_window(driver_meta, weekday, req_min, req.trip_minutes or 60):
            continue
        after_window_filter += 1
        
        # Filter 3: Calling points proximity and timing
        if not _check_calling_points_proximity(
            driver_meta, elements, start_idx, service_time_window,
            calling_point_proximity_miles, matrices, loc2idx, duty_id
        ):
            continue
        after_calling_points_filter += 1
        
        viable_candidates.append(duty_id)
        
        if len(viable_candidates) >= max_drivers:
            break
    
    print(f"[filter] Enhanced filtering: {total_drivers} → {after_day_filter} → {after_nan_filter} → {after_window_filter} → {after_calling_points_filter} candidates")
    return viable_candidates


def _check_cross_midnight_duty_window(
    driver_meta: Dict[str, Any], 
    weekday: str, 
    req_start_min: int, 
    service_duration_min: float
) -> bool:
    """Cross-midnight aware duty window checking"""
    
    daily_windows = driver_meta.get("daily_windows", {})
    if weekday not in daily_windows:
        return False
    
    window = daily_windows[weekday]
    duty_start = int(window.get("start_min", 0))
    duty_end = int(window.get("end_min", 1440))
    
    service_end_time = req_start_min + service_duration_min + 50
    
    if duty_end <= duty_start:
        # Cross-midnight duty
        duty_end_actual = duty_end + 24 * 60
        fits_same_day = (duty_start <= req_start_min <= 1440) and (req_start_min + service_duration_min + 50 <= 1440)
        fits_next_day = (0 <= req_start_min <= duty_end) and (service_end_time <= duty_end_actual)
        return fits_same_day or fits_next_day
    else:
        # Normal same-day duty
        return duty_start <= req_start_min and service_end_time <= duty_end


def _check_calling_points_proximity(
    driver_meta: Dict[str, Any],
    elements: List[Dict[str, Any]],
    service_start_idx: int,
    service_time_window: Tuple[int, int],
    max_proximity_miles: float,
    matrices: Dict[str, Any],
    loc2idx: Dict[str, int],
    driver_id: str
) -> bool:
    """Check if driver has calling points within proximity of service dispatch point"""
    
    Mdist = matrices["dist"]
    earliest_service_time, latest_service_time = service_time_window
    
    # Expand window for positioning
    window_buffer = 120  # 2 hours
    earliest_position_time = earliest_service_time - window_buffer
    latest_position_time = latest_service_time + window_buffer
    
    for element in elements:
        if not element.get("is_travel", False):
            continue
            
        element_start_time = element.get("start_min", 0)
        element_end_time = element.get("end_min", 0)
        
        # Check time window overlap
        if not (element_start_time <= latest_position_time and element_end_time >= earliest_position_time):
            continue
        
        # Check calling points proximity
        from_loc = str(element.get("from", "")).upper().strip()
        to_loc = str(element.get("to", "")).upper().strip()
        
        for loc_name in [from_loc, to_loc]:
            if loc_name in loc2idx:
                loc_idx = loc2idx[loc_name]
                distance_to_service = float(Mdist[loc_idx, service_start_idx])
                
                if distance_to_service <= max_proximity_miles:
                    print(f"[geo] Driver {driver_id} viable: calling point {loc_name} "
                          f"({distance_to_service:.1f}mi from service) during relevant time")
                    return True
    
    return False


# ============================================================================
# SINGLE-DRIVER CUOPT EVALUATION
# ============================================================================

def _evaluate_single_driver_with_cuopt(
    driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    weekday: str
) -> Optional[CascadeCandidateOut]:
    """
    Evaluate single driver using cuOpt optimization.
    """
    
    if not CUOPT_CLIENT_AVAILABLE:
        print(f"[cuopt] Client not available for {driver_id}")
        return None
    
    try:
        # CREATE THE CLIENT FIRST - THIS IS THE FIX
        cuopt_host = os.getenv("CUOPT_HOST", "cuopt")
        cuopt_client = CuOptServiceSelfHostClient(
            ip=cuopt_host,
            port=5000,
            polling_timeout=25,
            timeout_exception=False
        )
        
        # Build payload
        payload = _build_correct_cuopt_payload(
            [driver_id], new_trip_req, DATA, matrices, weekday
        )
        
        print(f"[cuopt] Solving for {driver_id}")
        
        # Solve
        solution = cuopt_client.get_optimized_routes(payload)
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=50)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:  # Success
                cost = solver_response.get("solution_cost", 0)
                print(f"[cuopt] SUCCESS for {driver_id}: cost={cost}")
                
                # Build result with schedule
                return _build_cascade_result_from_cuopt(
                    driver_id, new_trip_req, cost, DATA, matrices, weekday
                )
            else:
                print(f"[cuopt] Solver failed for {driver_id}: status={status}")
        else:
            print(f"[cuopt] No valid response for {driver_id}")
        
        return None
        
    except Exception as e:
        import traceback
        print(f"[cuopt] Exception for {driver_id}: {e}")
        print(f"[cuopt] Traceback: {traceback.format_exc()}")
        return None


def _build_cascade_result_from_cuopt(
    driver_id: str,
    new_trip_req: PlanRequest,
    cost: float,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str
) -> CascadeCandidateOut:
    """Build CascadeCandidateOut with RSL-aware reconstruction"""
    
    driver_meta = DATA["driver_states"]["drivers"].get(driver_id)
    if not driver_meta:
        raise ValueError(f"Driver {driver_id} not found")
    
    driver_elements = driver_meta.get("elements", [])
    active_elements = [e for e in driver_elements if element_active_on_weekday(e, weekday)]
    
    # Build before schedule
    before_schedule = []
    for element in active_elements:
        before_schedule.append({
            "index": len(before_schedule),
            "element_type": element.get("element_type", "TRAVEL"),
            "from": element.get("from", ""),
            "to": element.get("to", ""),
            "start_time": f"{element.get('start_min', 0)//60:02d}:{element.get('start_min', 0)%60:02d}",
            "end_time": f"{element.get('end_min', 0)//60:02d}:{element.get('end_min', 0)%60:02d}",
            "priority": element.get("priority", 3),
            "is_travel": element.get("is_travel", False),
            "start_min": element.get("start_min", 0),
            "end_min": element.get("end_min", 0),
            "changes": ""
        })
    
    # Use RSL-aware reconstruction
    task_map = _build_task_mapping(new_trip_req, displaced_work=None)
    structure = _identify_rsl_structure(before_schedule)
    
    # For single-driver, cuOpt just optimizes timing
    # Use task_ids from cuOpt response if available, otherwise default to duty_append
    task_ids = ["0"]  # Just the new service
    arrival_stamps = [minute_of_day_local(new_trip_req.when_local)]
    
    strategy = _determine_insertion_strategy(
        task_ids, arrival_stamps, task_map, structure, before_schedule, matrices
    )
    
    print(f"[rsl] Strategy: {strategy['type']}")
    
    # Reconstruct based on strategy
    if strategy['type'] == 'empty_replacement':
        after_schedule = _reconstruct_empty_replacement(
            strategy, before_schedule, new_trip_req, matrices, {}
        )
    elif strategy['type'] == 'duty_append':
        after_schedule = _reconstruct_duty_append(
            strategy, before_schedule, new_trip_req, structure, matrices, {}
        )
    else:
        # Fallback: duty append
        after_schedule = _reconstruct_duty_append(
            {"end_facility_index": len(before_schedule)}, 
            before_schedule, new_trip_req, structure, matrices, {}
        )
    
    return CascadeCandidateOut(
        candidate_id=f"CUOPT_{driver_id}",
        primary_driver_id=driver_id,
        total_system_cost=float(cost),
        drivers_affected=1,
        cascade_chain=[],
        before_after_schedules={
            driver_id: {
                "before": before_schedule,
                "after": after_schedule
            }
        },
        is_fully_feasible=True,
        uncovered_p4_tasks=[],
        disposed_p5_tasks=[]
        )

# ==============================================================================  
# RSL-AWARE RECONSTRUCTION (Restored from working version)
# ==============================================================================

def _build_task_mapping(
    new_trip_req: PlanRequest,
    displaced_work: List[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """Map cuOpt task IDs to actual work elements"""
    
    task_map = {}
    
    # Task "0" is always the new service
    task_map["0"] = {
        "type": "NEW_SERVICE",
        "from": new_trip_req.start_location,
        "to": new_trip_req.end_location,
        "priority": new_trip_req.priority,
        "duration": int(new_trip_req.trip_minutes or 60)
    }
    
    # For single-driver, we don't have displaced work yet
    if displaced_work:
        for i, task in enumerate(displaced_work, start=1):
            element = task.get("element", {})
            task_map[str(i)] = {
                "type": "DISPLACED_WORK",
                "from": element.get("from", ""),
                "to": element.get("to", ""),
                "priority": element.get("priority", 3),
                "duration": element.get("end_min", 0) - element.get("start_min", 0),
                "original_element": element
            }
    
    return task_map


def _identify_rsl_structure(schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify key RSL structural elements"""
    
    structure = {
        "start_facility": None,
        "end_facility": None,
        "as_directed_blocks": [],
        "meal_reliefs": [],
        "empty_legs": [],
        "home_base": None
    }
    
    for i, element in enumerate(schedule):
        et = str(element.get("element_type", "")).upper()
        
        if "START FACILITY" in et:
            structure["start_facility"] = {"index": i, "element": element}
            structure["home_base"] = element.get("from")
            
        elif "END FACILITY" in et:
            structure["end_facility"] = {"index": i, "element": element}
            
        elif "AS DIRECTED" in et:
            structure["as_directed_blocks"].append({"index": i, "element": element})
            
        elif "MEAL" in et or "RELIEF" in et:
            structure["meal_reliefs"].append({"index": i, "element": element})
            
        elif element.get("is_travel") and element.get("priority", 3) == 5:
            structure["empty_legs"].append({"index": i, "element": element})
    
    return structure


def _determine_insertion_strategy(
    task_ids: List[str],
    arrival_stamps: List[float],
    task_map: Dict[str, Dict[str, Any]],
    structure: Dict[str, Any],
    original_schedule: List[Dict[str, Any]],
    matrices: Dict[str, Any]
) -> Dict[str, Any]:
    """Determine how to insert cuOpt's route into the RSL duty"""
    
    # Skip depot visits (task_id might be empty string or "depot")
    work_tasks = [(tid, arr) for tid, arr in zip(task_ids, arrival_stamps) 
                  if tid and tid in task_map]
    
    if not work_tasks:
        return {"type": "duty_append", "new_service": task_map.get("0")}
    
    # Get the new service task
    new_service_task = next((tid for tid, _ in work_tasks if task_map[tid]["type"] == "NEW_SERVICE"), None)
    if not new_service_task:
        return {"type": "duty_append", "new_service": task_map.get("0")}
    
    new_service_arrival = next(arr for tid, arr in work_tasks if tid == new_service_task)
    new_service_info = task_map[new_service_task]
    
    # Strategy 1: Check for empty leg replacement
    for empty_leg in structure["empty_legs"]:
        elem = empty_leg["element"]
        if (elem.get("from", "").upper() == new_service_info["from"].upper() and
            elem.get("to", "").upper() == new_service_info["to"].upper() and
            abs(elem.get("start_min", 0) - new_service_arrival) < 60):
            return {
                "type": "empty_replacement",
                "target_index": empty_leg["index"],
                "target_element": elem,
                "new_service": new_service_info
            }
    
    # Strategy 2: Check for AS DIRECTED replacement
    for as_dir in structure["as_directed_blocks"]:
        elem = as_dir["element"]
        start = elem.get("start_min", 0)
        end = elem.get("end_min", 1440)
        if start <= new_service_arrival <= end:
            return {
                "type": "as_directed_replacement",
                "target_index": as_dir["index"],
                "target_element": elem,
                "new_service": new_service_info,
                "insertion_time": new_service_arrival
            }
    
    # Strategy 3: Default to duty append
    return {
        "type": "duty_append",
        "new_service": new_service_info,
        "end_facility_index": structure["end_facility"]["index"] if structure["end_facility"] else len(original_schedule)
    }


def _reconstruct_empty_replacement(
    strategy: Dict[str, Any],
    original_schedule: List[Dict[str, Any]],
    new_trip_req: PlanRequest,
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Replace an empty leg with loaded service"""
    
    reconstructed = [dict(e) for e in original_schedule]  # Deep copy
    target_idx = strategy["target_index"]
    target_elem = strategy["target_element"]
    
    reconstructed[target_idx] = {
        **target_elem,
        "priority": new_trip_req.priority,
        "load_type": "NEW_SERVICE",
        "planz_code": "DELIVERY",
        "changes": "EMPTY_REPLACED"
    }
    
    print(f"[rsl] ✅ Empty leg replaced with loaded service at index {target_idx}")
    return reconstructed


def _reconstruct_duty_append(
    strategy: Dict[str, Any],
    original_schedule: List[Dict[str, Any]],
    new_trip_req: PlanRequest,
    structure: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Append new work before END FACILITY with full RSL structure"""
    
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    
    home_base = structure.get("home_base")
    end_facility_index = strategy.get("end_facility_index", len(original_schedule))
    
    # Find last location before END FACILITY
    last_location = home_base
    for i in range(end_facility_index - 1, -1, -1):
        elem = original_schedule[i]
        if elem.get("is_travel"):
            last_location = elem.get("to") or elem.get("from")
            break
    
    # Start time
    if end_facility_index > 0:
        current_time = original_schedule[end_facility_index - 1].get("end_min", 0)
    else:
        current_time = 0
    
    service_from = new_trip_req.start_location.upper()
    service_to = new_trip_req.end_location.upper()
    
    append_sequence = []
    
    # 1. Deadhead to pickup
    if last_location and last_location.upper() != service_from:
        if last_location.upper() in loc2idx and service_from in loc2idx:
            deadhead_time = int(Mtime[loc2idx[last_location.upper()], loc2idx[service_from]])
            append_sequence.append({
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": last_location,
                "to": new_trip_req.start_location,
                "start_min": current_time,
                "end_min": current_time + deadhead_time,
                "priority": 5,
                "load_type": "EMPTY_DEADHEAD",
                "planz_code": "DEADHEAD",
                "changes": "DEADHEAD_ADDED"
            })
            current_time += deadhead_time
    
    # 2. Loading (30 min)
    append_sequence.append({
        "element_type": "LOAD/ASSIST",
        "is_travel": False,
        "from": new_trip_req.start_location,
        "to": new_trip_req.start_location,
        "start_min": current_time,
        "end_min": current_time + LOADING_TIME_MINUTES,
        "priority": new_trip_req.priority,
        "load_type": "LOADING",
        "planz_code": "LOAD_ASSIST",
        "changes": "LOADING_ADDED"
    })
    current_time += LOADING_TIME_MINUTES
    
    # 3. Service leg
    service_minutes = int(new_trip_req.trip_minutes or 60)
    append_sequence.append({
        "element_type": "TRAVEL",
        "is_travel": True,
        "from": new_trip_req.start_location,
        "to": new_trip_req.end_location,
        "start_min": current_time,
        "end_min": current_time + service_minutes,
        "priority": new_trip_req.priority,
        "load_type": "LOADED",
        "planz_code": "DELIVERY",
        "changes": "SERVICE_ADDED"
    })
    current_time += service_minutes
    
    # 4. Offloading (20 min)
    append_sequence.append({
        "element_type": "LOAD/ASSIST",
        "is_travel": False,
        "from": new_trip_req.end_location,
        "to": new_trip_req.end_location,
        "start_min": current_time,
        "end_min": current_time + OFFLOADING_TIME_MINUTES,
        "priority": new_trip_req.priority,
        "load_type": "OFFLOADING",
        "planz_code": "UNLOAD_ASSIST",
        "changes": "OFFLOADING_ADDED"
    })
    current_time += OFFLOADING_TIME_MINUTES
    
    # 5. Return to home base
    if home_base and service_to != home_base.upper():
        if service_to in loc2idx and home_base.upper() in loc2idx:
            return_time = int(Mtime[loc2idx[service_to], loc2idx[home_base.upper()]])
            append_sequence.append({
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": new_trip_req.end_location,
                "to": home_base,
                "start_min": current_time,
                "end_min": current_time + return_time,
                "priority": 5,
                "load_type": "RETURN_TO_BASE",
                "planz_code": "RETURN",
                "changes": "RETURN_ADDED"
            })
            current_time += return_time
    
    # Reconstruct
    reconstructed = []
    for i, elem in enumerate(original_schedule):
        if i < end_facility_index:
            reconstructed.append(elem)
        elif i == end_facility_index:
            reconstructed.extend(append_sequence)
            if "END FACILITY" in elem.get("element_type", "").upper():
                updated_end = dict(elem)
                updated_end["start_min"] = current_time
                updated_end["end_min"] = current_time + 15
                reconstructed.append(updated_end)
            else:
                reconstructed.append(elem)
        else:
            reconstructed.append(elem)
    
    # Re-index and format
    for idx, elem in enumerate(reconstructed):
        elem["index"] = idx
        if "start_min" in elem:
            elem["start_time"] = f"{int(elem['start_min'])//60:02d}:{int(elem['start_min'])%60:02d}"
        if "end_min" in elem:
            elem["end_time"] = f"{int(elem['end_min'])//60:02d}:{int(elem['end_min'])%60:02d}"
    
    print(f"[rsl] ✅ Duty append: {len(append_sequence)} elements added before END FACILITY")
    return reconstructed

# ============================================================================
# CUOPT CLIENT HELPERS
# ============================================================================

def _test_official_cuopt_client() -> bool:
    """Test cuOpt connectivity"""
    
    if not CUOPT_CLIENT_AVAILABLE:
        print("[test] Official cuOpt client not available")
        return False
    
    try:
        cuopt_host = os.getenv("CUOPT_HOST", "cuopt")
        cuopt_client = CuOptServiceSelfHostClient(
            ip=cuopt_host,
            port=5000,
            polling_timeout=10,
            timeout_exception=False
        )
        
        # Simple test from cuOpt docs
        test_data = {
            "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
            "task_data": {"task_locations": [0, 1]},
            "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]}
        }
        
        print("[test] Testing official cuOpt client...")
        solution = cuopt_client.get_optimized_routes(test_data)
        
        # Handle repoll if needed
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=10)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:
                cost = solver_response.get("solution_cost", 0)
                print(f"[test] Official client SUCCESS: status={status}, cost={cost}")
                return True
            else:
                print(f"[test] Official client solver failed: status={status}")
        else:
            print("[test] Official client returned no valid response")
            
        return False
        
    except Exception as e:
        print(f"[test] Official client test failed: {e}")
        return False


def _repoll_solution(cuopt_client, solution, repoll_tries=50):
    """
    Implement the repoll pattern from cuOpt documentation
    """
    
    # If solver is still busy, repoll using request ID
    if "reqId" in solution and "response" not in solution:
        req_id = solution["reqId"]
        print(f"[cuopt] Repolling reqId: {req_id}")
        
        for i in range(repoll_tries):
            try:
                solution = cuopt_client.repoll(req_id, response_type="dict")
                
                if "reqId" in solution and "response" in solution:
                    print(f"[cuopt] Repoll successful after {i+1} attempts")
                    break
                    
                # Sleep briefly between polls
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[cuopt] Repoll attempt {i+1} failed: {e}")
                time.sleep(0.5)
                continue
    
    return solution


def _build_correct_cuopt_payload(
    candidates: List[str],
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str
) -> Dict[str, Any]:
    """
    Build cuOpt payload with CORRECT cost matrix format for vehicle types
    """
    
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    
    start_idx = loc2idx.get(new_trip_req.start_location.upper())
    end_idx = loc2idx.get(new_trip_req.end_location.upper())
    
    if start_idx is None or end_idx is None:
        raise ValueError(f"Invalid locations: {new_trip_req.start_location} -> {new_trip_req.end_location}")
    
    # Simple 2-location cost matrix
    cost_matrix = [
        [0, int(Mdist[start_idx, end_idx])],
        [int(Mdist[end_idx, start_idx]), 0]
    ]
    
    # Build vehicle data for the first candidate
    if not candidates:
        raise ValueError("No candidates provided")
    
    primary_driver = candidates[0]
    driver_meta = DATA["driver_states"]["drivers"].get(primary_driver, {})
    elements = driver_meta.get("elements", [])
    active_elements = [e for e in elements if element_active_on_weekday(e, weekday)]
    
    if not active_elements:
        raise ValueError(f"No active elements for driver {primary_driver}")
    
    duty_start = min(e.get("start_min", 1440) for e in active_elements)
    duty_end = max(e.get("end_min", 0) for e in active_elements)
    
    payload = {
        "cost_matrix_data": {
            "data": {"1": cost_matrix}  # Matrix for vehicle type 1
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]],  # Vehicle starts at location 0
            "vehicle_time_windows": [[duty_start, duty_end]],
            "capacities": [[100]],
            "vehicle_types": [1]
        },
        "task_data": {
            "task_locations": [1],  # Task at location 1
            "task_time_windows": [[duty_start, duty_end]],
            "service_times": [int(new_trip_req.trip_minutes or 60)],
            "demand": [[1]]
        },
        "solver_config": {
            "time_limit": 10
        }
    }
    
    return payload


# ============================================================================
# UI FORMATTING
# ============================================================================

def _compute_cascade_schedules(
    DATA: Dict[str, Any], 
    cascade_solutions: List[CascadeCandidateOut]
) -> List[Dict[str, Any]]:
    """
    Convert cascade solutions to schedule format expected by UI
    """
    
    all_schedules = []
    
    for cascade_solution in cascade_solutions:
        # Convert each cascade solution's schedules to UI format
        ui_schedules = _build_ui_schedules_from_cascade(cascade_solution, DATA)
        all_schedules.extend(ui_schedules)
    
    return all_schedules


def _build_ui_schedules_from_cascade(
    cascade_result: CascadeCandidateOut,
    DATA: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert CascadeCandidateOut to the format expected by the UI
    """
    
    ui_schedules = []
    
    for driver_id, schedules in cascade_result.before_after_schedules.items():
        ui_schedule = {
            "driver_id": driver_id,
            "before": schedules["before"],
            "after": schedules["after"]
        }
        ui_schedules.append(ui_schedule)
    
    return ui_schedules


def _enhanced_to_candidate_out(cascade_result: CascadeCandidateOut) -> CandidateOut:
    """
    Enhanced conversion with better details for UI display
    """
    
    # Build detailed reason string
    drivers_text = f"{cascade_result.drivers_affected} driver" + ("s" if cascade_result.drivers_affected != 1 else "")
    cost_text = f"£{cascade_result.total_system_cost:.2f}"
    
    if cascade_result.cascade_chain:
        chain_summary = " → ".join([step.get("vehicle_id", "?") for step in cascade_result.cascade_chain])
        reason = f"Multi-driver cascade: {chain_summary} ({drivers_text}, {cost_text})"
    else:
        reason = f"cuOpt optimized ({drivers_text}, {cost_text})"
    
    return CandidateOut(
        candidate_id=cascade_result.candidate_id,
        driver_id=cascade_result.primary_driver_id,
        type="cascade_optimized",
        est_cost=cascade_result.total_system_cost,
        deadhead_miles=0.0,
        delay_minutes=0.0,
        overtime_minutes=0.0,
        uses_emergency_rest=False,
        miles_delta=0.0,
        feasible_hard=cascade_result.is_fully_feasible,
        reason=reason
    )

#         ui_schedules = _build_ui_schedules_from_cascade(cascade_solution, DATA)
#         all_schedules.extend(ui_schedules)
    
#     return all_schedules

# def generate_cascade_candidates_with_schedules(
#     req: PlanRequest,
#     DATA: Dict[str,Any],
#     matrices: Dict[str,Any],
#     cost_cfg: Dict[str,float],
#     loc_meta: Dict[str,Any],
#     sla_windows: Dict[int, Dict[str,int]],
#     max_cascade_depth: int = 2,
#     max_candidates: int = 10,
# ) -> Tuple[str, float, float, List[CandidateOut], List[Dict[str, Any]]]:
#     """
#     Enhanced version that returns both candidates AND schedule data for UI
#     """
    
#     weekday = weekday_from_local(req.when_local)
#     req_min = minute_of_day_local(req.when_local)
    
#     # Test official cuOpt client on first run
#     cuopt_working = _test_official_cuopt_client()
#     cuopt_status = "OFFICIAL_CLIENT" if cuopt_working else "HEURISTIC_FALLBACK"
    
#     print(f"[cascade] Generating candidates for {req.start_location}→{req.end_location} (cuOpt: {cuopt_status})")
    
#     try:
#         base_driver_candidates = _enhanced_driver_filtering(
#             req, DATA, matrices, loc_meta, weekday, req_min, sla_windows,
#             calling_point_proximity_miles=50,
#             max_drivers=20
#         )
        
#         if not base_driver_candidates:
#             print("[cascade] No viable base drivers found")
#             return weekday, 0.0, 0.0, [], []
        
#         cascade_solutions = []

#         with ThreadPoolExecutor(max_workers=5) as executor:
#             futures = {
#                 executor.submit(
#                     _evaluate_cascade_scenario,
#                     base_driver_id=driver_id,      # Changed to keyword arg
#                     new_trip_req=req,               # Changed to match old code
#                     DATA=DATA,
#                     matrices=matrices,
#                     cost_cfg=cost_cfg,
#                     weekday=weekday,
#                     sla_windows=sla_windows,
#                     max_cascade_depth=max_cascade_depth
#                 ): driver_id 
#                 for driver_id in base_driver_candidates[:req.top_n]
#             }
            
#             for future in as_completed(futures):
#                 result = future.result()
#                 if result and result.is_fully_feasible:
#                     cascade_solutions.append(result)
        
#         # Convert to UI format
#         final_candidates = []
#         for solution in cascade_solutions[:max_candidates]:
#             candidate = _enhanced_to_candidate_out(solution)
#             final_candidates.append(candidate)
        
#         # Build schedule data for UI
#         schedules = _compute_cascade_schedules(DATA, cascade_solutions[:max_candidates])
        
#         # Calculate trip details
#         if cascade_solutions:
#             Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
#             start_idx = idx_of(req.start_location, loc2idx)
#             end_idx = idx_of(req.end_location, loc2idx)
#             trip_minutes = float(Mtime[start_idx, end_idx])
#             trip_miles = float(Mdist[start_idx, end_idx])
#         else:
#             trip_minutes, trip_miles = 0.0, 0.0
        
#         print(f"[cascade] Generated {len(final_candidates)} candidates with {len(schedules)} schedule entries")
#         return weekday, trip_minutes, trip_miles, final_candidates, schedules
        
#     except Exception as e:
#         print(f"[cascade] Generation failed: {e}")
#         return weekday, 0.0, 0.0, [], []
    
# def _enhanced_to_candidate_out(cascade_result: CascadeCandidateOut) -> CandidateOut:
#     """
#     Enhanced conversion with better details for UI display
#     """
    
#     # Build detailed reason string
#     drivers_text = f"{cascade_result.drivers_affected} driver" + ("s" if cascade_result.drivers_affected != 1 else "")
#     cost_text = f"£{cascade_result.total_system_cost:.2f}"
    
#     if cascade_result.cascade_chain:
#         chain_summary = " → ".join([step.get("vehicle_id", "?") for step in cascade_result.cascade_chain])
#         reason = f"Multi-driver cascade: {chain_summary} ({drivers_text}, {cost_text})"
#     else:
#         reason = f"Enhanced cascade ({drivers_text}, {cost_text})"
    
#     return CandidateOut(
#         candidate_id=cascade_result.candidate_id,
#         driver_id=cascade_result.primary_driver_id,
#         type="cascade_optimized",
#         est_cost=cascade_result.total_system_cost,
#         deadhead_miles=0.0,  # TODO: Extract from cascade_chain
#         delay_minutes=0.0,
#         overtime_minutes=0.0,
#         uses_emergency_rest=False,
#         miles_delta=0.0,
#         feasible_hard=cascade_result.is_fully_feasible,
#         reason=reason
#     )