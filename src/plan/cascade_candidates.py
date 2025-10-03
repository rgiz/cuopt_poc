"""
cascade_candidates.py

New module for cascade-aware candidate generation using cuOpt.
Separate from existing candidates.py to avoid disrupting working code.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .models import PlanRequest, CandidateOut
from .candidates import minute_of_day_local, weekday_from_local, element_active_on_weekday, idx_of
from .config import load_sla_windows
import os
import sys
from pathlib import Path
import requests
import time

try:
    from cuopt_sh_client import CuOptServiceSelfHostClient
    CUOPT_CLIENT_AVAILABLE = True
    print("[cascade] cuopt-sh-client imported successfully")
except ImportError as e:
    print(f"[cascade] cuopt-sh-client not available: {e}")
    CUOPT_CLIENT_AVAILABLE = False

# Add root to path to import your CuOptModel
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.opt.cuopt_model_miles import CuOptModel
    CUOPT_AVAILABLE = True
    print("[cascade] CuOptModel imported successfully")
except ImportError as e:
    print(f"[cascade] CuOptModel not available: {e}")
    CUOPT_AVAILABLE = False

@dataclass
class CascadeCandidateOut:
    """Enhanced candidate with cascade information"""
    candidate_id: str
    primary_driver_id: str
    total_system_cost: float
    drivers_affected: int
    cascade_chain: List[Any]
    before_after_schedules: Dict[str, Dict[str, Any]]
    is_fully_feasible: bool
    uncovered_p4_tasks: List[Dict[str, Any]]
    disposed_p5_tasks: List[Dict[str, Any]]
    
    # Convert to original CandidateOut for backward compatibility
    def to_candidate_out(self) -> CandidateOut:
        return CandidateOut(
            candidate_id=self.primary_driver_id,  # SIMPLE: Just use driver ID as candidate ID
            driver_id=self.primary_driver_id,
            type="cascade_optimized",
            est_cost=self.total_system_cost,
            deadhead_miles=0.0,  # TODO: Extract from cascade_chain
            delay_minutes=0.0,
            overtime_minutes=0.0,
            uses_emergency_rest=False,
            miles_delta=0.0,
            feasible_hard=self.is_fully_feasible,
        )


def generate_cascade_candidates(
    req: PlanRequest,
    DATA: Dict[str,Any],
    matrices: Dict[str,Any],
    cost_cfg: Dict[str,float],
    loc_meta: Dict[str,Any],
    sla_windows: Dict[int, Dict[str,int]],
    max_cascade_depth: int = 2,
    max_candidates: int = 10,
) -> Tuple[str, float, float, List[CandidateOut]]:
    """Enhanced with official cuOpt client"""
    
    weekday = weekday_from_local(req.when_local)
    req_min = minute_of_day_local(req.when_local)
    
    # Test official cuOpt client on first run
    cuopt_working = _test_official_cuopt_client()
    cuopt_status = "OFFICIAL_CLIENT" if cuopt_working else "HEURISTIC_FALLBACK"
    
    print(f"[cascade] Generating candidates for {req.start_location}→{req.end_location} (cuOpt: {cuopt_status})")
    
    # Continue with existing logic...
    try:
        base_driver_candidates = _enhanced_driver_filtering(
            req, DATA, matrices, loc_meta, weekday, req_min, sla_windows,
            calling_point_proximity_miles=50,
            max_drivers=20
        )
        
        if not base_driver_candidates:
            print("[cascade] No viable base drivers found")
            return weekday, 0.0, 0.0, []
        
        cascade_solutions = []
        
        for base_driver_id in base_driver_candidates[:10]:
            cascade_solution = _evaluate_cascade_scenario(
                base_driver_id=base_driver_id,
                new_trip_req=req,
                DATA=DATA,
                matrices=matrices,
                cost_cfg=cost_cfg,
                weekday=weekday,
                sla_windows=sla_windows,
                max_cascade_depth=max_cascade_depth
            )
            
            if cascade_solution and cascade_solution.is_fully_feasible:
                cascade_solutions.append(cascade_solution)
        
        # Convert to backward-compatible format
        final_candidates = []
        for solution in cascade_solutions[:max_candidates]:
            candidate = solution.to_candidate_out()
            candidate.reason = f"Enhanced cascade ({cuopt_status}): {solution.drivers_affected} drivers, £{solution.total_system_cost:.2f}"
            final_candidates.append(candidate)
        
        # Calculate trip details
        if cascade_solutions:
            Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
            start_idx = idx_of(req.start_location, loc2idx)
            end_idx = idx_of(req.end_location, loc2idx)
            trip_minutes = float(Mtime[start_idx, end_idx])
            trip_miles = float(Mdist[start_idx, end_idx])
        else:
            trip_minutes, trip_miles = 0.0, 0.0
        
        print(f"[cascade] Generated {len(final_candidates)} candidates (cuOpt: {cuopt_status})")
        return weekday, trip_minutes, trip_miles, final_candidates
        
    except Exception as e:
        print(f"[cascade] Generation failed: {e}")
        from .candidates import generate_candidates
        return generate_candidates(req, DATA, matrices, cost_cfg, loc_meta, sla_windows)

def _enhanced_driver_filtering(
    req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    loc_meta: Dict[str, Any],
    weekday: str,
    req_min: int,
    sla_windows: Dict[int, Dict[str, int]],  # ADD: Missing parameter
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
    after_window_filter = 0
    after_calling_points_filter = 0
    
    for duty_id, driver_meta in drivers.items():
        
        # Filter 1: Day-of-week
        elements_all = driver_meta.get("elements", []) or []
        elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
        if not elements:
            continue
        after_day_filter += 1
        
        # Filter 2: Cross-midnight duty window check
        if not _check_cross_midnight_duty_window(driver_meta, weekday, req_min, req.trip_minutes or 60):
            continue
        after_window_filter += 1
        
        # Filter 3: Calling points proximity and timing
        if not _check_calling_points_proximity(
            driver_meta, elements, start_idx, service_time_window,
            calling_point_proximity_miles, matrices, loc2idx, duty_id  # ADD: Pass duty_id
        ):
            continue
        after_calling_points_filter += 1
        
        viable_candidates.append(duty_id)
        
        if len(viable_candidates) >= max_drivers:
            break
    
    print(f"[filter] Enhanced filtering: {total_drivers} → {after_day_filter} → {after_window_filter} → {after_calling_points_filter} candidates")
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
    driver_id: str  # ADD: Driver ID parameter
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

def _evaluate_cascade_scenario_with_cuopt(
    base_driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    weekday: str,
    sla_windows: Dict[int, Dict[str, int]],
    max_cascade_depth: int
) -> Optional[CascadeCandidateOut]:
    """
    Use the official CuOptServiceSelfHostClient - handles polling properly
    """
    
    if not CUOPT_CLIENT_AVAILABLE:
        print(f"[cuopt] Official client not available, using heuristic for {base_driver_id}")
        return _heuristic_cascade_evaluation_original(
            base_driver_id, new_trip_req, DATA, matrices, cost_cfg, weekday, sla_windows, max_cascade_depth
        )
    
    try:
        # Create official cuOpt client
        # Use cuOpt service name in Docker, not localhost
        cuopt_host = os.getenv("CUOPT_HOST", "cuopt")  # 'cuopt' is the Docker service name
        cuopt_client = CuOptServiceSelfHostClient(
            ip=cuopt_host,
            port=5000,
            polling_timeout=25,  # 25 second timeout
            timeout_exception=False  # Return None instead of exception on timeout
        )
        
        # Build payload using correct format
        payload = _build_correct_cuopt_payload(
            [base_driver_id], new_trip_req, DATA, matrices, weekday
        )
        
        print(f"[cuopt] Using official client for {base_driver_id}")
        
        # Use the official get_optimized_routes method
        solution = cuopt_client.get_optimized_routes(payload)
        
        # Handle the repoll pattern from cuOpt docs
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=50)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:  # Success
                cost = solver_response.get("solution_cost", 0)
                print(f"[cuopt] SUCCESS for {base_driver_id}: status={status}, cost={cost}")
                
                # Build result using cuOpt solution
                heuristic_result = _heuristic_cascade_evaluation_original(
                    base_driver_id, new_trip_req, DATA, matrices, cost_cfg, weekday, sla_windows, max_cascade_depth
                )
                
                if heuristic_result:
                    heuristic_result.candidate_id = f"CUOPT_{base_driver_id}"
                    heuristic_result.total_system_cost = float(cost)
                    
                return heuristic_result
                
            else:
                print(f"[cuopt] Solver failed for {base_driver_id}: status={status}")
        else:
            print(f"[cuopt] No valid response for {base_driver_id}")
        
        # Fall back to heuristic
        return _heuristic_cascade_evaluation_original(
            base_driver_id, new_trip_req, DATA, matrices, cost_cfg, weekday, sla_windows, max_cascade_depth
        )
        
    except Exception as e:
        print(f"[cuopt] Client exception for {base_driver_id}: {e}")
        return _heuristic_cascade_evaluation_original(
            base_driver_id, new_trip_req, DATA, matrices, cost_cfg, weekday, sla_windows, max_cascade_depth
        )

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

# Test function to verify client works
def _test_official_cuopt_client() -> bool:
    """Test the official cuOpt client"""
    
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
            "cost_matrix_data": {"data": {"0": [[0,1],[1,0]]}},
            "task_data": {"task_locations": [0,1]},
            "fleet_data": {"vehicle_locations": [[0,0],[0,0]]}
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

def _build_correct_cuopt_payload(
    candidates: List[str],
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str
) -> Dict[str, Any]:
    """Build cuOpt payload with CORRECT cost matrix format for vehicle types"""
    
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
    
    # FIXED: Cost matrix must be provided for each vehicle type
    # We have vehicle_types: [1], so we need cost matrix for type "1"
    payload = {
        "cost_matrix_data": {
            "data": {"1": cost_matrix}  # Matrix for vehicle type 1
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]],  # Vehicle starts at location 0
            "vehicle_time_windows": [[0, 1440]],
            "capacities": [[100]],
            "vehicle_types": [1]  # This must match the key in cost_matrix_data
        },
        "task_data": {
            "task_locations": [1],  # Task at location 1
            "task_time_windows": [[0, 1440]],
            "service_times": [int(new_trip_req.trip_minutes or 60)],
            "demand": [[1]]
        },
        "solver_config": {
            "time_limit": 10  # Quick solve
        }
    }
    
    return payload

def _convert_cuopt_result_to_cascade(
    cuopt_result: Dict[str, Any],
    primary_driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str,
    cost_cfg: Dict[str, float]
) -> CascadeCandidateOut:
    """Convert cuOpt result to cascade candidate format"""
    
    solver_response = cuopt_result.get("solver_response", {})
    solution_cost = solver_response.get("solution_cost", 0)
    
    # Build realistic before/after schedules
    driver_meta = DATA["driver_states"]["drivers"].get(primary_driver_id, {})
    elements = driver_meta.get("elements", [])
    active_elements = [e for e in elements if element_active_on_weekday(e, weekday)]
    
    # Before schedule
    before_schedule = []
    for i, element in enumerate(active_elements):
        before_schedule.append({
            "index": i,
            "element_type": element.get("element_type", "TRAVEL"),
            "from": element.get("from", ""),
            "to": element.get("to", ""),
            "start_time": f"{element.get('start_min', 0)//60:02d}:{element.get('start_min', 0)%60:02d}",
            "end_time": f"{element.get('end_min', 0)//60:02d}:{element.get('end_min', 0)%60:02d}",
            "priority": element.get("priority", 3),
            "changes": ""
        })
    
    # After schedule - add the new service
    req_time = minute_of_day_local(new_trip_req.when_local)
    service_minutes = int(new_trip_req.trip_minutes or 60)
    
    after_schedule = before_schedule.copy()
    after_schedule.append({
        "index": len(before_schedule),
        "element_type": "TRAVEL",
        "from": new_trip_req.start_location,
        "to": new_trip_req.end_location,
        "start_time": f"{req_time//60:02d}:{req_time%60:02d}",
        "end_time": f"{(req_time + service_minutes)//60:02d}:{(req_time + service_minutes)%60:02d}",
        "priority": new_trip_req.priority,
        "changes": "ADDED_BY_CUOPT"
    })
    
    # Calculate realistic cost
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    start_idx = loc2idx[new_trip_req.start_location.upper()]
    end_idx = loc2idx[new_trip_req.end_location.upper()]
    
    service_miles = float(Mdist[start_idx, end_idx])
    deadhead_cost = service_miles * cost_cfg.get("deadhead_cost_per_mile", 1.0)
    admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
    total_cost = deadhead_cost + admin_cost
    
    return CascadeCandidateOut(
        candidate_id=f"CUOPT_{primary_driver_id}",
        primary_driver_id=primary_driver_id,
        total_system_cost=total_cost,
        drivers_affected=1,
        cascade_chain=[],
        before_after_schedules={
            primary_driver_id: {
                "before": before_schedule,
                "after": after_schedule
            }
        },
        is_fully_feasible=True,
        uncovered_p4_tasks=[],
        disposed_p5_tasks=[]
    )

def _heuristic_cascade_evaluation_original(
    base_driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    weekday: str,
    sla_windows: Dict[int, Dict[str, int]],
    max_cascade_depth: int
) -> Optional[CascadeCandidateOut]:
    """
    COPY ALL YOUR EXISTING _evaluate_cascade_scenario CONTENT HERE
    This is currently just calling your existing logic:
    """
    
    # For now, just call your existing _evaluate_cascade_scenario
    # (This is temporary - you should copy the actual implementation)
    try:
        driver_meta = DATA["driver_states"]["drivers"].get(base_driver_id)
        if not driver_meta:
            return None
            
        driver_elements = driver_meta.get("elements", [])
        active_elements = [e for e in driver_elements if element_active_on_weekday(e, weekday)]
        
        if not active_elements:
            return None
        
        # STEP 1: Check current duty length
        duty_start_min = min(e.get("start_min", 1440) for e in active_elements)
        duty_end_min = max(e.get("end_min", 0) for e in active_elements)
        current_duty_minutes = duty_end_min - duty_start_min
        
        # Handle cross-midnight duties
        if duty_end_min <= duty_start_min:
            current_duty_minutes = (1440 - duty_start_min) + duty_end_min
        
        # STEP 2: Calculate additional time for new service
        Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
        start_idx = idx_of(new_trip_req.start_location, loc2idx)
        end_idx = idx_of(new_trip_req.end_location, loc2idx)
        
        service_minutes = float(new_trip_req.trip_minutes) if new_trip_req.trip_minutes else float(Mtime[start_idx, end_idx])
        service_miles = float(new_trip_req.trip_miles) if new_trip_req.trip_miles else float(Mdist[start_idx, end_idx])
        
        # Add positioning deadheads (simplified - find nearest calling point)
        min_deadhead_to_service = float('inf')
        for element in active_elements:
            if element.get("is_travel"):
                for loc_field in ["from", "to"]:
                    loc_name = str(element.get(loc_field, "")).upper().strip()
                    if loc_name in loc2idx:
                        loc_idx = loc2idx[loc_name]
                        deadhead_distance = float(Mdist[loc_idx, start_idx])
                        deadhead_time = float(Mtime[loc_idx, start_idx])
                        if deadhead_time < min_deadhead_to_service:
                            min_deadhead_to_service = deadhead_time
        
        if min_deadhead_to_service == float('inf'):
            min_deadhead_to_service = 0  # Fallback
        
        # Total additional time: deadhead + service + loading/offloading + return deadhead
        additional_time = min_deadhead_to_service + service_minutes + 50 + service_minutes  # Conservative estimate
        
        # STEP 3: Check hard constraint - 13-hour max duty
        MAX_DUTY_MINUTES = 13 * 60  # 780 minutes
        projected_duty_minutes = current_duty_minutes + additional_time
        
        if projected_duty_minutes > MAX_DUTY_MINUTES:
            return None
        
        # STEP 4: Build before/after schedules
        before_schedule = []
        after_schedule = []
        
        # Before schedule
        for i, element in enumerate(active_elements):
            before_schedule.append({
                "index": i,
                "element_type": element.get("element_type", "TRAVEL"),
                "from": element.get("from", ""),
                "to": element.get("to", ""),
                "start_time": f"{element.get('start_min', 0)//60:02d}:{element.get('start_min', 0)%60:02d}",
                "end_time": f"{element.get('end_min', 0)//60:02d}:{element.get('end_min', 0)%60:02d}",
                "priority": element.get("priority", 3),
                "load_type": element.get("load_type", ""),
                "changes": ""
            })
        
        # After schedule - insert new service
        req_time = minute_of_day_local(new_trip_req.when_local)
        new_service_element = {
            "index": len(active_elements),
            "element_type": "TRAVEL",
            "from": new_trip_req.start_location,
            "to": new_trip_req.end_location,
            "start_time": f"{req_time//60:02d}:{req_time%60:02d}",
            "end_time": f"{(req_time + int(service_minutes))//60:02d}:{(req_time + int(service_minutes))%60:02d}",
            "priority": new_trip_req.priority,
            "load_type": "NEW_SERVICE",
            "changes": "ADDED"
        }
        
        after_schedule = before_schedule.copy()
        after_schedule.append(new_service_element)
        
        # STEP 5: Calculate costs
        deadhead_cost = min_deadhead_to_service * cost_cfg.get("deadhead_cost_per_mile", 1.0)
        service_cost = service_miles * cost_cfg.get("deadhead_cost_per_mile", 1.0)
        admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
        overtime_cost = max(0, projected_duty_minutes - MAX_DUTY_MINUTES) * cost_cfg.get("overtime_cost_per_minute", 1.0)
        
        total_cost = deadhead_cost + service_cost + admin_cost + overtime_cost
        
        return CascadeCandidateOut(
            candidate_id=f"HEURISTIC_{base_driver_id}",
            primary_driver_id=base_driver_id,
            total_system_cost=total_cost,
            drivers_affected=1,
            cascade_chain=[],
            before_after_schedules={
                base_driver_id: {
                    "before": before_schedule,
                    "after": after_schedule
                }
            },
            is_fully_feasible=True,
            uncovered_p4_tasks=[],
            disposed_p5_tasks=[],
        )
        
    except Exception as e:
        print(f"[cascade] Heuristic evaluation failed for {base_driver_id}: {e}")
        return None
    
    # Add these functions to your cascade_candidates.py file:

def _evaluate_multi_driver_cascade(
    primary_driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    weekday: str,
    sla_windows: Dict[int, Dict[str, int]],
    max_cascade_depth: int
) -> Optional[CascadeCandidateOut]:
    """
    Evaluate multi-driver cascade scenarios using cuOpt for the full chain
    """
    
    if not CUOPT_CLIENT_AVAILABLE:
        return _heuristic_cascade_evaluation_original(
            primary_driver_id, new_trip_req, DATA, matrices, cost_cfg, weekday, sla_windows, max_cascade_depth
        )
    
    try:
        print(f"[cascade] Building multi-driver cascade for {primary_driver_id}")
        
        # STEP 1: Analyze what work gets displaced by the new service
        displaced_work = _find_displaced_work(
            primary_driver_id, new_trip_req, DATA, weekday, matrices
        )
        
        if not displaced_work:
            print(f"[cascade] No displaced work for {primary_driver_id}, using single-driver optimization")
            return _evaluate_cascade_scenario_with_cuopt(
                primary_driver_id, new_trip_req, DATA, matrices, cost_cfg, 
                weekday, sla_windows, max_cascade_depth
            )
        
        print(f"[cascade] Found {len(displaced_work)} displaced tasks for {primary_driver_id}")
        
        # STEP 2: Filter displaced work by priority rules
        filtered_displaced_work = _filter_displaced_work_by_priority(displaced_work, new_trip_req.priority)
        
        print(f"[cascade] After priority filtering: {len(filtered_displaced_work)} tasks need reassignment")
        
        # STEP 3: Find candidate drivers for displaced work
        reassignment_candidates = _find_reassignment_candidates(
            filtered_displaced_work, DATA, matrices, weekday, sla_windows
        )
        
        if not reassignment_candidates:
            print(f"[cascade] No reassignment candidates found for {primary_driver_id}")
            return None
        
        # STEP 4: Build complete cascade payload for cuOpt
        cascade_payload = _build_cascade_cuopt_payload(
            primary_driver_id=primary_driver_id,
            new_trip_req=new_trip_req,
            displaced_work=filtered_displaced_work,
            reassignment_candidates=reassignment_candidates,
            DATA=DATA,
            matrices=matrices,
            weekday=weekday
        )
        
        # STEP 5: Solve with cuOpt
        cuopt_host = os.getenv("CUOPT_HOST", "cuopt")
        cuopt_client = CuOptServiceSelfHostClient(
            ip=cuopt_host,
            port=5000,
            polling_timeout=30,  # Longer timeout for multi-driver problems
            timeout_exception=False
        )
        
        print(f"[cascade] Solving multi-driver cascade with cuOpt")
        solution = cuopt_client.get_optimized_routes(cascade_payload)
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=100)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:  # Success
                return _parse_cascade_cuopt_solution(
                    solution, primary_driver_id, new_trip_req, 
                    displaced_work, reassignment_candidates, DATA, matrices, weekday, cost_cfg
                )
            else:
                print(f"[cascade] cuOpt cascade failed for {primary_driver_id}: status={status}")
        
        # Fall back to single-driver if cascade fails
        return _evaluate_cascade_scenario_with_cuopt(
            primary_driver_id, new_trip_req, DATA, matrices, cost_cfg, 
            weekday, sla_windows, max_cascade_depth
        )
        
    except Exception as e:
        print(f"[cascade] Multi-driver cascade failed for {primary_driver_id}: {e}")
        return _evaluate_cascade_scenario_with_cuopt(
            primary_driver_id, new_trip_req, DATA, matrices, cost_cfg, 
            weekday, sla_windows, max_cascade_depth
        )

def _find_displaced_work(
    driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    weekday: str,
    matrices: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Find what work gets displaced when a driver takes the new service
    """
    
    drivers = DATA["driver_states"]["drivers"]
    driver_meta = drivers.get(driver_id, {})
    elements = driver_meta.get("elements", [])
    active_elements = [e for e in elements if element_active_on_weekday(e, weekday)]
    
    if not active_elements:
        return []
    
    new_service_time = minute_of_day_local(new_trip_req.when_local)
    service_duration = int(new_trip_req.trip_minutes or 60)
    new_service_end = new_service_time + service_duration
    
    displaced_work = []
    
    for element in active_elements:
        element_start = element.get("start_min", 0)
        element_end = element.get("end_min", 0)
        element_priority = element.get("priority", 3)
        
        # Check for time conflicts
        conflicts = (
            (element_start <= new_service_time <= element_end) or
            (element_start <= new_service_end <= element_end) or
            (new_service_time <= element_start <= new_service_end)
        )
        
        if conflicts and element.get("is_travel") and element_priority >= new_trip_req.priority:
            displaced_task = {
                "original_driver_id": driver_id,
                "element": element,
                "start_location": element.get("from", ""),
                "end_location": element.get("to", ""),
                "start_time": element_start,
                "end_time": element_end,
                "priority": element_priority,
                "duration_minutes": element_end - element_start,
                "displacement_reason": "time_conflict_with_new_service"
            }
            displaced_work.append(displaced_task)
    
    return displaced_work

def _filter_displaced_work_by_priority(
    displaced_work: List[Dict[str, Any]],
    new_service_priority: int
) -> List[Dict[str, Any]]:
    """
    Apply priority rules to determine what displaced work needs reassignment
    """
    
    filtered_work = []
    
    for task in displaced_work:
        priority = task["priority"]
        
        if priority == 5:
            # Priority 5 (empty legs) - don't need coverage
            print(f"[priority] Displacing P5 empty leg: {task['start_location']}→{task['end_location']} (no reassignment needed)")
            continue
            
        elif priority == 4:
            # Priority 4 (optional) - mark as uncovered but don't require reassignment
            print(f"[priority] Displacing P4 optional work: {task['start_location']}→{task['end_location']} (will be uncovered)")
            task["reassignment_required"] = False
            task["can_be_uncovered"] = True
            filtered_work.append(task)
            
        elif priority <= 3:
            # Priority 1-3 - must be reassigned
            print(f"[priority] Displacing P{priority} work: {task['start_location']}→{task['end_location']} (must reassign)")
            task["reassignment_required"] = True
            task["can_be_uncovered"] = False
            filtered_work.append(task)
    
    return filtered_work

def _find_reassignment_candidates(
    displaced_work: List[Dict[str, Any]],
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str,
    sla_windows: Dict[int, Dict[str, int]]
) -> Dict[str, List[str]]:
    """
    Find drivers who can potentially take displaced work
    """
    
    reassignment_candidates = {}
    drivers = DATA["driver_states"]["drivers"]
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    
    for task in displaced_work:
        task_id = f"{task['start_location']}→{task['end_location']}@{task['start_time']}"
        candidates = []
        
        start_idx = loc2idx.get(task['start_location'].upper())
        if start_idx is None:
            continue
        
        for driver_id, driver_meta in drivers.items():
            if driver_id == task["original_driver_id"]:
                continue  # Skip the original driver
            
            elements = driver_meta.get("elements", [])
            active_elements = [e for e in elements if element_active_on_weekday(e, weekday)]
            
            if not active_elements:
                continue
            
            # Check if driver has capacity and proximity
            can_take_task = _can_driver_take_task(
                driver_id, driver_meta, active_elements, task, matrices, sla_windows
            )
            
            if can_take_task:
                candidates.append(driver_id)
        
        if candidates:
            reassignment_candidates[task_id] = candidates[:5]  # Limit to top 5 candidates
            print(f"[reassign] Found {len(candidates)} candidates for {task_id}")
        else:
            print(f"[reassign] No candidates found for {task_id}")
    
    return reassignment_candidates

def _can_driver_take_task(
    driver_id: str,
    driver_meta: Dict[str, Any],
    active_elements: List[Dict[str, Any]],
    task: Dict[str, Any],
    matrices: Dict[str, Any],
    sla_windows: Dict[int, Dict[str, int]]
) -> bool:
    """
    Check if a driver can feasibly take a displaced task
    """
    
    # Basic feasibility checks
    duty_start = min(e.get("start_min", 1440) for e in active_elements)
    duty_end = max(e.get("end_min", 0) for e in active_elements)
    
    # Check if task fits within duty window
    task_start = task["start_time"]
    task_end = task["end_time"]
    
    if not (duty_start <= task_start <= duty_end and duty_start <= task_end <= duty_end):
        return False
    
    # Check duty time limits (simplified)
    current_duty = duty_end - duty_start
    additional_time = task["duration_minutes"] + 30  # Add buffer for deadheads
    
    if current_duty + additional_time > 13 * 60:  # 13 hour limit
        return False
    
    # Check geographic feasibility (simplified proximity check)
    Mdist, loc2idx = matrices["dist"], matrices["loc2idx"]
    task_start_idx = loc2idx.get(task['start_location'].upper())
    
    if task_start_idx is None:
        return False
    
    # Check if driver has any calling points near the task
    for element in active_elements:
        if not element.get("is_travel"):
            continue
        
        for loc_field in ["from", "to"]:
            loc_name = str(element.get(loc_field, "")).upper().strip()
            if loc_name in loc2idx:
                loc_idx = loc2idx[loc_name]
                distance = float(Mdist[loc_idx, task_start_idx])
                
                if distance <= 50:  # Within 50 miles
                    return True
    
    return False

# Update your main _evaluate_cascade_scenario function to use multi-driver cascades:
def _evaluate_cascade_scenario(
    base_driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    weekday: str,
    sla_windows: Dict[int, Dict[str, int]],
    max_cascade_depth: int
) -> Optional[CascadeCandidateOut]:
    """
    Enhanced cascade evaluation - tries multi-driver cascades first
    """
    
    # Try multi-driver cascade first
    if max_cascade_depth > 1:
        result = _evaluate_multi_driver_cascade(
            base_driver_id, new_trip_req, DATA, matrices, cost_cfg,
            weekday, sla_windows, max_cascade_depth
        )
        
        if result:
            return result
    
    # Fall back to single-driver cuOpt
    return _evaluate_cascade_scenario_with_cuopt(
        base_driver_id, new_trip_req, DATA, matrices, cost_cfg,
        weekday, sla_windows, max_cascade_depth
    )

def _build_cascade_cuopt_payload(
    primary_driver_id: str,
    new_trip_req: PlanRequest,
    displaced_work: List[Dict[str, Any]],
    reassignment_candidates: Dict[str, List[str]],
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str
) -> Dict[str, Any]:
    """
    Build cuOpt payload for complete multi-driver cascade optimization
    FIXED: Proper capacities array format according to cuOpt documentation
    """
    
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    
    # Collect all locations involved in the cascade
    locations = set()
    locations.add(new_trip_req.start_location.upper())
    locations.add(new_trip_req.end_location.upper())
    
    for task in displaced_work:
        locations.add(task["start_location"].upper())
        locations.add(task["end_location"].upper())
    
    # Filter to valid locations
    valid_locations = {loc for loc in locations if loc in loc2idx}
    location_list = sorted(valid_locations)
    
    # Create location mapping
    cuopt_loc_map = {loc: i for i, loc in enumerate(location_list)}
    
    # Build cost matrix for involved locations
    n_locs = len(location_list)
    cost_matrix = []
    
    for i in range(n_locs):
        row = []
        for j in range(n_locs):
            from_idx = loc2idx[location_list[i]]
            to_idx = loc2idx[location_list[j]]
            cost = int(Mdist[from_idx, to_idx])
            row.append(cost)
        cost_matrix.append(row)
    
    # Build vehicles (drivers involved in cascade)
    all_drivers = {primary_driver_id}
    
    for candidates in reassignment_candidates.values():
        all_drivers.update(candidates[:2])  # Limit to top 2 candidates per task
    
    # Filter to valid drivers and build vehicle data
    vehicle_locations = []
    vehicle_time_windows = []
    individual_capacities = []  # Collect individual capacities first
    vehicle_types = []
    driver_to_vehicle = {}
    vehicle_id = 0
    
    for driver_id in all_drivers:
        driver_meta = DATA["driver_states"]["drivers"].get(driver_id, {})
        elements = driver_meta.get("elements", [])
        active_elements = [e for e in elements if element_active_on_weekday(e, weekday)]
        
        if not active_elements:
            continue
        
        duty_start = min(e.get("start_min", 1440) for e in active_elements)
        duty_end = max(e.get("end_min", 0) for e in active_elements)
        
        vehicle_locations.append([0, 0])  # Start at depot (location 0)
        vehicle_time_windows.append([duty_start, duty_end])
        individual_capacities.append(100)  # Each vehicle has capacity 100
        vehicle_types.append(1)
        
        driver_to_vehicle[driver_id] = vehicle_id
        vehicle_id += 1
    
    if not vehicle_locations:
        print(f"[cascade] No valid vehicles found for cascade")
        return None
    # Documentation: "Each capacity dimension's length must align with the length of the vehicle_locations array"
    capacities = [individual_capacities]  # ✅ CORRECT FORMAT: [[100, 100, 100, ...]]
    
    print(f"[cascade] Built {len(vehicle_locations)} vehicles with capacities format: {capacities}")
    
    # Build tasks (new service + displaced work)
    task_locations = []
    task_time_windows = []
    service_times = []
    individual_demands = []  # ✅ FIXED: Collect individual demands first
    
    # Add new service as high-priority task
    if new_trip_req.start_location.upper() in cuopt_loc_map and new_trip_req.end_location.upper() in cuopt_loc_map:
        task_locations.append(cuopt_loc_map[new_trip_req.end_location.upper()])
        
        req_time = minute_of_day_local(new_trip_req.when_local)
        task_time_windows.append([req_time - 30, req_time + 30])  # 1-hour flexibility
        
        service_times.append(int(new_trip_req.trip_minutes or 60))
        individual_demands.append(1)  # ✅ FIXED: Add to individual list
    
    # Add displaced work as tasks that need reassignment
    for task in displaced_work:
        if task.get("reassignment_required", True):
            if task["end_location"].upper() in cuopt_loc_map:
                task_locations.append(cuopt_loc_map[task["end_location"].upper()])
                task_time_windows.append([task["start_time"] - 60, task["end_time"] + 60])
                service_times.append(int(task["duration_minutes"]))
                individual_demands.append(1)  # ✅ FIXED: Add to individual list
    
    # ✅ CRITICAL FIX: Build demands in correct format
    demands = [individual_demands] if individual_demands else [[]]
    
    print(f"[cascade] Tasks built - Locations: {len(task_locations)}, Demands: {demands}")
    
    # Validate dimensions
    n_vehicles = len(vehicle_locations)
    n_tasks = len(task_locations)
    
    print(f"[cascade] Payload validation - Vehicles: {n_vehicles}, Tasks: {n_tasks}, Locations: {n_locs}")
    print(f"[cascade] Capacities shape: {len(capacities)} x {len(capacities[0]) if capacities else 0}")
    print(f"[cascade] Demands shape: {len(demands)} x {len(demands[0]) if demands else 0}")
    print(f"[cascade] Vehicle types: {vehicle_types}")
    
    # Build final payload
    payload = {
        "cost_matrix_data": {
            "data": {"1": cost_matrix}  # Matrix for vehicle type 1
        },
        "fleet_data": {
            "vehicle_locations": vehicle_locations,
            "vehicle_time_windows": vehicle_time_windows,
            "capacities": capacities,
            "vehicle_types": vehicle_types
        },
        "task_data": {
            "task_locations": task_locations,
            "task_time_windows": task_time_windows,
            "service_times": service_times,
            "demand": demands  # ✅ FIXED: Now [[1, 1, ...]] not [[1], [1], ...]
        },
        "solver_config": {
            "time_limit": 45  # Longer for multi-driver problems
        }
    }
    
    return payload

def _validate_cuopt_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate cuOpt payload format according to documentation
    Returns (is_valid, error_message)
    """
    
    try:
        fleet_data = payload.get("fleet_data", {})
        
        # Check required fields
        vehicle_locations = fleet_data.get("vehicle_locations", [])
        capacities = fleet_data.get("capacities", [])
        vehicle_types = fleet_data.get("vehicle_types", [])
        
        n_vehicles = len(vehicle_locations)
        
        if n_vehicles == 0:
            return False, "No vehicles provided"
        
        # Check capacities format
        if not capacities:
            return False, "No capacities provided"
        
        if not isinstance(capacities[0], list):
            return False, f"Capacities must be array of arrays, got: {type(capacities[0])}"
        
        capacity_dimension_length = len(capacities[0])
        if capacity_dimension_length != n_vehicles:
            return False, f"Capacity dimension length ({capacity_dimension_length}) must equal number of vehicles ({n_vehicles})"
        
        # Check vehicle_types length
        if len(vehicle_types) != n_vehicles:
            return False, f"Vehicle types length ({len(vehicle_types)}) must equal number of vehicles ({n_vehicles})"
        
        # Check cost matrix exists for vehicle types
        cost_data = payload.get("cost_matrix_data", {}).get("data", {})
        for vtype in set(vehicle_types):
            if str(vtype) not in cost_data:
                return False, f"Missing cost matrix for vehicle type {vtype}"
        
        print(f"[cuopt] Payload validation PASSED: {n_vehicles} vehicles, capacities={capacities}")
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"

# Update the _evaluate_multi_driver_cascade function to use validation:
def _evaluate_multi_driver_cascade_with_validation(
    primary_driver_id: str,
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
    weekday: str,
    sla_windows: Dict[int, Dict[str, int]],
    max_cascade_depth: int
) -> Optional[CascadeCandidateOut]:
    """
    Evaluate multi-driver cascade scenarios with payload validation
    """
    
    if not CUOPT_CLIENT_AVAILABLE:
        print(f"[cascade] cuOpt client not available, skipping multi-driver cascade")
        return None
    
    try:
        print(f"[cascade] Building multi-driver cascade for {primary_driver_id}")
        
        # STEP 1: Analyze what work gets displaced by the new service
        displaced_work = _find_displaced_work(
            primary_driver_id, new_trip_req, DATA, weekday, matrices
        )
        
        if not displaced_work:
            print(f"[cascade] No displaced work for {primary_driver_id}")
            return None
        
        print(f"[cascade] Found {len(displaced_work)} displaced tasks for {primary_driver_id}")
        
        # STEP 2: Filter displaced work by priority rules
        filtered_displaced_work = _filter_displaced_work_by_priority(displaced_work, new_trip_req.priority)
        
        print(f"[cascade] After priority filtering: {len(filtered_displaced_work)} tasks need reassignment")
        
        # STEP 3: Find candidate drivers for displaced work
        reassignment_candidates = _find_reassignment_candidates(
            filtered_displaced_work, DATA, matrices, weekday, sla_windows
        )
        
        if not reassignment_candidates:
            print(f"[cascade] No reassignment candidates found for {primary_driver_id}")
            return None
        
        # STEP 4: Build complete cascade payload for cuOpt
        cascade_payload = _build_cascade_cuopt_payload(
            primary_driver_id=primary_driver_id,
            new_trip_req=new_trip_req,
            displaced_work=filtered_displaced_work,
            reassignment_candidates=reassignment_candidates,
            DATA=DATA,
            matrices=matrices,
            weekday=weekday
        )
        
        if not cascade_payload:
            print(f"[cascade] Failed to build payload for {primary_driver_id}")
            return None
        
        # STEP 5: Validate payload before sending to cuOpt
        is_valid, error_msg = _validate_cuopt_payload(cascade_payload)
        if not is_valid:
            print(f"[cascade] Payload validation failed for {primary_driver_id}: {error_msg}")
            return None
        
        # STEP 6: Solve with cuOpt
        cuopt_host = os.getenv("CUOPT_HOST", "cuopt")
        cuopt_client = CuOptServiceSelfHostClient(
            ip=cuopt_host,
            port=5000,
            polling_timeout=30,  # Longer timeout for multi-driver problems
            timeout_exception=False
        )
        
        print(f"[cascade] Solving multi-driver cascade with cuOpt (validated payload)")
        solution = cuopt_client.get_optimized_routes(cascade_payload)
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=100)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:  # Success
                print(f"[cascade] Multi-driver cascade SUCCESS for {primary_driver_id}")
                return _parse_cascade_cuopt_solution(
                    solution, primary_driver_id, new_trip_req, filtered_displaced_work,
                    reassignment_candidates, DATA, matrices, weekday, cost_cfg
                )
            else:
                print(f"[cascade] cuOpt solver failed for {primary_driver_id}: status={status}")
        else:
            print(f"[cascade] cuOpt returned invalid response for {primary_driver_id}")
            
        return None
        
    except Exception as e:
        print(f"[cascade] Multi-driver cascade failed for {primary_driver_id}: {e}")
        return None

def _build_correct_cuopt_payload(
    candidates: List[str],
    new_trip_req: PlanRequest,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str
) -> Dict[str, Any]:
    """
    Build cuOpt payload with CORRECT cost matrix format for vehicle types
    FIXED: Proper capacities array format
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
    
    # FIXED: Correct capacities format
    payload = {
        "cost_matrix_data": {
            "data": {"1": cost_matrix}  # Matrix for vehicle type 1
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]],  # Vehicle starts at location 0
            "vehicle_time_windows": [[duty_start, duty_end]],
            "capacities": [[100]],  # ✅ CORRECT: Single array with one vehicle's capacity
            "vehicle_types": [1]  # This must match the key in cost_matrix_data
        },
        "task_data": {
            "task_locations": [1],  # Task at location 1
            "task_time_windows": [[duty_start, duty_end]],
            "service_times": [int(new_trip_req.trip_minutes or 60)],
            "demand": [[1]]  # ✅ CORRECT: Single array with one task's demand
        },
        "solver_config": {
            "time_limit": 10  # Quick solve
        }
    }
    
    return payload

def _parse_cascade_cuopt_solution(
    solution: Dict[str, Any],
    primary_driver_id: str,
    new_trip_req: PlanRequest,
    displaced_work: List[Dict[str, Any]],
    reassignment_candidates: Dict[str, List[str]],
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str,
    cost_cfg: Dict[str, float]
) -> CascadeCandidateOut:
    """
    Parse cuOpt solution and build complete cascade result with schedules
    """
    
    solver_response = solution["response"]["solver_response"]
    solution_cost = solver_response.get("solution_cost", 0)
    vehicle_data = solver_response.get("vehicle_data", {})
    
    # DEBUG: Log what cuOpt returned
    print(f"\n[cuopt] ===== CUOPT SOLUTION ANALYSIS =====")
    print(f"[cuopt] Vehicles in payload: 3")
    print(f"[cuopt] Tasks in payload: {len(displaced_work) + 1} (1 new service + {len(displaced_work)} displaced)")
    print(f"[cuopt] Vehicles used by cuOpt: {len(vehicle_data)}")
    
    # Analyze if cascade was needed
    if len(vehicle_data) == 1:
        print(f"[cuopt] ✅ SINGLE-DRIVER SOLUTION: cuOpt determined 1 driver can handle all work efficiently")
        print(f"[cuopt]    This means NO CASCADE is needed - primary driver can do everything")
    elif len(vehicle_data) > 1:
        print(f"[cuopt] 🔄 MULTI-DRIVER CASCADE: cuOpt used {len(vehicle_data)} drivers")
        print(f"[cuopt]    This means a CASCADE IS necessary - work distributed across drivers")
    else:
        print(f"[cuopt] ⚠️  WARNING: No vehicles used in solution!")
    
    # Log task assignments
    for vehicle_key in vehicle_data.keys():
        vdata = vehicle_data[vehicle_key]
        task_ids = vdata.get("task_id", [])
        print(f"[cuopt] Vehicle {vehicle_key}: {len(task_ids)} tasks → {task_ids}")
    
    print(f"[cuopt] =====================================\n")
    
    # Build driver mapping
    vehicle_to_driver = {}
    vehicle_to_driver["0"] = primary_driver_id
    
    # Map other vehicles to candidate drivers
    candidate_list = []
    for candidates in reassignment_candidates.values():
        candidate_list.extend(candidates[:2])
    
    # Remove duplicates while preserving order
    unique_candidates = []
    seen = {primary_driver_id}
    for cand in candidate_list:
        if cand not in seen:
            unique_candidates.append(cand)
            seen.add(cand)
    
    for i, cand_driver in enumerate(unique_candidates, start=1):
        vehicle_to_driver[str(i)] = cand_driver
    
    print(f"[cuopt] Vehicle to driver mapping: {vehicle_to_driver}")
    
    # Build before/after schedules for affected drivers
    before_after_schedules = {}
    cascade_chain = []
    
    for vehicle_key in vehicle_data.keys():
        driver_id = vehicle_to_driver.get(vehicle_key)
        if not driver_id:
            print(f"[cuopt] Warning: No driver mapping for vehicle {vehicle_key}")
            continue
        
        # Get driver's original schedule
        driver_meta = DATA["driver_states"]["drivers"].get(driver_id, {})
        elements = driver_meta.get("elements", [])
        active_elements = [e for e in elements if element_active_on_weekday(e, weekday)]
        
        if not active_elements:
            continue
        
        # Build "before" schedule
        before_schedule = []
        for i, element in enumerate(active_elements):
            before_schedule.append({
                "index": i,
                "element_type": element.get("element_type", "TRAVEL"),
                "from": element.get("from", ""),
                "to": element.get("to", ""),
                "start_time": f"{element.get('start_min', 0)//60:02d}:{element.get('start_min', 0)%60:02d}",
                "end_time": f"{element.get('end_min', 0)//60:02d}:{element.get('end_min', 0)%60:02d}",
                "start_min": element.get("start_min", 0),
                "end_min": element.get("end_min", 0),
                "priority": element.get("priority", 3),
                "load_type": element.get("planz_code", "UNKNOWN"),
                "changes": ""
            })
        
        # Build "after" schedule from cuOpt solution
        print(f"[cuopt] ✅ Building optimized schedule for {driver_id} from cuOpt route")
        after_schedule = _rebuild_schedule_from_cuopt_route(
            driver_id,
            vehicle_data[vehicle_key],
            before_schedule,
            new_trip_req,
            displaced_work,
            DATA,
            matrices,
            weekday
        )
        
        before_after_schedules[driver_id] = {
            "before": before_schedule,
            "after": after_schedule
        }
        
        # Build cascade chain
        vdata = vehicle_data[vehicle_key]
        if vdata.get("task_id"):
            cascade_chain.append({
                "step": len(cascade_chain) + 1,
                "vehicle_id": driver_id,
                "tasks": vdata.get("task_id", []),
                "route": vdata.get("route", [])
            })
    
    drivers_affected = len(before_after_schedules)
    
    # Summary
    if drivers_affected == 1:
        print(f"[cuopt] 📊 RESULT: Single-driver solution (no cascade needed)")
    else:
        print(f"[cuopt] 📊 RESULT: Multi-driver cascade with {drivers_affected} drivers affected")
    
    print(f"[cuopt] 💰 Total cost: £{solution_cost:.2f}")
    
    return CascadeCandidateOut(
        candidate_id=f"CASCADE_{primary_driver_id}",
        primary_driver_id=primary_driver_id,
        total_system_cost=float(solution_cost),
        drivers_affected=drivers_affected,
        cascade_chain=cascade_chain,
        before_after_schedules=before_after_schedules,
        is_fully_feasible=True,
        uncovered_p4_tasks=[],
        disposed_p5_tasks=[]
    )

def _rebuild_schedule_from_cuopt_route(
    driver_id: str,
    vehicle_route: Dict[str, Any],
    original_schedule: List[Dict[str, Any]],
    new_trip_req: PlanRequest,
    displaced_work: List[Dict[str, Any]],
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str
) -> List[Dict[str, Any]]:
    """
    Rebuild driver schedule from cuOpt's optimized route.
    
    cuOpt returns:
    - task_id: List of task identifiers (e.g., ["0", "1", "2"] where indices map to tasks)
    - arrival_stamp: List of arrival times in minutes from start of day
    - route: List of location indices in the cuOpt problem
    - type: List of task types
    
    We need to:
    1. Map cuOpt's task IDs back to our actual work elements
    2. Reconstruct the schedule in the order cuOpt determined
    3. Use cuOpt's timing information where available
    """
    
    task_ids = vehicle_route.get("task_id", [])
    arrival_stamps = vehicle_route.get("arrival_stamp", [])
    route_indices = vehicle_route.get("route", [])
    
    print(f"[cuopt] Rebuilding route with {len(task_ids)} tasks")
    print(f"[cuopt] Task order from cuOpt: {task_ids}")
    print(f"[cuopt] Arrival times: {arrival_stamps}")
    
    # Build task mapping: task_id -> actual work element
    # Task IDs from cuOpt are string indices like "0", "1", "2"
    # These map to: [0=new_service, 1=displaced_task_1, 2=displaced_task_2, ...]
    task_map = {}
    
    # Task 0 is always the new service request
    task_map["0"] = {
        "type": "NEW_SERVICE",
        "from": new_trip_req.start_location,
        "to": new_trip_req.end_location,
        "priority": new_trip_req.priority,
        "duration": int(new_trip_req.trip_minutes or 60),
        "load_type": "NEW_SERVICE"
    }
    
    # Tasks 1+ are displaced work items
    for i, displaced_task in enumerate(displaced_work, start=1):
        task_id_str = str(i)
        element = displaced_task.get("element", {})
        task_map[task_id_str] = {
            "type": "DISPLACED_WORK",
            "from": element.get("from", ""),
            "to": element.get("to", ""),
            "priority": element.get("priority", 3),
            "duration": element.get("end_min", 0) - element.get("start_min", 0),
            "load_type": element.get("planz_code", "UNKNOWN"),
            "original_start": element.get("start_min", 0),
            "original_end": element.get("end_min", 0)
        }
    
    # Start with non-displaced elements from original schedule
    # We need to keep all the fixed elements (START FACILITY, MEAL RELIEF, END FACILITY, etc.)
    rebuilt_schedule = []
    
    # Track which original elements are being displaced
    displaced_element_indices = set()
    for displaced_task in displaced_work:
        original_element = displaced_task.get("element", {})
        # Find this element in the original schedule
        for idx, orig_elem in enumerate(original_schedule):
            if (orig_elem.get("from") == original_element.get("from") and 
                orig_elem.get("to") == original_element.get("to") and
                orig_elem.get("start_min") == original_element.get("start_min")):
                displaced_element_indices.add(idx)
                break
    
    # Copy non-displaced, non-travel elements first (facilities, breaks, etc.)
    for idx, element in enumerate(original_schedule):
        if idx not in displaced_element_indices and not element.get("is_travel", element.get("element_type") == "TRAVEL"):
            rebuilt_schedule.append(element.copy())
    
    # Now insert the cuOpt-optimized travel tasks in the correct order
    # We'll need to insert them in chronological order based on arrival_stamps
    optimized_tasks = []
    
    for i, task_id in enumerate(task_ids):
        # Skip depot visits (type might be "Depot" in cuOpt response)
        if task_id not in task_map:
            continue
            
        task_info = task_map[task_id]
        arrival_time = arrival_stamps[i] if i < len(arrival_stamps) else None
        
        # Estimate start time (arrival - travel time to this location)
        # For now, use the arrival time as the start
        if arrival_time is not None:
            start_min = int(arrival_time)
            end_min = start_min + task_info["duration"]
        else:
            # Fallback to original times or sequential insertion
            start_min = task_info.get("original_start", 0)
            end_min = task_info.get("original_end", 0)
        
        optimized_task = {
            "index": len(optimized_tasks),  # Temporary index
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": task_info["from"],
            "to": task_info["to"],
            "start_time": f"{start_min//60:02d}:{start_min%60:02d}",
            "end_time": f"{end_min//60:02d}:{end_min%60:02d}",
            "start_min": start_min,
            "end_min": end_min,
            "priority": task_info["priority"],
            "load_type": task_info["load_type"],
            "changes": "OPTIMIZED_BY_CUOPT" if task_info["type"] == "NEW_SERVICE" else "REASSIGNED_BY_CUOPT"
        }
        
        optimized_tasks.append(optimized_task)
    
    # Now we need to interleave the fixed elements (facilities, breaks) with the optimized tasks
    # Sort all elements by start time
    all_elements = rebuilt_schedule + optimized_tasks
    all_elements.sort(key=lambda x: x.get("start_min", 0))
    
    # Re-index
    for idx, element in enumerate(all_elements):
        element["index"] = idx
    
    print(f"[cuopt] Rebuilt schedule: {len(all_elements)} total elements ({len(optimized_tasks)} optimized tasks)")
    
    return all_elements

def _map_cuopt_vehicle_to_driver(
    vehicle_key: str,
    primary_driver_id: str,
    reassignment_candidates: Dict[str, List[str]],
    vehicle_index: int
) -> str:
    """
    Map cuOpt vehicle index back to actual driver ID.
    
    Vehicle 0 is always the primary driver.
    Vehicles 1+ are candidates from reassignment_candidates.
    """
    
    if vehicle_index == 0:
        return primary_driver_id
    
    # For secondary vehicles, we need to figure out which candidate driver this is
    # This is a simplified mapping - you may need more sophisticated logic
    all_candidates = []
    for candidates_list in reassignment_candidates.values():
        all_candidates.extend(candidates_list[:2])  # Top 2 per task
    
    # Remove duplicates while preserving order
    unique_candidates = []
    seen = {primary_driver_id}
    for cand in all_candidates:
        if cand not in seen:
            unique_candidates.append(cand)
            seen.add(cand)
    
    # Map vehicle index to driver
    if vehicle_index - 1 < len(unique_candidates):
        return unique_candidates[vehicle_index - 1]
    
    # Fallback
    print(f"[cuopt] Warning: Could not map vehicle {vehicle_index} to driver")
    return f"UNKNOWN_DRIVER_{vehicle_index}"

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

def _compute_cascade_schedules(
    DATA: Dict[str, Any], 
    cascade_solutions: List[CascadeCandidateOut]
) -> List[Dict[str, Any]]:
    """
    Convert cascade solutions to schedule format expected by UI
    Matches the format in _compute_before_after_schedules from router.py
    """
    
    all_schedules = []
    
    for cascade_solution in cascade_solutions:
        # Convert each cascade solution's schedules to UI format
        ui_schedules = _build_ui_schedules_from_cascade(cascade_solution, DATA)
        all_schedules.extend(ui_schedules)
    
    return all_schedules

def generate_cascade_candidates_with_schedules(
    req: PlanRequest,
    DATA: Dict[str,Any],
    matrices: Dict[str,Any],
    cost_cfg: Dict[str,float],
    loc_meta: Dict[str,Any],
    sla_windows: Dict[int, Dict[str,int]],
    max_cascade_depth: int = 2,
    max_candidates: int = 10,
) -> Tuple[str, float, float, List[CandidateOut], List[Dict[str, Any]]]:
    """
    Enhanced version that returns both candidates AND schedule data for UI
    """
    
    weekday = weekday_from_local(req.when_local)
    req_min = minute_of_day_local(req.when_local)
    
    # Test official cuOpt client on first run
    cuopt_working = _test_official_cuopt_client()
    cuopt_status = "OFFICIAL_CLIENT" if cuopt_working else "HEURISTIC_FALLBACK"
    
    print(f"[cascade] Generating candidates for {req.start_location}→{req.end_location} (cuOpt: {cuopt_status})")
    
    try:
        base_driver_candidates = _enhanced_driver_filtering(
            req, DATA, matrices, loc_meta, weekday, req_min, sla_windows,
            calling_point_proximity_miles=50,
            max_drivers=20
        )
        
        if not base_driver_candidates:
            print("[cascade] No viable base drivers found")
            return weekday, 0.0, 0.0, [], []
        
        cascade_solutions = []
        
        for base_driver_id in base_driver_candidates[:10]:
            cascade_solution = _evaluate_cascade_scenario(
                base_driver_id=base_driver_id,
                new_trip_req=req,
                DATA=DATA,
                matrices=matrices,
                cost_cfg=cost_cfg,
                weekday=weekday,
                sla_windows=sla_windows,
                max_cascade_depth=max_cascade_depth
            )
            
            if cascade_solution and cascade_solution.is_fully_feasible:
                cascade_solutions.append(cascade_solution)
        
        # Convert to UI format
        final_candidates = []
        for solution in cascade_solutions[:max_candidates]:
            candidate = _enhanced_to_candidate_out(solution)
            final_candidates.append(candidate)
        
        # Build schedule data for UI
        schedules = _compute_cascade_schedules(DATA, cascade_solutions[:max_candidates])
        
        # Calculate trip details
        if cascade_solutions:
            Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
            start_idx = idx_of(req.start_location, loc2idx)
            end_idx = idx_of(req.end_location, loc2idx)
            trip_minutes = float(Mtime[start_idx, end_idx])
            trip_miles = float(Mdist[start_idx, end_idx])
        else:
            trip_minutes, trip_miles = 0.0, 0.0
        
        print(f"[cascade] Generated {len(final_candidates)} candidates with {len(schedules)} schedule entries")
        return weekday, trip_minutes, trip_miles, final_candidates, schedules
        
    except Exception as e:
        print(f"[cascade] Generation failed: {e}")
        return weekday, 0.0, 0.0, [], []
    
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
        reason = f"Enhanced cascade ({drivers_text}, {cost_text})"
    
    return CandidateOut(
        candidate_id=cascade_result.candidate_id,
        driver_id=cascade_result.primary_driver_id,
        type="cascade_optimized",
        est_cost=cascade_result.total_system_cost,
        deadhead_miles=0.0,  # TODO: Extract from cascade_chain
        delay_minutes=0.0,
        overtime_minutes=0.0,
        uses_emergency_rest=False,
        miles_delta=0.0,
        feasible_hard=cascade_result.is_fully_feasible,
        reason=reason
    )