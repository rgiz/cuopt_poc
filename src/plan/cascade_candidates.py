"""
cascade_candidates.py

New module for cascade-aware candidate generation using cuOpt.
Separate from existing candidates.py to avoid disrupting working code.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

from .models import PlanRequest, CandidateOut
from .candidates import minute_of_day_local, weekday_from_local, element_active_on_weekday, idx_of
from .config import load_sla_windows


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
    """
    Main entry point for cascade-aware candidate generation.
    
    Returns same format as existing generate_candidates() for compatibility.
    """
    
    weekday = weekday_from_local(req.when_local)
    req_min = minute_of_day_local(req.when_local)
    
    print(f"[cascade] Generating cascade candidates for {req.start_location}→{req.end_location}")
    
    try:
        # Enhanced pre-filtering with calling points logic
        base_driver_candidates = _enhanced_driver_filtering(
            req, DATA, matrices, loc_meta, weekday, req_min, sla_windows,  # ADD: Pass sla_windows
            calling_point_proximity_miles=50,
            max_drivers=100
        )
        
        if not base_driver_candidates:
            print("[cascade] No viable base drivers found")
            return weekday, 0.0, 0.0, []
        
        # Cascade scenario evaluation
        cascade_solutions = []
        
        for base_driver_id in base_driver_candidates[:20]:  # Limit for performance
            try:
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
                    
            except Exception as e:
                print(f"[cascade] Failed to evaluate {base_driver_id}: {e}")
                continue
        
        # Convert to backward-compatible format
        final_candidates = []
        for solution in cascade_solutions[:max_candidates]:
            candidate = solution.to_candidate_out()
            # Add a reason field to store cascade info (this field exists in CandidateOut)
            candidate.reason = f"Cascade: {solution.drivers_affected} drivers affected, £{solution.total_system_cost:.2f} total cost"
            final_candidates.append(candidate)
        
        # Extract trip details from first solution if available
        if cascade_solutions:
            trip_minutes = 60.0  # Will be calculated properly later
            trip_miles = 50.0    # Will be calculated properly later
        else:
            # Calculate from matrices as fallback
            Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
            start_idx = idx_of(req.start_location, loc2idx)
            end_idx = idx_of(req.end_location, loc2idx)
            trip_minutes = float(Mtime[start_idx, end_idx])
            trip_miles = float(Mdist[start_idx, end_idx])
        
        print(f"[cascade] Generated {len(final_candidates)} cascade candidates")
        return weekday, trip_minutes, trip_miles, final_candidates
        
    except Exception as e:
        print(f"[cascade] Cascade generation failed, falling back to heuristic: {e}")
        
        # Fallback to existing logic
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
    Evaluate cascade scenario with proper constraint checking.
    
    CRITICAL: This now validates duty time constraints and route feasibility.
    """
    
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
        
        print(f"[constraint] Driver {base_driver_id} current duty: {current_duty_minutes//60}h {current_duty_minutes%60}m")
        
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
        
        print(f"[constraint] Projected duty time: {projected_duty_minutes//60}h {projected_duty_minutes%60}m (limit: 13h)")
        
        if projected_duty_minutes > MAX_DUTY_MINUTES:
            print(f"[constraint] REJECTED: Driver {base_driver_id} would exceed 13-hour limit")
            return None
        
        # STEP 4: Check if new service fits in reasonable insertion point
        req_time = minute_of_day_local(new_trip_req.when_local)
        valid_insertion_found = False
        
        for i, element in enumerate(active_elements):
            element_start = element.get("start_min", 0)
            element_end = element.get("end_min", 0)
            
            # Check if service time is compatible with this element's timing
            time_gap = abs(req_time - element_start)
            if time_gap <= 120:  # Within 2 hours
                valid_insertion_found = True
                break
        
        if not valid_insertion_found:
            print(f"[constraint] REJECTED: No valid insertion point for {base_driver_id} at {req_time//60:02d}:{req_time%60:02d}")
            return None
        
        # STEP 5: Build realistic before/after schedules
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
        
        # After schedule - insert new service at best position
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
        
        # Copy before schedule and add new service
        after_schedule = before_schedule.copy()
        after_schedule.append(new_service_element)
        
        # STEP 6: Calculate realistic costs
        deadhead_cost = min_deadhead_to_service * cost_cfg.get("deadhead_cost_per_mile", 1.0)
        service_cost = service_miles * cost_cfg.get("deadhead_cost_per_mile", 1.0)
        admin_cost = cost_cfg.get("reassignment_admin_cost", 10.0)
        overtime_cost = max(0, projected_duty_minutes - MAX_DUTY_MINUTES) * cost_cfg.get("overtime_cost_per_minute", 1.0)
        
        total_cost = deadhead_cost + service_cost + admin_cost + overtime_cost
        
        print(f"[cascade] Driver {base_driver_id} feasible - Cost: £{total_cost:.2f}, Duty: {projected_duty_minutes//60}h{projected_duty_minutes%60}m")
        
        return CascadeCandidateOut(
            candidate_id=f"ENHANCED_{base_driver_id}",
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
        print(f"[cascade] Failed to evaluate {base_driver_id}: {e}")
        import traceback
        traceback.print_exc()
        return None