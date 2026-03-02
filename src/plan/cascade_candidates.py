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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from types import SimpleNamespace
import os
import time

from .models import PlanRequest, CandidateOut
from .candidates import minute_of_day_local, weekday_from_local, element_active_on_weekday, idx_of
from .config import load_sla_windows, ENABLE_AS_DIRECTED_INSERTION, ENABLE_STRICT_LEGAL_CONSTRAINTS, ENABLE_TRUE_CASCADE
from .geo import build_loc_meta_from_locations_csv, same_island_by_meta
from src.runtime import configure_logging


LOGGER = configure_logging(__name__)

_CUOPT_HEALTH_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "ok": False,
}

try:
    from cuopt_sh_client import CuOptServiceSelfHostClient
    CUOPT_CLIENT_AVAILABLE = True
    LOGGER.info("[cascade] cuopt-sh-client imported successfully")
except ImportError as e:
    LOGGER.warning("[cascade] cuopt-sh-client not available: %s", e)
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
        reason, reason_code, reason_detail = _build_cascade_reason_fields(self)
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
            reason=reason,
            reason_code=reason_code,
            reason_detail=reason_detail,
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
    preferred_driver_id: Optional[str] = None,
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
    LOGGER.info("[cascade] Generating candidates for %s→%s (cuOpt: %s)", req.start_location, req.end_location, status)
    
    try:
        pickup_radius = float(getattr(req, "geography_radius_miles", 15.0) or 15.0)
        home_base_radius = float(getattr(req, "home_base_radius_miles", 30.0) or 30.0)

        # Filter viable drivers
        base_drivers = _enhanced_driver_filtering(
            req, DATA, matrices, loc_meta, weekday, req_min, sla_windows,
            calling_point_proximity_miles=pickup_radius,
            home_base_to_delivery_miles=home_base_radius,
            max_drivers=20
        )
        
        if not base_drivers:
            LOGGER.info("[cascade] No viable drivers found")
            return weekday, 0.0, 0.0, [], []
        
        # Evaluate each driver with cuOpt (parallel)
        cascade_solutions = []
        max_drivers_affected = int(getattr(req, "max_drivers_affected", 5) or 5)
        target_driver = str(preferred_driver_id or "").strip()
        requested_top_n = max(1, int(getattr(req, "top_n", 10) or 10))
        drivers_to_evaluate = base_drivers[:requested_top_n]
        if target_driver and target_driver in base_drivers:
            drivers_to_evaluate = [target_driver]

        eval_timeout_sec = float(os.getenv("CASCADE_EVAL_TIMEOUT_SEC", "45"))

        executor = ThreadPoolExecutor(max_workers=5)
        try:
            futures = {
                executor.submit(
                    _evaluate_single_driver_with_cuopt,
                    driver_id=driver_id,
                    new_trip_req=req,
                    DATA=DATA,
                    matrices=matrices,
                    cost_cfg=cost_cfg,
                    weekday=weekday,
                    max_cascade_depth=max_cascade_depth,
                    max_drivers_affected=max_drivers_affected,
                ): driver_id
                for driver_id in drivers_to_evaluate
            }

            try:
                for future in as_completed(futures, timeout=eval_timeout_sec):
                    result = future.result()
                    if result and result.is_fully_feasible:
                        cascade_solutions.append(result)
            except FuturesTimeoutError:
                LOGGER.warning(
                    "[cascade] Candidate evaluation timed out after %.1fs (%d pending)",
                    eval_timeout_sec,
                    sum(1 for f in futures if not f.done()),
                )

            for future in futures:
                if not future.done():
                    future.cancel()
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
        
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
        
        LOGGER.info("[cascade] Generated %d candidates with %d schedules", len(final_candidates), len(schedules))
        return weekday, trip_minutes, trip_miles, final_candidates, schedules
        
    except Exception as e:
        LOGGER.exception("[cascade] Generation failed: %s", e)
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
    home_base_to_delivery_miles: float = 10,
    max_drivers: int = 100
) -> List[str]:
    """Enhanced filtering with calling points proximity logic"""
    
    Mtime, Mdist, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    start_idx = idx_of(req.start_location, loc2idx)
    end_idx = idx_of(req.end_location, loc2idx)
    
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
    after_home_base_filter = 0
    
    # INVALID location names to exclude
    INVALID_LOCATIONS = {"NAN", "NO_DATA", "UNKNOWN", "NONE", "", "NULL", "N/A"}

    def _resolve_home_base_idx(driver_meta_local: Dict[str, Any], elements_local: List[Dict[str, Any]]) -> Optional[int]:
        home_center_id = driver_meta_local.get("home_center_id")
        if home_center_id is not None:
            try:
                return int(home_center_id)
            except Exception:
                pass

        for key in ["home_loc", "start_loc"]:
            value = str(driver_meta_local.get(key, "")).upper().strip()
            if value and value in loc2idx:
                return int(loc2idx[value])

        for elem in elements_local:
            et = str(elem.get("element_type", "")).upper()
            if "START FACILITY" not in et:
                continue
            from_name = str(elem.get("from", "")).upper().strip()
            if from_name in loc2idx:
                return int(loc2idx[from_name])
        return None
    
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
                LOGGER.debug("[filter] Driver %s EXCLUDED: has invalid location (from=%s, to=%s)", duty_id, from_loc, to_loc)
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
            int(req.priority), calling_point_proximity_miles, matrices, loc2idx, duty_id
        ):
            continue
        after_calling_points_filter += 1

        # Filter 4: Home base should be near DELIVERY location (configurable slider)
        home_base_idx = _resolve_home_base_idx(driver_meta, elements)
        if home_base_idx is None:
            continue

        home_to_delivery_miles = float(Mdist[home_base_idx, end_idx])
        if home_to_delivery_miles != home_to_delivery_miles:
            continue
        if home_to_delivery_miles > float(home_base_to_delivery_miles):
            continue
        after_home_base_filter += 1
        
        viable_candidates.append(duty_id)
        
        if len(viable_candidates) >= max_drivers:
            break
    
    LOGGER.info(
        "[filter] Enhanced filtering: %d → %d → %d → %d → %d → %d candidates",
        total_drivers,
        after_day_filter,
        after_nan_filter,
        after_window_filter,
        after_calling_points_filter,
        after_home_base_filter,
    )
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
    req_priority: int,
    max_proximity_miles: float,
    matrices: Dict[str, Any],
    loc2idx: Dict[str, int],
    driver_id: str
) -> bool:
    """Check if driver has calling points within proximity of service dispatch point"""
    
    Mdist = matrices["dist"]
    earliest_service_time, latest_service_time = service_time_window

    def _as_int(value: Any, default: int = 3) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)
    
    # Expand window for positioning
    window_buffer = 120  # 2 hours
    earliest_position_time = earliest_service_time - window_buffer
    latest_position_time = latest_service_time + window_buffer
    
    for element in elements:
        element_type = str(element.get("element_type", "")).upper().strip()
        is_travel = bool(element.get("is_travel", False)) or ("TRAVEL" in element_type)
        is_as_directed = "AS DIRECTED" in element_type

        if not (is_travel or is_as_directed):
            continue

        if is_as_directed:
            as_directed_duration = element.get("duration_min")
            if as_directed_duration is None:
                try:
                    as_directed_duration = float(element.get("end_min", 0)) - float(element.get("start_min", 0))
                except Exception:
                    as_directed_duration = 0
            if float(as_directed_duration) < 30.0:
                continue

        if is_travel:
            existing_priority = _as_int(element.get("priority", 3), 3)
            load_type = str(element.get("load_type", "")).upper().strip()
            planz = str(element.get("planz_code", element.get("Planz Code", ""))).upper().strip()
            is_emptyish = (
                bool(element.get("is_empty", False))
                or "EMPTY" in load_type
                or "DEADHEAD" in load_type
                or "RETURN" in load_type
                or "EMPTY" in planz
                or "DEADHEAD" in planz
            )
            priority_displaceable = existing_priority >= int(req_priority)
            if not (is_emptyish or priority_displaceable):
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
                    LOGGER.debug(
                        "[geo] Driver %s viable: calling point %s (%.1fmi from service) during relevant time",
                        driver_id,
                        loc_name,
                        distance_to_service,
                    )
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
    weekday: str,
    max_cascade_depth: int = 2,
    max_drivers_affected: int = 5,
) -> Optional[CascadeCandidateOut]:
    """
    Evaluate single driver using cuOpt optimization.
    """
    
    if not CUOPT_CLIENT_AVAILABLE:
        LOGGER.debug("[cuopt] Client not available for %s", driver_id)
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
        
        LOGGER.debug("[cuopt] Solving for %s", driver_id)
        
        # Solve
        solution = cuopt_client.get_optimized_routes(payload)
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=50)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:  # Success
                cost = solver_response.get("solution_cost", 0)
                LOGGER.info("[cuopt] SUCCESS for %s: cost=%s", driver_id, cost)
                
                # Build result with schedule
                return _build_cascade_result_from_cuopt(
                    driver_id,
                    new_trip_req,
                    cost,
                    DATA,
                    matrices,
                    weekday,
                    max_cascade_depth=max_cascade_depth,
                    max_drivers_affected=max_drivers_affected,
                )
            else:
                LOGGER.warning("[cuopt] Solver failed for %s: status=%s", driver_id, status)
        else:
            LOGGER.warning("[cuopt] No valid response for %s", driver_id)
        
        return None
        
    except Exception as e:
        LOGGER.exception("[cuopt] Exception for %s: %s", driver_id, e)
        return None


def _build_cascade_result_from_cuopt(
    driver_id: str,
    new_trip_req: PlanRequest,
    cost: float,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str,
    max_cascade_depth: int = 2,
    max_drivers_affected: int = 5,
) -> CascadeCandidateOut:
    """Build CascadeCandidateOut with RSL-aware reconstruction"""
    
    driver_meta = DATA["driver_states"]["drivers"].get(driver_id)
    if not driver_meta:
        raise ValueError(f"Driver {driver_id} not found")
    
    driver_elements = driver_meta.get("elements", [])
    active_elements = [e for e in driver_elements if element_active_on_weekday(e, weekday)]
    
    # Build before schedule
    before_schedule = _build_driver_schedule(active_elements)
    
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
    
    LOGGER.debug("[rsl] Strategy: %s", strategy["type"])
    
    after_schedule = _apply_insertion_strategy_to_schedule(
        strategy=strategy,
        before_schedule=before_schedule,
        new_trip_req=new_trip_req,
        structure=structure,
        matrices=matrices,
    )

    continuity_ok, continuity_reason = _validate_schedule_continuity(after_schedule)
    if not continuity_ok:
        LOGGER.warning("[rsl] Continuity validation failed for %s: %s", driver_id, continuity_reason)

    home_base_name = _resolve_home_base_name(driver_meta, before_schedule, matrices.get("loc2idx", {}))
    home_ok, home_reason = _validate_home_base_return(
        after_schedule,
        home_base_name,
        enforce_strict=ENABLE_STRICT_LEGAL_CONSTRAINTS,
    )
    if not continuity_ok:
        home_ok = False
    if not home_ok:
        LOGGER.warning("[rsl] Home-base return validation failed for %s: %s", driver_id, home_reason)

    chain: List[Dict[str, Any]] = []
    uncovered_p4_tasks: List[Dict[str, Any]] = []
    disposed_p5_tasks: List[Dict[str, Any]] = []
    before_after_schedules = {
        driver_id: {
            "before": before_schedule,
            "after": after_schedule
        }
    }

    if ENABLE_TRUE_CASCADE:
        displaced_task = _extract_displaced_task_from_strategy(strategy)
        if displaced_task:
            assignment = _attempt_secondary_assignment_for_displaced(
                displaced_task=displaced_task,
                primary_driver_id=driver_id,
                DATA=DATA,
                matrices=matrices,
                weekday=weekday,
                max_depth=max(1, int(max_cascade_depth)),
                max_drivers_affected=max(1, int(max_drivers_affected)),
            )
            chain.extend(assignment.get("chain_steps", []))

            for sid, schedules in assignment.get("driver_schedules", {}).items():
                before_after_schedules[sid] = schedules

            unresolved = assignment.get("unresolved_tasks", [])
            if unresolved:
                for task_record in unresolved:
                    displaced_priority = int(task_record.get("priority", 5))
                    if displaced_priority <= 4:
                        uncovered_p4_tasks.append(task_record)
                    else:
                        disposed_p5_tasks.append(task_record)
            elif not assignment.get("assigned", False):
                displaced_priority = int(displaced_task.get("priority", 5))
                task_record = {
                    "from": displaced_task.get("from", ""),
                    "to": displaced_task.get("to", ""),
                    "priority": displaced_priority,
                    "reason": "secondary_assignment_unavailable",
                }
                if displaced_priority <= 4:
                    uncovered_p4_tasks.append(task_record)
                else:
                    disposed_p5_tasks.append(task_record)
    
    return CascadeCandidateOut(
        candidate_id=f"CUOPT_{driver_id}",
        primary_driver_id=driver_id,
        total_system_cost=float(cost),
        drivers_affected=max(1, len(before_after_schedules)),
        cascade_chain=chain,
        before_after_schedules=before_after_schedules,
        is_fully_feasible=bool(home_ok),
        uncovered_p4_tasks=uncovered_p4_tasks,
        disposed_p5_tasks=disposed_p5_tasks,
        )


def _apply_insertion_strategy_to_schedule(
    strategy: Dict[str, Any],
    before_schedule: List[Dict[str, Any]],
    new_trip_req: Any,
    structure: Dict[str, Any],
    matrices: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if strategy['type'] == 'empty_replacement':
        return _reconstruct_empty_replacement(strategy, before_schedule, new_trip_req, matrices, {})

    if strategy['type'] in {
        'leg_replacement',
        'overlap_prepickup_replacement',
        'overlap_postdrop_replacement',
        'nearby_substitution_replacement',
    }:
        return _reconstruct_leg_replacement(strategy, before_schedule, new_trip_req, matrices, {})

    if strategy['type'] == 'as_directed_replacement' and ENABLE_AS_DIRECTED_INSERTION:
        as_directed_schedule = _reconstruct_as_directed_replacement(
            strategy, before_schedule, new_trip_req, matrices, {}
        )
        if as_directed_schedule is not None:
            return as_directed_schedule
        return _reconstruct_duty_append(
            {"end_facility_index": len(before_schedule)},
            before_schedule,
            new_trip_req,
            structure,
            matrices,
            {},
        )

    if strategy['type'] == 'duty_append':
        return _reconstruct_duty_append(strategy, before_schedule, new_trip_req, structure, matrices, {})

    return _reconstruct_duty_append(
        {"end_facility_index": len(before_schedule)},
        before_schedule,
        new_trip_req,
        structure,
        matrices,
        {},
    )


def _build_driver_schedule(active_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    for element in active_elements:
        start_min = int(element.get("start_min", 0) or 0)
        end_min = int(element.get("end_min", start_min) or start_min)
        schedule.append(
            {
                "index": len(schedule),
                "element_type": element.get("element_type", "TRAVEL"),
                "from": element.get("from", ""),
                "to": element.get("to", ""),
                "start_time": f"{start_min//60:02d}:{start_min%60:02d}",
                "end_time": f"{end_min//60:02d}:{end_min%60:02d}",
                "priority": element.get("priority", 3),
                "is_travel": element.get("is_travel", False),
                "start_min": start_min,
                "end_min": end_min,
                "changes": "",
            }
        )
    return schedule


def _extract_displaced_task_from_strategy(strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    strategy_type = str(strategy.get("type", "")).lower()
    if strategy_type in {
        "empty_replacement",
        "leg_replacement",
        "overlap_prepickup_replacement",
        "overlap_postdrop_replacement",
        "nearby_substitution_replacement",
    }:
        preferred_displaced = dict(strategy.get("displaced_element") or {})
        target = preferred_displaced if preferred_displaced else dict(strategy.get("target_element") or {})
        if not target:
            return None
        return {
            "from": target.get("from", ""),
            "to": target.get("to", ""),
            "start_min": int(target.get("start_min", 0) or 0),
            "end_min": int(target.get("end_min", target.get("start_min", 0)) or 0),
            "priority": int(target.get("priority", 5) or 5),
            "element_type": target.get("element_type", "TRAVEL"),
            "is_travel": bool(target.get("is_travel", True)),
        }
    return None


def _attempt_secondary_assignment_for_displaced(
    displaced_task: Dict[str, Any],
    primary_driver_id: str,
    DATA: Dict[str, Any],
    matrices: Dict[str, Any],
    weekday: str,
    max_depth: int = 2,
    max_drivers_affected: int = 5,
    step: int = 1,
    visited_driver_ids: Optional[set] = None,
) -> Dict[str, Any]:
    visited = set(visited_driver_ids or set())
    visited.add(str(primary_driver_id))

    if len(visited) >= max(1, int(max_drivers_affected)):
        return {
            "assigned": False,
            "secondary_driver_id": None,
            "before_schedule": [],
            "after_schedule": [],
            "chain_step": {
                "step": step,
                "vehicle_id": primary_driver_id,
                "status": "blocked",
                "detail": "max_drivers_affected_reached",
            },
            "chain_steps": [
                {
                    "step": step,
                    "vehicle_id": primary_driver_id,
                    "status": "blocked",
                    "detail": "max_drivers_affected_reached",
                }
            ],
            "driver_schedules": {},
            "unresolved_tasks": [
                {
                    "from": displaced_task.get("from", ""),
                    "to": displaced_task.get("to", ""),
                    "priority": int(displaced_task.get("priority", 5)),
                    "reason": "max_drivers_affected_reached",
                }
            ],
        }

    if max_depth <= 0:
        return {
            "assigned": False,
            "secondary_driver_id": None,
            "before_schedule": [],
            "after_schedule": [],
            "chain_step": {
                "step": step,
                "vehicle_id": primary_driver_id,
                "status": "blocked",
                "detail": "max_depth_reached",
            },
            "chain_steps": [
                {
                    "step": step,
                    "vehicle_id": primary_driver_id,
                    "status": "blocked",
                    "detail": "max_depth_reached",
                }
            ],
            "driver_schedules": {},
            "unresolved_tasks": [
                {
                    "from": displaced_task.get("from", ""),
                    "to": displaced_task.get("to", ""),
                    "priority": int(displaced_task.get("priority", 5)),
                    "reason": "max_depth_reached",
                }
            ],
        }

    drivers_blob = DATA.get("driver_states", {}) or {}
    drivers = drivers_blob.get("drivers", drivers_blob)
    loc2idx = matrices.get("loc2idx", {})
    Mtime = matrices.get("time")
    Mdist = matrices.get("dist")

    max_secondary_pickup_miles = float(os.getenv("CASCADE_SECONDARY_MAX_PICKUP_MI", "80"))
    enforce_same_island = os.getenv("CASCADE_SECONDARY_ENFORCE_SAME_ISLAND", "true").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    loc_meta: Dict[str, Any] = {}
    try:
        locations_df = DATA.get("locations_df")
        if locations_df is not None:
            loc_meta = build_loc_meta_from_locations_csv(locations_df)
    except Exception:
        loc_meta = {}

    task_from = str(displaced_task.get("from", "")).upper().strip()
    task_to = str(displaced_task.get("to", "")).upper().strip()
    if not task_from or not task_to or task_from not in loc2idx or task_to not in loc2idx:
        return {
            "assigned": False,
            "secondary_driver_id": None,
            "before_schedule": [],
            "after_schedule": [],
            "chain_step": {
                "step": step,
                "vehicle_id": primary_driver_id,
                "status": "blocked",
                "detail": "invalid_displaced_task",
            },
            "chain_steps": [
                {
                    "step": step,
                    "vehicle_id": primary_driver_id,
                    "status": "blocked",
                    "detail": "invalid_displaced_task",
                }
            ],
            "driver_schedules": {},
            "unresolved_tasks": [
                {
                    "from": displaced_task.get("from", ""),
                    "to": displaced_task.get("to", ""),
                    "priority": int(displaced_task.get("priority", 5)),
                    "reason": "invalid_displaced_task",
                }
            ],
        }

    for driver_id, driver_meta in drivers.items():
        if str(driver_id) in visited:
            continue

        elements_all = driver_meta.get("elements", []) or []
        elements = [e for e in elements_all if element_active_on_weekday(e, weekday)]
        if not elements:
            continue

        before = _build_driver_schedule(elements)

        if not before:
            continue

        displaced_start = int(displaced_task.get("start_min", 0) or 0)
        task_duration = max(
            1,
            int(displaced_task.get("end_min", displaced_start) or displaced_start) - displaced_start,
        )
        if task_duration <= 0 and Mtime is not None:
            task_duration = int(Mtime[loc2idx[task_from], loc2idx[task_to]])
        task_duration = max(1, int(task_duration))

        candidate_last_loc = None
        for e in sorted(before, key=lambda x: int(x.get("end_min", 0) or 0), reverse=True):
            loc_name = str(e.get("to", "")).upper().strip() or str(e.get("from", "")).upper().strip()
            if loc_name:
                candidate_last_loc = loc_name
                break

        if not candidate_last_loc or candidate_last_loc not in loc2idx:
            continue

        if Mdist is not None:
            try:
                reposition_miles = float(Mdist[loc2idx[candidate_last_loc], loc2idx[task_from]])
            except Exception:
                reposition_miles = float("nan")
            if reposition_miles == reposition_miles and reposition_miles > max_secondary_pickup_miles:
                continue

        if enforce_same_island and loc_meta:
            src_meta = loc_meta.get(str(candidate_last_loc).upper().strip())
            dst_meta = loc_meta.get(str(task_from).upper().strip())
            same_island = same_island_by_meta(src_meta, dst_meta)
            if same_island is False:
                continue

        pseudo_req = SimpleNamespace(
            start_location=task_from,
            end_location=task_to,
            trip_minutes=task_duration,
            priority=int(displaced_task.get("priority", 5)),
            when_local="",
        )

        task_map = {
            "0": {
                "type": "NEW_SERVICE",
                "from": task_from,
                "to": task_to,
                "priority": int(displaced_task.get("priority", 5)),
                "duration": task_duration,
            }
        }
        structure = _identify_rsl_structure(before)
        candidate_arrival = displaced_start
        exact_match_start: Optional[int] = None
        nearest_travel_start: Optional[int] = None
        nearest_gap = float("inf")
        for elem in before:
            if not bool(elem.get("is_travel", False)):
                continue
            elem_start = int(elem.get("start_min", 0) or 0)
            elem_from = str(elem.get("from", "")).upper().strip()
            elem_to = str(elem.get("to", "")).upper().strip()
            gap = abs(elem_start - displaced_start)
            if elem_from == task_from and elem_to == task_to and gap < nearest_gap:
                exact_match_start = elem_start
                nearest_gap = gap
            if gap < nearest_gap:
                nearest_travel_start = elem_start
                nearest_gap = gap

        if exact_match_start is not None:
            candidate_arrival = exact_match_start
        elif nearest_travel_start is not None:
            candidate_arrival = nearest_travel_start

        strategy = _determine_insertion_strategy(
            task_ids=["0"],
            arrival_stamps=[candidate_arrival],
            task_map=task_map,
            structure=structure,
            original_schedule=before,
            matrices=matrices,
        )

        # Secondary cascade should reassign by displacing/replacing existing work,
        # not by appending additional work at duty tail.
        strategy_type = str(strategy.get("type", "")).lower()
        if strategy_type == "duty_append":
            continue

        # Secondary reassignment must be strict like-for-like displacement.
        # Reject flexible overlap/nearby/as-directed strategies here to avoid
        # malformed reconstructed duties and silent orphan suppression.
        if strategy_type not in {"empty_replacement", "leg_replacement"}:
            continue

        after = _apply_insertion_strategy_to_schedule(
            strategy=strategy,
            before_schedule=before,
            new_trip_req=pseudo_req,
            structure=structure,
            matrices=matrices,
        )

        valid, _ = _validate_schedule_continuity(after)
        if not valid:
            continue

        chain_step = {
            "step": step,
            "vehicle_id": str(driver_id),
            "status": "assigned",
            "detail": "displaced_task_reassigned",
        }

        driver_schedules = {
            str(driver_id): {
                "before": before,
                "after": after,
            }
        }
        chain_steps = [chain_step]
        unresolved_tasks: List[Dict[str, Any]] = []

        if ENABLE_STRICT_LEGAL_CONSTRAINTS:
            home_base_name = _resolve_home_base_name(driver_meta, before, loc2idx)
            home_ok, _ = _validate_home_base_return(after, home_base_name, enforce_strict=True)
            if not home_ok:
                continue

        next_displaced_task = _extract_displaced_task_from_strategy(strategy)
        if next_displaced_task is not None and max_depth > 1:
            recursive = _attempt_secondary_assignment_for_displaced(
                displaced_task=next_displaced_task,
                primary_driver_id=str(driver_id),
                DATA=DATA,
                matrices=matrices,
                weekday=weekday,
                max_depth=max_depth - 1,
                max_drivers_affected=max_drivers_affected,
                step=step + 1,
                visited_driver_ids=visited.union({str(driver_id)}),
            )
            chain_steps.extend(recursive.get("chain_steps", []))
            driver_schedules.update(recursive.get("driver_schedules", {}))
            unresolved_tasks.extend(recursive.get("unresolved_tasks", []))

        return {
            "assigned": True,
            "secondary_driver_id": str(driver_id),
            "before_schedule": before,
            "after_schedule": after,
            "chain_step": chain_step,
            "chain_steps": chain_steps,
            "driver_schedules": driver_schedules,
            "unresolved_tasks": unresolved_tasks,
        }

    return {
        "assigned": False,
        "secondary_driver_id": None,
        "before_schedule": [],
        "after_schedule": [],
        "chain_step": {
            "step": step,
            "vehicle_id": primary_driver_id,
            "status": "blocked",
            "detail": "no_secondary_driver_fit",
        },
        "chain_steps": [
            {
                "step": step,
                "vehicle_id": primary_driver_id,
                "status": "blocked",
                "detail": "no_secondary_driver_fit",
            }
        ],
        "driver_schedules": {},
        "unresolved_tasks": [
            {
                "from": displaced_task.get("from", ""),
                "to": displaced_task.get("to", ""),
                "priority": int(displaced_task.get("priority", 5)),
                "reason": "no_secondary_driver_fit",
            }
        ],
    }


def _resolve_home_base_name(
    driver_meta: Dict[str, Any],
    schedule: List[Dict[str, Any]],
    loc2idx: Dict[str, int],
) -> Optional[str]:
    home_loc = str(driver_meta.get("home_loc", "")).upper().strip()
    if home_loc:
        return home_loc

    start_loc = str(driver_meta.get("start_loc", "")).upper().strip()
    if start_loc:
        return start_loc

    home_center_id = driver_meta.get("home_center_id")
    if home_center_id is not None:
        try:
            home_center_id = int(home_center_id)
            mapped = next((k for k, v in loc2idx.items() if int(v) == home_center_id), None)
            if mapped:
                return str(mapped).upper().strip()
        except Exception:
            pass

    for element in schedule:
        et = str(element.get("element_type", "")).upper()
        if "START FACILITY" in et:
            start_from = str(element.get("from", "")).upper().strip()
            if start_from:
                return start_from

    return None


def _schedule_terminal_location(schedule: List[Dict[str, Any]]) -> Optional[str]:
    for element in sorted(schedule, key=lambda e: int(e.get("end_min", e.get("start_min", 0))), reverse=True):
        to_loc = str(element.get("to", "")).upper().strip()
        if to_loc:
            return to_loc
        from_loc = str(element.get("from", "")).upper().strip()
        if from_loc:
            return from_loc
    return None


def _validate_home_base_return(
    schedule: List[Dict[str, Any]],
    home_base_name: Optional[str],
    enforce_strict: bool,
) -> Tuple[bool, str]:
    if not enforce_strict:
        return True, "disabled"
    if not home_base_name:
        return False, "home_base_unknown"

    terminal = _schedule_terminal_location(schedule)
    if not terminal:
        return False, "terminal_location_unknown"

    if str(terminal).upper().strip() != str(home_base_name).upper().strip():
        return False, "not_returned_to_home_base"

    return True, "ok"


def _build_true_cascade_scaffold(primary_driver_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
    if max_depth <= 1:
        return []
    return [
        {
            "step": 1,
            "vehicle_id": primary_driver_id,
            "status": "scaffold",
            "detail": "true_cascade_placeholder",
        }
    ]

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

    loc2idx = matrices.get("loc2idx", {}) if isinstance(matrices, dict) else {}
    dist_matrix = matrices.get("dist") if isinstance(matrices, dict) else None
    
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

    # Strategy 1B: Exact non-empty leg replacement under priority precedence
    for idx, elem in enumerate(original_schedule):
        if not bool(elem.get("is_travel", False)):
            continue
        elem_from = str(elem.get("from", "")).upper().strip()
        elem_to = str(elem.get("to", "")).upper().strip()
        if elem_from != str(new_service_info["from"]).upper().strip():
            continue
        if elem_to != str(new_service_info["to"]).upper().strip():
            continue

        existing_priority = int(elem.get("priority", 5) or 5)
        requested_priority = int(new_service_info.get("priority", 3) or 3)
        if existing_priority < requested_priority:
            continue

        if abs(int(elem.get("start_min", 0) or 0) - int(new_service_arrival)) >= 60:
            continue

        return {
            "type": "leg_replacement",
            "target_index": idx,
            "target_element": elem,
            "new_service": new_service_info,
        }

    # Strategy 1C: Overlap pre-pickup replacement (planned C->B, new A->B)
    for idx, elem in enumerate(original_schedule):
        if not bool(elem.get("is_travel", False)):
            continue
        elem_from = str(elem.get("from", "")).upper().strip()
        elem_to = str(elem.get("to", "")).upper().strip()
        req_from = str(new_service_info["from"]).upper().strip()
        req_to = str(new_service_info["to"]).upper().strip()

        if elem_to != req_to or elem_from == req_from:
            continue

        existing_priority = int(elem.get("priority", 5) or 5)
        requested_priority = int(new_service_info.get("priority", 3) or 3)
        if existing_priority < requested_priority:
            continue

        if abs(int(elem.get("start_min", 0) or 0) - int(new_service_arrival)) >= 90:
            continue

        return {
            "type": "overlap_prepickup_replacement",
            "target_index": idx,
            "target_element": elem,
            "new_service": new_service_info,
        }

    # Strategy 1D: Overlap post-drop replacement (planned A->C, new A->B)
    for idx, elem in enumerate(original_schedule):
        if not bool(elem.get("is_travel", False)):
            continue
        elem_from = str(elem.get("from", "")).upper().strip()
        elem_to = str(elem.get("to", "")).upper().strip()
        req_from = str(new_service_info["from"]).upper().strip()
        req_to = str(new_service_info["to"]).upper().strip()

        if elem_from != req_from or elem_to == req_to:
            continue

        existing_priority = int(elem.get("priority", 5) or 5)
        requested_priority = int(new_service_info.get("priority", 3) or 3)
        if existing_priority < requested_priority:
            continue

        if abs(int(elem.get("start_min", 0) or 0) - int(new_service_arrival)) >= 90:
            continue

        displaced_follow_on = None
        old_drop = elem_to
        for next_idx in range(idx + 1, len(original_schedule)):
            next_elem = original_schedule[next_idx]
            if not bool(next_elem.get("is_travel", False)):
                continue
            next_from = str(next_elem.get("from", "")).upper().strip()
            if next_from == old_drop:
                displaced_follow_on = next_elem
                break

        return {
            "type": "overlap_postdrop_replacement",
            "target_index": idx,
            "target_element": elem,
            "displaced_element": displaced_follow_on,
            "new_service": new_service_info,
        }

    # Strategy 1E: Nearby substitution replacement (C~A and D~B)
    nearby_radius_miles = 50.0
    for idx, elem in enumerate(original_schedule):
        if not bool(elem.get("is_travel", False)):
            continue

        elem_from = str(elem.get("from", "")).upper().strip()
        elem_to = str(elem.get("to", "")).upper().strip()
        req_from = str(new_service_info["from"]).upper().strip()
        req_to = str(new_service_info["to"]).upper().strip()

        if elem_from == req_from and elem_to == req_to:
            continue
        if elem_from not in loc2idx or elem_to not in loc2idx or req_from not in loc2idx or req_to not in loc2idx:
            continue
        if dist_matrix is None:
            continue

        try:
            pickup_gap = float(dist_matrix[loc2idx[elem_from], loc2idx[req_from]])
            drop_gap = float(dist_matrix[loc2idx[elem_to], loc2idx[req_to]])
        except Exception:
            continue

        if pickup_gap > nearby_radius_miles or drop_gap > nearby_radius_miles:
            continue

        existing_priority = int(elem.get("priority", 5) or 5)
        requested_priority = int(new_service_info.get("priority", 3) or 3)
        if existing_priority < requested_priority:
            continue

        if abs(int(elem.get("start_min", 0) or 0) - int(new_service_arrival)) >= 120:
            continue

        return {
            "type": "nearby_substitution_replacement",
            "target_index": idx,
            "target_element": elem,
            "new_service": new_service_info,
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


def _finalize_schedule(reconstructed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, elem in enumerate(reconstructed):
        elem["index"] = idx
        if "start_min" in elem:
            elem["start_time"] = f"{int(elem['start_min'])//60:02d}:{int(elem['start_min'])%60:02d}"
        if "end_min" in elem:
            elem["end_time"] = f"{int(elem['end_min'])//60:02d}:{int(elem['end_min'])%60:02d}"

    valid, reason = _validate_schedule_continuity(reconstructed)
    if not valid:
        LOGGER.warning("[rsl] Continuity validation warning: %s", reason)
    return reconstructed


def _validate_schedule_continuity(schedule: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not schedule:
        return True, "ok"

    sorted_schedule = sorted(schedule, key=lambda e: int(e.get("start_min", 0)))

    previous_end = None
    previous_to = None
    for element in sorted_schedule:
        start = int(element.get("start_min", 0))
        end = int(element.get("end_min", start))

        if end < start:
            return False, "end_before_start"

        if previous_end is not None and start < previous_end:
            return False, "time_overlap"

        current_from = str(element.get("from", "")).upper().strip()
        if previous_to is not None and current_from and previous_to and current_from != previous_to:
            return False, "location_discontinuity"

        previous_end = end
        previous_to = str(element.get("to", "")).upper().strip() or current_from

    return True, "ok"


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
    
    LOGGER.info("[rsl] ✅ Empty leg replaced with loaded service at index %s", target_idx)
    return _finalize_schedule(reconstructed)


def _reconstruct_leg_replacement(
    strategy: Dict[str, Any],
    original_schedule: List[Dict[str, Any]],
    new_trip_req: PlanRequest,
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float]
) -> List[Dict[str, Any]]:
    reconstructed = [dict(e) for e in original_schedule]
    target_idx = strategy["target_index"]
    target_elem = strategy["target_element"]
    strategy_type = str(strategy.get("type", "leg_replacement")).lower()
    Mtime = matrices.get("time") if isinstance(matrices, dict) else None
    loc2idx = matrices.get("loc2idx", {}) if isinstance(matrices, dict) else {}

    change_marker = "LEG_REPLACED"
    if strategy_type == "overlap_prepickup_replacement":
        change_marker = "OVERLAP_PREPICKUP_REPLACED"
    elif strategy_type == "overlap_postdrop_replacement":
        change_marker = "OVERLAP_POSTDROP_REPLACED"
    elif strategy_type == "nearby_substitution_replacement":
        change_marker = "NEARBY_SUBSTITUTION_REPLACED"

    target_start = int(target_elem.get("start_min", 0) or 0)
    req_trip_minutes = getattr(new_trip_req, "trip_minutes", None)
    service_minutes = int(req_trip_minutes or max(1, int(target_elem.get("end_min", target_start) or target_start) - target_start) or 60)
    new_service_end = target_start + max(1, service_minutes)
    service_from = getattr(new_trip_req, "start_location", target_elem.get("from", ""))
    service_to = getattr(new_trip_req, "end_location", target_elem.get("to", ""))

    reconstructed[target_idx] = {
        **target_elem,
        "from": service_from,
        "to": service_to,
        "start_min": target_start,
        "end_min": new_service_end,
        "priority": new_trip_req.priority,
        "load_type": "NEW_SERVICE",
        "planz_code": "DELIVERY",
        "changes": change_marker,
    }

    if strategy_type == "overlap_prepickup_replacement":
        rebuilt = [dict(e) for e in reconstructed[:target_idx]]
        prev_loc = ""
        cursor = target_start
        if rebuilt:
            prev = rebuilt[-1]
            cursor = int(prev.get("end_min", target_start) or target_start)
            prev_loc = str(prev.get("to", "") or prev.get("from", "")).strip().upper()
        else:
            prev_loc = str(target_elem.get("from", "") or "").strip().upper()

        pickup_loc = str(new_trip_req.start_location or "").strip().upper()
        if prev_loc and pickup_loc and prev_loc != pickup_loc and Mtime is not None and prev_loc in loc2idx and pickup_loc in loc2idx:
            try:
                deadhead_minutes = max(1, int(Mtime[loc2idx[prev_loc], loc2idx[pickup_loc]]))
            except Exception:
                deadhead_minutes = 0
            if deadhead_minutes > 0:
                rebuilt.append(
                    {
                        "element_type": "TRAVEL",
                        "is_travel": True,
                        "from": prev_loc,
                        "to": new_trip_req.start_location,
                        "start_min": cursor,
                        "end_min": cursor + deadhead_minutes,
                        "priority": 5,
                        "load_type": "EMPTY_DEADHEAD",
                        "planz_code": "DEADHEAD",
                        "changes": "OVERLAP_PREPICKUP_DEADHEAD",
                    }
                )
                cursor += deadhead_minutes

        rebuilt.append(
            {
                **reconstructed[target_idx],
                "start_min": cursor,
                "end_min": cursor + max(1, service_minutes),
                "changes": "OVERLAP_PREPICKUP_REPLACED",
            }
        )
        cursor += max(1, service_minutes)

        for idx in range(target_idx + 1, len(reconstructed)):
            elem = dict(reconstructed[idx])
            duration = max(0, int(elem.get("end_min", 0) or 0) - int(elem.get("start_min", 0) or 0))
            elem["start_min"] = cursor
            elem["end_min"] = cursor + duration
            rebuilt.append(elem)
            cursor += duration

        reconstructed = rebuilt

    if strategy_type == "overlap_postdrop_replacement":
        old_drop = str(target_elem.get("to", "")).upper().strip()
        new_drop = str(new_trip_req.end_location).upper().strip()
        displaced_follow_on = strategy.get("displaced_element") or {}

        rebuilt = [dict(e) for e in reconstructed[: target_idx + 1]]
        cursor = int(rebuilt[-1].get("end_min", new_service_end) or new_service_end)
        return_inserted = False

        for idx in range(target_idx + 1, len(reconstructed)):
            elem = dict(reconstructed[idx])
            elem_from = str(elem.get("from", "")).upper().strip()
            elem_to = str(elem.get("to", "")).upper().strip()
            is_travel = bool(elem.get("is_travel", False))

            # Drop local old-drop tasks that are no longer reachable after reroute
            if not return_inserted and not is_travel and (elem_from == old_drop or elem_to == old_drop):
                continue

            # Replace old-drop departure leg with empty return from new drop
            if not return_inserted and is_travel and elem_from == old_drop:
                original_duration = max(1, int(elem.get("end_min", 0) or 0) - int(elem.get("start_min", 0) or 0))
                destination = str(elem.get("to", "")).strip()
                travel_duration = original_duration

                if Mtime is not None and new_drop in loc2idx and str(destination).upper().strip() in loc2idx:
                    try:
                        travel_duration = max(
                            1,
                            int(Mtime[loc2idx[new_drop], loc2idx[str(destination).upper().strip()]]),
                        )
                    except Exception:
                        travel_duration = original_duration

                rebuilt.append(
                    {
                        **elem,
                        "from": new_trip_req.end_location,
                        "to": destination,
                        "start_min": cursor,
                        "end_min": cursor + travel_duration,
                        "priority": 5,
                        "load_type": "RETURN_TO_BASE",
                        "planz_code": "RETURN",
                        "changes": "OVERLAP_POSTDROP_RETURN_RETIMED",
                    }
                )
                cursor += travel_duration
                return_inserted = True
                continue

            # Preserve remaining duty but retime sequentially
            duration = max(0, int(elem.get("end_min", 0) or 0) - int(elem.get("start_min", 0) or 0))
            elem["start_min"] = cursor
            elem["end_min"] = cursor + duration
            rebuilt.append(elem)
            cursor += duration

        reconstructed = rebuilt

        # If no follow-on displaced element captured from strategy, infer from old schedule.
        if not displaced_follow_on:
            for idx in range(target_idx + 1, len(original_schedule)):
                probe = original_schedule[idx]
                if not bool(probe.get("is_travel", False)):
                    continue
                if str(probe.get("from", "")).upper().strip() == old_drop:
                    strategy["displaced_element"] = dict(probe)
                    break

    LOGGER.info("[rsl] ✅ Exact leg replaced at index %s", target_idx)
    return _finalize_schedule(reconstructed)


def _reconstruct_as_directed_replacement(
    strategy: Dict[str, Any],
    original_schedule: List[Dict[str, Any]],
    new_trip_req: PlanRequest,
    matrices: Dict[str, Any],
    cost_cfg: Dict[str, float],
) -> Optional[List[Dict[str, Any]]]:
    target_idx = strategy.get("target_index")
    target_elem = strategy.get("target_element") or {}
    if target_idx is None:
        return None

    start_min = int(target_elem.get("start_min", 0))
    end_min = int(target_elem.get("end_min", start_min))
    if end_min <= start_min:
        return None

    when_local_value = getattr(new_trip_req, "when_local", "")
    requested_departure = minute_of_day_local(when_local_value) if when_local_value else start_min
    insertion_time = int(strategy.get("insertion_time", requested_departure))
    insertion_time = max(start_min, min(end_min, insertion_time))

    Mtime, _, loc2idx = matrices["time"], matrices["dist"], matrices["loc2idx"]
    service_from = new_trip_req.start_location.upper()
    service_to = new_trip_req.end_location.upper()

    block_from = str(target_elem.get("from", "")).upper().strip()
    block_to = str(target_elem.get("to", block_from)).upper().strip() or block_from
    load_start_target = max(start_min, insertion_time - LOADING_TIME_MINUTES)
    current_time = start_min

    insertion: List[Dict[str, Any]] = []

    if current_time < load_start_target:
        insertion.append({
            "element_type": "AS DIRECTED",
            "is_travel": False,
            "from": block_from or block_to,
            "to": block_from or block_to,
            "start_min": current_time,
            "end_min": load_start_target,
            "priority": int(target_elem.get("priority", 3)),
            "load_type": "AS_DIRECTED_REMAINDER",
            "planz_code": "AS_DIRECTED",
            "changes": "AS_DIRECTED_REMAINING",
        })
        current_time = load_start_target

    if block_from and block_from != service_from and block_from in loc2idx and service_from in loc2idx:
        deadhead_to = int(Mtime[loc2idx[block_from], loc2idx[service_from]])
        insertion.append({
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": block_from,
            "to": new_trip_req.start_location,
            "start_min": current_time,
            "end_min": current_time + deadhead_to,
            "priority": 5,
            "load_type": "EMPTY_DEADHEAD",
            "planz_code": "DEADHEAD",
            "changes": "AS_DIRECTED_REPLACED",
        })
        current_time += deadhead_to

    insertion.append({
        "element_type": "LOAD/ASSIST",
        "is_travel": False,
        "from": new_trip_req.start_location,
        "to": new_trip_req.start_location,
        "start_min": current_time,
        "end_min": current_time + LOADING_TIME_MINUTES,
        "priority": new_trip_req.priority,
        "load_type": "LOADING",
        "planz_code": "LOAD_ASSIST",
        "changes": "AS_DIRECTED_REPLACED",
    })
    current_time += LOADING_TIME_MINUTES

    service_minutes = int(getattr(new_trip_req, "trip_minutes", 60) or 60)
    insertion.append({
        "element_type": "TRAVEL",
        "is_travel": True,
        "from": new_trip_req.start_location,
        "to": new_trip_req.end_location,
        "start_min": current_time,
        "end_min": current_time + service_minutes,
        "priority": new_trip_req.priority,
        "load_type": "LOADED",
        "planz_code": "DELIVERY",
        "changes": "AS_DIRECTED_REPLACED",
    })
    current_time += service_minutes

    insertion.append({
        "element_type": "LOAD/ASSIST",
        "is_travel": False,
        "from": new_trip_req.end_location,
        "to": new_trip_req.end_location,
        "start_min": current_time,
        "end_min": current_time + OFFLOADING_TIME_MINUTES,
        "priority": new_trip_req.priority,
        "load_type": "OFFLOADING",
        "planz_code": "UNLOAD_ASSIST",
        "changes": "AS_DIRECTED_REPLACED",
    })
    current_time += OFFLOADING_TIME_MINUTES

    if block_to and service_to != block_to and service_to in loc2idx and block_to in loc2idx:
        deadhead_back = int(Mtime[loc2idx[service_to], loc2idx[block_to]])
        insertion.append({
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": new_trip_req.end_location,
            "to": block_to,
            "start_min": current_time,
            "end_min": current_time + deadhead_back,
            "priority": 5,
            "load_type": "EMPTY_DEADHEAD",
            "planz_code": "DEADHEAD",
            "changes": "AS_DIRECTED_REPLACED",
        })
        current_time += deadhead_back

    if current_time > end_min:
        LOGGER.debug("[rsl] AS DIRECTED block insufficient: required=%s available=%s", current_time - start_min, end_min - start_min)
        return None

    if current_time < end_min:
        insertion.append({
            "element_type": "AS DIRECTED",
            "is_travel": False,
            "from": block_to or block_from,
            "to": block_to or block_from,
            "start_min": current_time,
            "end_min": end_min,
            "priority": int(target_elem.get("priority", 3)),
            "load_type": "AS_DIRECTED_REMAINDER",
            "planz_code": "AS_DIRECTED",
            "changes": "AS_DIRECTED_REMAINING",
        })

    reconstructed = []
    for index, element in enumerate(original_schedule):
        if index == target_idx:
            reconstructed.extend(insertion)
        else:
            reconstructed.append(dict(element))

    LOGGER.info("[rsl] ✅ As Directed block replaced at index %s", target_idx)
    return _finalize_schedule(reconstructed)


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
    inserted = False
    for i, elem in enumerate(original_schedule):
        if i < end_facility_index:
            reconstructed.append(elem)
        elif i == end_facility_index:
            reconstructed.extend(append_sequence)
            inserted = True
            if "END FACILITY" in elem.get("element_type", "").upper():
                updated_end = dict(elem)
                updated_end["start_min"] = current_time
                updated_end["end_min"] = current_time + 15
                reconstructed.append(updated_end)
            else:
                reconstructed.append(elem)
        else:
            reconstructed.append(elem)

    if not inserted:
        reconstructed = [dict(e) for e in original_schedule]
        reconstructed.extend(append_sequence)
    
    LOGGER.info("[rsl] ✅ Duty append: %d elements added before END FACILITY", len(append_sequence))
    return _finalize_schedule(reconstructed)

# ============================================================================
# CUOPT CLIENT HELPERS
# ============================================================================

def _test_official_cuopt_client() -> bool:
    """Test cuOpt connectivity"""

    cache_ttl_sec = float(os.getenv("CUOPT_HEALTH_CACHE_TTL_SEC", "60"))
    now = time.time()
    if now - float(_CUOPT_HEALTH_CACHE.get("ts", 0.0) or 0.0) < cache_ttl_sec:
        return bool(_CUOPT_HEALTH_CACHE.get("ok", False))
    
    if not CUOPT_CLIENT_AVAILABLE:
        LOGGER.debug("[test] Official cuOpt client not available")
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
        
        LOGGER.debug("[test] Testing official cuOpt client...")
        solution = cuopt_client.get_optimized_routes(test_data)
        
        # Handle repoll if needed
        solution = _repoll_solution(cuopt_client, solution, repoll_tries=10)
        
        if solution and "response" in solution:
            solver_response = solution["response"].get("solver_response", {})
            status = solver_response.get("status", -1)
            
            if status == 0:
                cost = solver_response.get("solution_cost", 0)
                LOGGER.info("[test] Official client SUCCESS: status=%s, cost=%s", status, cost)
                _CUOPT_HEALTH_CACHE["ts"] = now
                _CUOPT_HEALTH_CACHE["ok"] = True
                return True
            else:
                LOGGER.warning("[test] Official client solver failed: status=%s", status)
        else:
            LOGGER.warning("[test] Official client returned no valid response")
            
        _CUOPT_HEALTH_CACHE["ts"] = now
        _CUOPT_HEALTH_CACHE["ok"] = False
        return False
        
    except Exception as e:
        LOGGER.exception("[test] Official client test failed: %s", e)
        _CUOPT_HEALTH_CACHE["ts"] = now
        _CUOPT_HEALTH_CACHE["ok"] = False
        return False


def _repoll_solution(cuopt_client, solution, repoll_tries=50):
    """
    Implement the repoll pattern from cuOpt documentation
    """
    
    # If solver is still busy, repoll using request ID
    if "reqId" in solution and "response" not in solution:
        req_id = solution["reqId"]
        LOGGER.debug("[cuopt] Repolling reqId: %s", req_id)
        
        for i in range(repoll_tries):
            try:
                solution = cuopt_client.repoll(req_id, response_type="dict")
                
                if "reqId" in solution and "response" in solution:
                    LOGGER.debug("[cuopt] Repoll successful after %d attempts", i + 1)
                    break
                    
                # Sleep briefly between polls
                time.sleep(0.5)
                
            except Exception as e:
                LOGGER.debug("[cuopt] Repoll attempt %d failed: %s", i + 1, e)
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
            "candidate_id": cascade_result.candidate_id,
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
    reason, reason_code, reason_detail = _build_cascade_reason_fields(cascade_result)
    
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
        reason=reason,
        reason_code=reason_code,
        reason_detail=reason_detail,
    )


def _build_cascade_reason_fields(cascade_result: CascadeCandidateOut) -> Tuple[str, str, str]:
    drivers_text = f"{cascade_result.drivers_affected} driver" + ("s" if cascade_result.drivers_affected != 1 else "")
    cost_text = f"£{cascade_result.total_system_cost:.2f}"

    chain_steps = list(cascade_result.cascade_chain or [])
    assigned_steps = [s for s in chain_steps if str(s.get("status", "")).lower() == "assigned"]
    blocked_steps = [s for s in chain_steps if str(s.get("status", "")).lower() == "blocked"]
    chain_depth = len(chain_steps)

    unresolved_total = len(cascade_result.uncovered_p4_tasks or []) + len(cascade_result.disposed_p5_tasks or [])

    if chain_steps:
        chain_summary = " → ".join([str(step.get("vehicle_id", "?")) for step in chain_steps])
        reason = f"Multi-driver cascade: {chain_summary} ({drivers_text}, {cost_text})"
        reason_code = "CASCADE_MULTI_DRIVER" if len(assigned_steps) > 1 else "CASCADE_SINGLE_HOP"
    else:
        reason = f"cuOpt optimized ({drivers_text}, {cost_text})"
        reason_code = "CASCADE_DIRECT_OPT"

    if unresolved_total > 0:
        reason_code = "CASCADE_PARTIAL_UNRESOLVED"

    def _serialize_unresolved_tasks(tasks: List[Dict[str, Any]]) -> str:
        entries: List[str] = []
        for task in tasks or []:
            task_from = str(task.get("from", "") or "").strip().upper() or "UNKNOWN"
            task_to = str(task.get("to", "") or "").strip().upper() or "UNKNOWN"
            try:
                task_priority = int(task.get("priority", 5) or 5)
            except Exception:
                task_priority = 5
            reason = str(task.get("reason", "") or "").strip().lower()
            reason_token = reason if reason else "unspecified"
            entries.append(f"{task_from}>{task_to}@P{task_priority}:{reason_token}")
        return "|".join(entries) if entries else "none"

    unresolved_p4_serialized = _serialize_unresolved_tasks(cascade_result.uncovered_p4_tasks or [])
    unresolved_p5_serialized = _serialize_unresolved_tasks(cascade_result.disposed_p5_tasks or [])

    reason_detail = (
        f"chain_depth={chain_depth}; "
        f"assigned_steps={len(assigned_steps)}; "
        f"blocked_steps={len(blocked_steps)}; "
        f"uncovered_p4={len(cascade_result.uncovered_p4_tasks or [])}; "
        f"disposed_p5={len(cascade_result.disposed_p5_tasks or [])}; "
        f"feasible_hard={bool(cascade_result.is_fully_feasible)}; "
        f"uncovered_p4_tasks={unresolved_p4_serialized}; "
        f"disposed_p5_tasks={unresolved_p5_serialized}"
    )

    return reason, reason_code, reason_detail
