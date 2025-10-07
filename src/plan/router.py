from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Callable
from fastapi import APIRouter, HTTPException, Request
import sys
import traceback
import re

from .models import (
    PlanRequest, PlanCandidatesResponse, CandidateOut,
    PlanSolveCascadeRequest, PlanSolveCascadeResponse, AssignmentOut,
    PlanSolveMultiRequest, PlanSolveMultiResponse, PlanSolutionOut, DriverScheduleOut
)

from .config import load_priority_map, load_sla_windows
# from .candidates import generate_candidates, weekday_from_local
from .candidates import weekday_from_local
from .cascade_candidates import generate_cascade_candidates as generate_candidates
from .geo import build_loc_meta_from_locations_csv, enhanced_distance_time_lookup, get_location_coordinates, check_matrix_bidirectional, get_location_coordinates, haversine_between_idx
from .cascade_candidates import generate_cascade_candidates
from .cascade_candidates import (
    generate_cascade_candidates,                 # Existing
    generate_cascade_candidates_with_schedules,  # NEW: Enhanced version
    _compute_cascade_schedules,                  # NEW: For schedule processing  
    _build_ui_schedules_from_cascade,           # NEW: UI format conversion
    _enhanced_to_candidate_out,                 # NEW: Enhanced candidate details
    CascadeCandidateOut                         # NEW: Data class
)

def create_router(
    get_data: Callable[[], Optional[Dict[str, Any]]],
    get_cost_config: Callable[[], Dict[str, float]],
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

    def analyze_duty_insertion_point(
        assignment: AssignmentOut, 
        driver_elements: List[Dict[str, Any]], 
        M: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze where and how a new assignment should be inserted into a driver's duty.
        Returns insertion strategy and metadata for duty reconstruction.
        """
        candidate_id = assignment.candidate_id or ""
        trip_id = assignment.trip_id or ""
        
        # Parse the new route from trip_id
        route_info = parse_route_from_trip_id(trip_id)
        if not route_info:
            return {"strategy": "unknown", "error": "Could not parse route from trip_id"}
        
        from_loc, to_loc = route_info
        
        # Determine insertion strategy based on candidate_id pattern
        if "take_empty@" in candidate_id:
            return analyze_empty_replacement(candidate_id, driver_elements, from_loc, to_loc)
        
        elif "slack@" in candidate_id:
            return analyze_slack_replacement(candidate_id, driver_elements, from_loc, to_loc, M)
        
        elif "swap_leg@" in candidate_id:
            return analyze_leg_swap(candidate_id, driver_elements, from_loc, to_loc)
        
        elif "append" in candidate_id:
            return analyze_duty_append(driver_elements, from_loc, to_loc, M)
        
        else:
            return {"strategy": "unknown", "error": f"Unknown candidate pattern: {candidate_id}"}

    def parse_route_from_trip_id(trip_id: str) -> Optional[Tuple[str, str]]:
        """Extract start and end locations from trip_id"""
        if ":" in trip_id and "->" in trip_id:
            try:
                # Format: "NEW:AMAZON (STN8) MK17 7AB->Midlands Super Hub@2025-09-02T10:30"
                route_part = trip_id.split(":", 1)[1].split("@")[0]
                if "->" in route_part:
                    parts = route_part.split("->", 1)
                    from_loc = parts[0].strip()
                    to_loc = parts[1].strip()
                    
                    # Clean up location names (remove postcodes/codes in parentheses)
                    from_loc = re.sub(r'\s*\([^)]*\)\s*', '', from_loc).strip()
                    to_loc = re.sub(r'\s*\([^)]*\)\s*', '', to_loc).strip()
                    
                    return (from_loc, to_loc)
            except Exception:
                pass
        return None

    def analyze_empty_replacement(
        candidate_id: str, 
        driver_elements: List[Dict[str, Any]], 
        from_loc: str, 
        to_loc: str
    ) -> Dict[str, Any]:
        """Analyze replacement of an empty leg with loaded assignment"""
        
        # Extract timing from candidate_id: "take_empty@540" means start at minute 540
        try:
            start_time = int(candidate_id.split("@")[1])
        except:
            return {"strategy": "error", "error": "Could not parse start time from candidate_id"}
        
        # Find the empty leg that matches this timing and route
        target_element = None
        target_index = None
        
        for i, element in enumerate(driver_elements):
            if (element.get("is_travel") and 
                element.get("start_min") == start_time and
                element.get("from", "").upper() == from_loc.upper() and
                element.get("to", "").upper() == to_loc.upper()):
                target_element = element
                target_index = i
                break
        
        if target_element is None:
            return {"strategy": "error", "error": f"Could not find empty leg at time {start_time}"}
        
        return {
            "strategy": "empty_replacement",
            "insertion_index": target_index,
            "target_element": target_element,
            "new_route": (from_loc, to_loc),
            "simple_swap": True,  # Just change load status
            "description": f"Replace empty leg {from_loc} -> {to_loc} with loaded assignment"
        }

    def analyze_slack_replacement(
        candidate_id: str, 
        driver_elements: List[Dict[str, Any]], 
        from_loc: str, 
        to_loc: str,
        M: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze replacement of AS DIRECTED time with positioned pickup + delivery"""
        
        # Extract location from candidate_id: "slack@17" means using slack at location index 17
        try:
            slack_location_id = int(candidate_id.split("@")[1])
        except:
            return {"strategy": "error", "error": "Could not parse location from slack candidate_id"}
        
        # Find the AS DIRECTED element
        slack_element = None
        slack_index = None
        
        for i, element in enumerate(driver_elements):
            element_type = str(element.get("element_type", "")).upper()
            if "AS DIRECTED" in element_type:
                slack_element = element
                slack_index = i
                break
        
        if slack_element is None:
            return {"strategy": "error", "error": "Could not find AS DIRECTED element"}
        
        # Get slack location name
        slack_location = get_location_name_by_id(slack_location_id, M)
        if not slack_location:
            return {"strategy": "error", "error": f"Could not resolve location ID {slack_location_id}"}
        
        return {
            "strategy": "slack_replacement",
            "insertion_index": slack_index,
            "slack_element": slack_element,
            "slack_location": slack_location,
            "new_route": (from_loc, to_loc),
            "positioning_required": True,
            "sequence": [
                {"type": "positioning", "from": slack_location, "to": from_loc, "load": "EMPTY"},
                {"type": "assignment", "from": from_loc, "to": to_loc, "load": "LOADED"},
                {"type": "return", "from": to_loc, "to": "TBD", "load": "EMPTY"}  # TBD = to be determined
            ],
            "description": f"Replace AS DIRECTED time with: {slack_location} -> {from_loc} (empty) -> {to_loc} (loaded) -> return"
        }

    def analyze_leg_swap(
        candidate_id: str, 
        driver_elements: List[Dict[str, Any]], 
        from_loc: str, 
        to_loc: str
    ) -> Dict[str, Any]:
        """Analyze swapping an existing leg with new assignment"""
        
        # Extract timing from candidate_id: "swap_leg@600" 
        try:
            start_time = int(candidate_id.split("@")[1])
        except:
            return {"strategy": "error", "error": "Could not parse start time from swap candidate_id"}
        
        # Find the leg being swapped
        target_element = None
        target_index = None
        
        for i, element in enumerate(driver_elements):
            if (element.get("is_travel") and 
                element.get("start_min") == start_time):
                target_element = element
                target_index = i
                break
        
        if target_element is None:
            return {"strategy": "error", "error": f"Could not find leg to swap at time {start_time}"}
        
        original_route = (target_element.get("from", ""), target_element.get("to", ""))
        
        return {
            "strategy": "leg_swap",
            "insertion_index": target_index,
            "target_element": target_element,
            "original_route": original_route,
            "new_route": (from_loc, to_loc),
            "displacement_required": True,  # The original leg needs to be reassigned
            "description": f"Swap {original_route[0]} -> {original_route[1]} with {from_loc} -> {to_loc}"
        }

    def analyze_duty_append(
        driver_elements: List[Dict[str, Any]], 
        from_loc: str, 
        to_loc: str,
        M: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze appending new work to end of duty"""
        
        # Find the last travel element and END FACILITY
        last_travel = None
        last_travel_index = None
        end_facility_index = None
        
        for i, element in enumerate(driver_elements):
            if element.get("is_travel"):
                last_travel = element
                last_travel_index = i
            elif str(element.get("element_type", "")).upper() == "END FACILITY":
                end_facility_index = i
        
        if last_travel is None:
            return {"strategy": "error", "error": "Could not find any travel elements to append after"}
        
        last_location = last_travel.get("to", "")
        
        return {
            "strategy": "duty_append",
            "insertion_index": end_facility_index or len(driver_elements),  # Insert before END FACILITY
            "last_travel_element": last_travel,
            "last_location": last_location,
            "new_route": (from_loc, to_loc),
            "positioning_required": last_location.upper() != from_loc.upper(),
            "sequence": [
                {"type": "positioning", "from": last_location, "to": from_loc, "load": "EMPTY"},
                {"type": "assignment", "from": from_loc, "to": to_loc, "load": "LOADED"},
                {"type": "return", "from": to_loc, "to": "TBD", "load": "EMPTY"}  # Return to base
            ],
            "description": f"Append after current duty: {last_location} -> {from_loc} (empty) -> {to_loc} (loaded) -> base"
        }

    def get_location_name_by_id(location_id: int, M: Dict[str, Any]) -> Optional[str]:
        """Get location name from location ID using the matrices lookup"""
        loc2idx = M.get("loc2idx", {})
        for name, idx in loc2idx.items():
            if int(idx) == int(location_id):
                return name
        return None

    def _compute_before_after_schedules(
        DATA: Dict[str, Any],
        assignments: List[AssignmentOut],
        cascades: List[Dict[str, Any]],
    ) -> List[DriverScheduleOut]:
        """
        UPDATED: Build per-driver schedules using the new duty analysis functions.
        """
        # Import the enhanced lookup functions
        from .geo import enhanced_distance_time_lookup, build_loc_meta_from_locations_csv
        
        def debug_log(msg):
            print(f"DEBUG SCHEDULE: {msg}", file=sys.stderr, flush=True)
        
        debug_log(f"Processing {len(assignments)} assignments and {len(cascades)} cascades")
        
        ds = DATA.get("driver_states", {})
        drivers = ds.get("drivers", ds) if isinstance(ds, dict) else {}

        # Get affected drivers from assignments and cascades
        affected_driver_ids = set()
        for a in assignments:
            if a.driver_id:
                affected_driver_ids.add(a.driver_id)
        for c in cascades:
            if c.get("driver_id"):
                affected_driver_ids.add(c["driver_id"])
        
        debug_log(f"Affected drivers: {affected_driver_ids}")
        
        # Safety limit
        if len(affected_driver_ids) > 10:
            affected_driver_ids = set(list(affected_driver_ids)[:10])
        
        schedule_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        # Initialize with original schedules for affected drivers
        for drv_id in affected_driver_ids:
            if drv_id not in drivers:
                debug_log(f"WARNING: Driver {drv_id} not found in driver_states")
                continue
                
            meta = drivers[drv_id]
            before = list(meta.get("elements", []))
            after = [dict(e) for e in before]  # Deep copy for modification
            schedule_map[drv_id] = {"before": before, "after": after}
            debug_log(f"Driver {drv_id} has {len(before)} original elements")

        # Build matrices reference for duty analysis
        M = {
            "dist": DATA["distance"],
            "time": DATA["time"],
            "loc2idx": DATA["location_to_index"],
        }
        
        # Build location metadata for coordinate lookups
        loc_meta: Dict[str, Dict[str, Any]] = {}
        try:
            if DATA.get("locations_df") is not None:
                loc_meta = build_loc_meta_from_locations_csv(DATA["locations_df"])
                debug_log(f"Built location metadata for {len(loc_meta)} locations")
        except Exception as e:
            debug_log(f"WARNING: Could not build location metadata: {e}")

        # Process cascades first (remove displaced legs)
        debug_log(f"Processing {len(cascades)} cascades...")
        for c in cascades:
            driver_id = c.get("driver_id")
            from_loc = str(c.get("from", "")).upper()
            to_loc = str(c.get("to", "")).upper()
            
            if driver_id and from_loc and to_loc and driver_id in schedule_map:
                after_schedule = schedule_map[driver_id]["after"]
                
                # Remove the displaced leg
                removed = False
                for idx in range(len(after_schedule) - 1, -1, -1):
                    e = after_schedule[idx]
                    if (e.get("is_travel") and 
                        str(e.get("from", "")).upper() == from_loc and 
                        str(e.get("to", "")).upper() == to_loc):
                        del after_schedule[idx]
                        debug_log(f"REMOVED displaced leg: {from_loc} → {to_loc} from {driver_id}")
                        removed = True
                        break
                if not removed:
                    debug_log(f"WARNING: Could not find leg to remove: {from_loc} → {to_loc} from {driver_id}")

        # Process assignments (reconstruct duties with new work)
        debug_log(f"Processing {len(assignments)} assignments...")
        for i, assignment in enumerate(assignments):
            debug_log(f"Assignment {i}: {assignment.trip_id} -> Driver {assignment.driver_id} ({assignment.type})")
            
            if assignment.type != "reassigned" or not assignment.driver_id:
                debug_log(f"SKIPPING assignment {i} - type: {assignment.type}, driver: {assignment.driver_id}")
                continue
                
            driver_id = assignment.driver_id
            if driver_id not in schedule_map:
                debug_log(f"WARNING: Driver {driver_id} not in schedule_map")
                continue
                
            debug_log(f"Processing assignment for driver {driver_id}")
            debug_log(f"Trip ID: {assignment.trip_id}")
            debug_log(f"Candidate ID: {assignment.candidate_id}")
            
            # Get current schedule state
            driver_elements = schedule_map[driver_id]["after"]
            debug_log(f"Driver {driver_id} has {len(driver_elements)} elements before insertion")
            
            # Try insertion analysis
            try:
                insertion_analysis = analyze_duty_insertion_point(assignment, driver_elements, M)
                debug_log(f"Insertion analysis result: {insertion_analysis.get('strategy')}")
                
                if insertion_analysis.get("strategy") == "error":
                    debug_log(f"Insertion analysis failed: {insertion_analysis.get('error')}")
                    debug_log("Falling back to simple append")
                    _fallback_append_assignment(assignment, schedule_map[driver_id]["after"], M, loc_meta)
                    continue
                
                # Apply the insertion strategy (now with loc_meta)
                debug_log(f"Applying strategy: {insertion_analysis.get('strategy')}")
                success = _apply_insertion_strategy(assignment, insertion_analysis, schedule_map[driver_id]["after"], M, loc_meta)
                
                if not success:
                    debug_log("Insertion strategy failed, falling back to append")
                    _fallback_append_assignment(assignment, schedule_map[driver_id]["after"], M, loc_meta)
                else:
                    debug_log("Successfully applied insertion strategy")
                    
            except Exception as e:
                debug_log(f"ERROR: Exception in insertion analysis: {e}")
                debug_log("Falling back to simple append")
                _fallback_append_assignment(assignment, schedule_map[driver_id]["after"], M, loc_meta)
            
            # Check final state
            final_elements = len(schedule_map[driver_id]["after"])
            debug_log(f"Driver {driver_id} now has {final_elements} elements after processing")

        # Return driver schedules
        out: List[DriverScheduleOut] = []
        for drv_id, schedule_data in schedule_map.items():
            debug_log(f"Final schedule for {drv_id}: {len(schedule_data['before'])} -> {len(schedule_data['after'])} elements")
            out.append(DriverScheduleOut(
                driver_id=drv_id, 
                before=schedule_data["before"],
                after=schedule_data["after"]
            ))
        
        debug_log("Schedule computation completed")
        return out

    def _apply_insertion_strategy(
        assignment: AssignmentOut, 
        analysis: Dict[str, Any], 
        after_schedule: List[Dict[str, Any]], 
        M: Dict[str, Any],
        loc_meta: Dict[str, Any]  # Added parameter
    ) -> bool:
        """Apply the insertion strategy to reconstruct the duty."""
        
        strategy = analysis.get("strategy")
        new_route = analysis.get("new_route")
        
        if not new_route:
            return False
            
        from_loc, to_loc = new_route
        
        try:
            if strategy == "empty_replacement":
                return _apply_empty_replacement(assignment, analysis, after_schedule)
                
            elif strategy == "slack_replacement":
                return _apply_slack_replacement(assignment, analysis, after_schedule, M, loc_meta)  # Added loc_meta
                
            elif strategy == "leg_swap":
                return _apply_leg_swap(assignment, analysis, after_schedule, M, loc_meta)  # Added loc_meta
                
            elif strategy == "duty_append":
                return _apply_duty_append(assignment, analysis, after_schedule, M, loc_meta)  # Added loc_meta
                
            else:
                print(f"Unknown insertion strategy: {strategy}", file=sys.stderr, flush=True)
                return False
                
        except Exception as e:
            print(f"Error applying {strategy}: {e}", file=sys.stderr, flush=True)
            return False

    def _apply_empty_replacement(
        assignment: AssignmentOut, 
        analysis: Dict[str, Any], 
        after_schedule: List[Dict[str, Any]]
    ) -> bool:
        """Replace an empty leg with the new loaded assignment."""
        
        insertion_index = analysis.get("insertion_index")
        target_element = analysis.get("target_element")
        new_route = analysis.get("new_route")
        
        if insertion_index is None or not target_element or not new_route:
            return False
            
        from_loc, to_loc = new_route
        
        # Update the existing element to be loaded
        new_element = dict(target_element)
        new_element.update({
            "from": from_loc,
            "to": to_loc,
            "priority": 1,  # High priority for new assignment
            "load_type": "URGENT_DELIVERY",
            "planz_code": "NEW_ASSIGNMENT",
            "note": f"🔄 LOADED: {from_loc} → {to_loc} (was empty)"
        })
        
        after_schedule[insertion_index] = new_element
        print(f"✅ Empty replacement: {from_loc} → {to_loc}")
        return True

    def _apply_slack_replacement(
        assignment: AssignmentOut, 
        analysis: Dict[str, Any], 
        after_schedule: List[Dict[str, Any]], 
        M: Dict[str, Any],
        loc_meta: Dict[str, Any]
    ) -> bool:
        """Replace AS DIRECTED time with complete pickup + delivery sequence."""
        
        insertion_index = analysis.get("insertion_index")
        slack_element = analysis.get("slack_element")
        slack_location = analysis.get("slack_location")
        new_route = analysis.get("new_route")
        
        if insertion_index is None or not slack_element or not slack_location or not new_route:
            return False
            
        from_loc, to_loc = new_route
        slack_start = slack_element.get("start_min")
        
        # GET LOC_META from DATA (you'll need to pass this in)
        # For now, use empty dict - you may need to modify function signature
        loc_meta = {}  # TODO: Pass this from caller
        
        current_time = slack_start
        sequence = []
        
        # 1. Positioning to pickup (if needed)
        if slack_location.upper() != from_loc.upper():
            pos_miles, pos_time, pos_warning = enhanced_distance_time_lookup(
                slack_location, from_loc, M, loc_meta
            )
            
            sequence.append({
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": slack_location,
                "to": from_loc,
                "start_min": current_time,
                "end_min": current_time + pos_time,
                "duration_min": pos_time,
                "miles": pos_miles,
                "priority": 2,
                "load_type": "EMPTY",
                "planz_code": "POSITIONING",
                "note": f"🚚 POSITIONING: {slack_location} → {from_loc} {pos_warning}".strip()
            })
            current_time += pos_time
        
        # 2. Loading at pickup
        sequence.append({
            "element_type": "LOAD(ASSIST)",
            "is_travel": False,
            "from": from_loc,
            "to": from_loc,
            "start_min": current_time,
            "end_min": current_time + 30,
            "duration_min": 30,
            "miles": 0.0,
            "priority": 1,
            "load_type": "LOADING",
            "planz_code": "LOAD_ASSIST",
            "note": f"📦 LOADING at {from_loc}"
        })
        current_time += 30
        
        # 3. Delivery leg  
        del_miles, del_time, del_warning = enhanced_distance_time_lookup(
            from_loc, to_loc, M, loc_meta
        )
        
        sequence.append({
            "element_type": "TRAVEL", 
            "is_travel": True,
            "from": from_loc,
            "to": to_loc,
            "start_min": current_time,
            "end_min": current_time + del_time,
            "duration_min": del_time,
            "miles": del_miles,
            "priority": 1,
            "load_type": "URGENT_DELIVERY",
            "planz_code": "NEW_ASSIGNMENT", 
            "note": f"📦 LOADED: {from_loc} → {to_loc} {del_warning}".strip()
        })
        current_time += del_time
        
        # 4. Unloading at delivery
        sequence.append({
            "element_type": "UNLOAD(ASSIST)",
            "is_travel": False,
            "from": to_loc,
            "to": to_loc,
            "start_min": current_time,
            "end_min": current_time + 30,
            "duration_min": 30,
            "miles": 0.0,
            "priority": 1,
            "load_type": "UNLOADING", 
            "planz_code": "UNLOAD_ASSIST",
            "note": f"📦 UNLOADING at {to_loc}"
        })
        
        # Replace the AS DIRECTED element
        del after_schedule[insertion_index]
        for i, element in enumerate(sequence):
            after_schedule.insert(insertion_index + i, element)
            
        print(f"✅ Slack replacement: Complete sequence with enhanced lookup", file=sys.stderr)
        return True

    def _apply_leg_swap(
        assignment: AssignmentOut, 
        analysis: Dict[str, Any], 
        after_schedule: List[Dict[str, Any]], 
        M: Dict[str, Any],
        loc_meta: Dict[str, Any]
    ) -> bool:
        """Swap an existing leg with the new assignment."""
        
        insertion_index = analysis.get("insertion_index")
        target_element = analysis.get("target_element")
        original_route = analysis.get("original_route")
        new_route = analysis.get("new_route")
        
        if insertion_index is None or not target_element or not new_route:
            return False
            
        from_loc, to_loc = new_route
        orig_from, orig_to = original_route
        
        # LOOK UP ACTUAL DISTANCE AND TIME
        loc2idx = M["loc2idx"]
        dist_matrix = M["dist"]
        time_matrix = M["time"]
        
        def get_distance_time(from_name: str, to_name: str):
            try:
                i = loc2idx.get(from_name.upper())
                j = loc2idx.get(to_name.upper()) 
                if i is not None and j is not None:
                    return float(dist_matrix[i, j]), float(time_matrix[i, j])
            except Exception:
                pass
            return 0.0, 30.0  # Fallback
        
        new_dist, new_time = get_distance_time(from_loc, to_loc)
        
        # Replace the existing leg
        new_element = {
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": from_loc,
            "to": to_loc,
            "start_min": target_element.get("start_min"),
            "end_min": target_element.get("start_min", 0) + new_time,  # Calculate end time
            "duration_min": new_time,
            "miles": new_dist,
            "priority": 1,
            "load_type": "URGENT_DELIVERY",
            "planz_code": "NEW_ASSIGNMENT",
            "note": f"🔄 SWAPPED: {orig_from}→{orig_to} with {from_loc}→{to_loc}"
        }
        
        after_schedule[insertion_index] = new_element
        print(f"✅ Leg swap: {orig_from}→{orig_to} replaced with {from_loc}→{to_loc} ({new_time}min, {new_dist:.1f}mi)", file=sys.stderr)
        return True
    
    def _apply_duty_append(
        assignment: AssignmentOut, 
        analysis: Dict[str, Any], 
        after_schedule: List[Dict[str, Any]], 
        M: Dict[str, Any],
        loc_meta: Dict[str, Any]
    ) -> bool:
        """Append new work to the end of the duty with complete operational sequence."""
        
        from .geo import enhanced_distance_time_lookup
        
        insertion_index = analysis.get("insertion_index", len(after_schedule))
        last_location = analysis.get("last_location")
        new_route = analysis.get("new_route")
        positioning_required = analysis.get("positioning_required", False)
        
        if not new_route:
            return False
            
        from_loc, to_loc = new_route
        
        # Find the end time of the last travel element
        current_time = None
        for element in reversed(after_schedule):
            if element.get("is_travel") and element.get("end_min") is not None:
                current_time = element.get("end_min")
                break
        
        if current_time is None:
            current_time = 480  # 8 AM fallback
            print(f"WARNING: Could not find last travel time, using 8 AM fallback", file=sys.stderr)
        
        # Build the complete sequence
        append_elements = []
        
        # 1. Positioning/deadhead leg (if needed)
        if positioning_required and last_location and last_location.upper() != from_loc.upper():
            pos_miles, pos_time, pos_warning = enhanced_distance_time_lookup(
                last_location, from_loc, M, loc_meta
            )
            
            append_elements.append({
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": last_location,
                "to": from_loc,
                "start_min": current_time,
                "end_min": current_time + pos_time,
                "duration_min": pos_time,
                "miles": pos_miles,
                "priority": 2,
                "load_type": "EMPTY",
                "planz_code": "POSITIONING",
                "note": f"🚚 DEADHEAD: {last_location} → {from_loc} {pos_warning}".strip()
            })
            current_time += pos_time
        
        # 2. Loading time at pickup location
        append_elements.append({
            "element_type": "LOAD(ASSIST)",
            "is_travel": False,
            "from": from_loc,
            "to": from_loc,  # Same location for loading
            "start_min": current_time,
            "end_min": current_time + 30,  # 30 min loading time
            "duration_min": 30,
            "miles": 0.0,
            "priority": 1,
            "load_type": "LOADING",
            "planz_code": "LOAD_ASSIST",
            "note": f"📦 LOADING at {from_loc}"
        })
        current_time += 30
        
        # 3. Main delivery leg (loaded)
        del_miles, del_time, del_warning = enhanced_distance_time_lookup(
            from_loc, to_loc, M, loc_meta
        )
        
        append_elements.append({
            "element_type": "TRAVEL",
            "is_travel": True,
            "from": from_loc,
            "to": to_loc,
            "start_min": current_time,
            "end_min": current_time + del_time,
            "duration_min": del_time,
            "miles": del_miles,
            "priority": 1,
            "load_type": "URGENT_DELIVERY",
            "planz_code": "NEW_ASSIGNMENT",
            "note": f"📦 LOADED: {from_loc} → {to_loc} {del_warning}".strip()
        })
        current_time += del_time
        
        # 4. Unloading time at delivery location
        append_elements.append({
            "element_type": "UNLOAD(ASSIST)",
            "is_travel": False,
            "from": to_loc,
            "to": to_loc,  # Same location for unloading
            "start_min": current_time,
            "end_min": current_time + 30,  # 30 min unloading time
            "duration_min": 30,
            "miles": 0.0,
            "priority": 1,
            "load_type": "UNLOADING",
            "planz_code": "UNLOAD_ASSIST",
            "note": f"📦 UNLOADING at {to_loc}"
        })
        current_time += 30
        
        # 5. Return to base (if END FACILITY exists)
        end_facility_location = None
        for element in after_schedule:
            if str(element.get("element_type", "")).upper() == "END FACILITY":
                end_facility_location = element.get("from")
                break
        
        if end_facility_location and end_facility_location.upper() != to_loc.upper():
            ret_miles, ret_time, ret_warning = enhanced_distance_time_lookup(
                to_loc, end_facility_location, M, loc_meta
            )
            
            append_elements.append({
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": to_loc,
                "to": end_facility_location,
                "start_min": current_time,
                "end_min": current_time + ret_time,
                "duration_min": ret_time,
                "miles": ret_miles,
                "priority": 2,
                "load_type": "EMPTY",
                "planz_code": "RETURN_TO_BASE",
                "note": f"🏠 RETURN: {to_loc} → {end_facility_location} {ret_warning}".strip()
            })
            current_time += ret_time
        
        # Insert before END FACILITY
        end_facility_index = len(after_schedule)
        for i, element in enumerate(after_schedule):
            if str(element.get("element_type", "")).upper() == "END FACILITY":
                end_facility_index = i
                break
        
        # Insert all elements
        for i, element in enumerate(append_elements):
            after_schedule.insert(end_facility_index + i, element)
            
        # Update END FACILITY time
        for element in after_schedule:
            if str(element.get("element_type", "")).upper() == "END FACILITY":
                element["start_min"] = current_time
                element["end_min"] = current_time
                break
            
        print(f"✅ Duty append: Complete sequence {from_loc} → {to_loc} ({len(append_elements)} elements)", file=sys.stderr)
        return True

    def _fallback_append_assignment(
        assignment: AssignmentOut, 
        after_schedule: List[Dict[str, Any]],
        M: Dict[str, Any],
        loc_meta: Dict[str, Any]  # Add this parameter
    ):
        """Fallback: simple append when insertion analysis fails."""
        
        # Parse route from trip_id
        trip_id = assignment.trip_id
        from_loc = to_loc = None
        
        if ":" in trip_id and "->" in trip_id:
            route_part = trip_id.split(":", 1)[1].split("@")[0]
            if "->" in route_part:
                parts = route_part.split("->", 1)
                from_loc = parts[0].strip()
                to_loc = parts[1].strip()
                
                # Clean up location names
                import re
                from_loc = re.sub(r'\s*\([^)]*\)\s*', '', from_loc).strip()
                to_loc = re.sub(r'\s*\([^)]*\)\s*', '', to_loc).strip()
        
        if from_loc and to_loc:
            fallback_element = {
                "element_type": "TRAVEL",
                "is_travel": True,
                "from": from_loc,
                "to": to_loc,
                "start_min": None,
                "end_min": None,
                "priority": 1,
                "load_type": "URGENT_DELIVERY",
                "planz_code": "NEW_ASSIGNMENT",
                "note": f"⚠️ FALLBACK: {from_loc} → {to_loc}"
            }
            after_schedule.append(fallback_element)
            print(f"⚠️ Fallback append: {from_loc} → {to_loc}")
    
# ------------------- Endpoints -------------------

    @router.post("/candidates", response_model=PlanCandidatesResponse)
    def plan_candidates(req: PlanRequest):
        """
        UPDATED: Now returns schedule data for UI visualization
        """
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        
        # Use enhanced version that returns schedules
        weekday, trip_minutes, trip_miles, cands, schedules = generate_cascade_candidates_with_schedules(
            req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
            max_cascade_depth=2, 
            max_candidates=10
        )
        
        # IMPORTANT: PlanCandidatesResponse needs to be updated to include schedules
        # See Fix 2 below for the model update
        return PlanCandidatesResponse(
            weekday=weekday, 
            trip_minutes=trip_minutes, 
            trip_miles=trip_miles, 
            candidates=cands,
            schedules=schedules  # NEW: Add schedule data
        )

    @router.post("/solve_cascades", response_model=PlanSolveCascadeResponse)
    def plan_and_solve_cascades_enhanced(req: PlanSolveCascadeRequest, request: Request):
        """Enhanced cascade solving with proper UI schedule output"""
        
        DATA, M, LOC_META = ensure_ready()
        cfg = get_cost_config()
        
        # Check if user selected a specific candidate
        if req.preferred_driver_id:
            print(f"[solve_cascades] Preferred driver specified: {req.preferred_driver_id}")
            
            # Generate candidates but filter to only the preferred driver
            weekday, trip_minutes, trip_miles, all_candidates, all_schedules = generate_cascade_candidates_with_schedules(
                req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
                max_cascade_depth=req.max_cascades,
                max_candidates=20  # Generate more to find the preferred one
            )
            
            # Filter to only the preferred candidate
            candidates = [c for c in all_candidates if c.driver_id == req.preferred_driver_id]
            schedules = [s for s in all_schedules if s.get('driver_id') == req.preferred_driver_id]
            
            if not candidates:
                # Fallback: if preferred not found, use all candidates
                print(f"[solve_cascades] Warning: Preferred driver {req.preferred_driver_id} not found in candidates")
                candidates = all_candidates[:1]  # Just take first
                schedules = all_schedules[:1]
        else:
            # No preference - return all candidates
            weekday, trip_minutes, trip_miles, candidates, schedules = generate_cascade_candidates_with_schedules(
                req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
                max_cascade_depth=req.max_cascades,
                max_candidates=req.max_cascades
            )
        
        if not candidates:
            return PlanSolveCascadeResponse(
                weekday=weekday,
                trip_minutes=trip_minutes,
                trip_miles=trip_miles,
                objective_value=0.0,
                assignments=[],
                details={"backend": "cascade-cuopt", "error": "no_candidates"},
                candidates_considered=0,
                cascades=[],
                schedules=[]
            )
        
        # Build trip_id from request
        trip_id = f"NEW-{req.start_location}→{req.end_location}@{req.when_local}"
        
        # Build assignments from filtered candidates
        assignments = []
        total_cost = 0.0
        
        for candidate in candidates:
            assignment = AssignmentOut(
                trip_id=trip_id,
                type="reassigned",
                driver_id=candidate.driver_id,
                candidate_id=candidate.candidate_id,
                cost=candidate.est_cost,
                deadhead_miles=candidate.deadhead_miles,
                overtime_minutes=candidate.overtime_minutes,
                delay_minutes=candidate.delay_minutes,
                miles_delta=candidate.miles_delta,
                uses_emergency_rest=candidate.uses_emergency_rest
            )
            assignments.append(assignment)
            total_cost += candidate.est_cost
        
        # Build cascade information
        cascades = []
        for candidate in candidates:
            cascades.append({
                "depth": 1,
                "displaced_by": "NEW_SERVICE",
                "driver_id": candidate.driver_id,
                "from": req.start_location,
                "to": req.end_location,
                "priority": req.priority,
                "reason": candidate.reason or "Enhanced cascade"
            })
        
        print(f"[solve_cascades] Returning {len(candidates)} candidates with {len(schedules)} schedules")
        
        return PlanSolveCascadeResponse(
            weekday=weekday,
            trip_minutes=trip_minutes,
            trip_miles=trip_miles,
            objective_value=total_cost,
            assignments=assignments,
            details={
                "backend": "cascade-cuopt-enhanced",
                "max_cascades": req.max_cascades,
                "drivers_touched": len(set(a.driver_id for a in assignments)),
                "preferred_driver": req.preferred_driver_id
            },
            candidates_considered=len(candidates),
            cascades=cascades,
            schedules=schedules  # Only the filtered schedule(s)
        )

    # @router.post("/solve_cascades", response_model=PlanSolveCascadeResponse)
    # def plan_and_solve_cascades_enhanced(req: PlanSolveCascadeRequest, request: Request):
    #     """
    #     Enhanced cascade solving with proper UI schedule output
    #     """
        
    #     DATA, M, LOC_META = ensure_ready()
    #     cfg = get_cost_config()
        
    #     # ✅ Already using enhanced version - good!
    #     weekday, trip_minutes, trip_miles, candidates, schedules = generate_cascade_candidates_with_schedules(
    #         req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
    #         max_cascade_depth=req.max_cascades,
    #         max_candidates=req.max_cascades
    #     )
        
    #     if not candidates:
    #         return PlanSolveCascadeResponse(
    #             weekday=weekday,
    #             trip_minutes=trip_minutes,
    #             trip_miles=trip_miles,
    #             objective_value=0.0,
    #             assignments=[],
    #             details={"backend": "cascade-cuopt", "error": "no_candidates"},
    #             candidates_considered=0,
    #             cascades=[],
    #             schedules=[]  # Return empty schedules
    #         )
        
    #     # Build trip_id from request
    #     trip_id = f"NEW-{req.start_location}→{req.end_location}@{req.when_local}"
        
    #     # Build assignments from candidates
    #     assignments = []
    #     total_cost = 0.0
        
    #     for candidate in candidates:
    #         assignment = AssignmentOut(
    #             trip_id=trip_id,
    #             type="reassigned",
    #             driver_id=candidate.driver_id,
    #             candidate_id=candidate.candidate_id,
    #             cost=candidate.est_cost,
    #             deadhead_miles=candidate.deadhead_miles,
    #             overtime_minutes=candidate.overtime_minutes,
    #             delay_minutes=candidate.delay_minutes,
    #             miles_delta=candidate.miles_delta,
    #             uses_emergency_rest=candidate.uses_emergency_rest
    #         )
    #         assignments.append(assignment)
    #         total_cost += candidate.est_cost
        
    #     # Build cascade information
    #     cascades = []
    #     for i, candidate in enumerate(candidates):
    #         cascades.append({
    #             "depth": 1,
    #             "displaced_by": "NEW_SERVICE",
    #             "driver_id": candidate.driver_id,
    #             "from": req.start_location,
    #             "to": req.end_location,
    #             "priority": req.priority,
    #             "reason": candidate.reason or "Enhanced cascade"
    #         })
        
    #     return PlanSolveCascadeResponse(
    #         weekday=weekday,
    #         trip_minutes=trip_minutes,
    #         trip_miles=trip_miles,
    #         objective_value=total_cost,
    #         assignments=assignments,
    #         details={
    #             "backend": "cascade-cuopt-enhanced",
    #             "max_cascades": req.max_cascades,
    #             "drivers_touched": len(set(a.driver_id for a in assignments))
    #         },
    #         candidates_considered=len(candidates),
    #         cascades=cascades,
    #         schedules=schedules  # ✅ Schedules already being returned
    #     )

    # @router.post("/solve_cascades", response_model=PlanSolveCascadeResponse)
    # def plan_and_solve_cascades_enhanced(req: PlanSolveCascadeRequest, request: Request):
    #     """Enhanced cascade solving with proper UI schedule output"""
        
    #     DATA, M, LOC_META = ensure_ready()
    #     cfg = get_cost_config()
        
    #     # Use enhanced version that returns schedules
    #     weekday, trip_minutes, trip_miles, candidates, schedules = generate_cascade_candidates_with_schedules(
    #         req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 
    #         max_cascade_depth=req.max_cascades,
    #         max_candidates=req.max_cascades
    #     )
        
    #     if not candidates:
    #         return PlanSolveCascadeResponse(
    #             weekday=weekday,
    #             trip_minutes=trip_minutes,
    #             trip_miles=trip_miles,
    #             objective_value=0.0,
    #             assignments=[],
    #             details={"backend": "cascade-cuopt", "error": "no_candidates"},
    #             candidates_considered=0,
    #             cascades=[],
    #             schedules=[]
    #         )
        
    #     # Build trip_id from request
    #     trip_id = f"NEW:{req.start_location}->{req.end_location}@{req.when_local}"
        
    #     # Build assignments from candidates
    #     assignments = []
    #     total_cost = 0.0
        
    #     for candidate in candidates:
    #         # ✅ FIXED: Include required trip_id and type fields
    #         assignment = AssignmentOut(
    #             trip_id=trip_id,                    # ✅ REQUIRED: Trip identifier
    #             type="reassigned",                  # ✅ REQUIRED: Assignment type
    #             driver_id=candidate.driver_id,
    #             candidate_id=candidate.candidate_id,
    #             cost=candidate.est_cost,
    #             deadhead_miles=candidate.deadhead_miles,
    #             overtime_minutes=candidate.overtime_minutes,
    #             delay_minutes=candidate.delay_minutes,
    #             miles_delta=candidate.miles_delta,
    #             uses_emergency_rest=candidate.uses_emergency_rest
    #         )
    #         assignments.append(assignment)
    #         total_cost += candidate.est_cost
        
    #     # Build cascade information
    #     cascades = []
    #     for i, candidate in enumerate(candidates):
    #         cascades.append({
    #             "depth": 1,
    #             "displaced_by": "NEW_SERVICE",
    #             "driver_id": candidate.driver_id,
    #             "from": req.start_location,
    #             "to": req.end_location,
    #             "priority": req.priority,
    #             "reason": candidate.reason or "Enhanced cascade"
    #         })
        
    #     return PlanSolveCascadeResponse(
    #         weekday=weekday,
    #         trip_minutes=trip_minutes,
    #         trip_miles=trip_miles,
    #         objective_value=total_cost,
    #         assignments=assignments,
    #         details={
    #             "backend": "cascade-cuopt-enhanced",
    #             "max_cascades": req.max_cascades,
    #             "drivers_touched": len(set(a.driver_id for a in assignments))
    #         },
    #         candidates_considered=len(candidates),
    #         cascades=cascades,
    #         schedules=schedules  # ✅ CRITICAL: Pass schedules to UI
    #     )

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
            wk, tmn, tmi, cands = generate_candidates(cand_req, DATA, M, cfg, LOC_META, SLA_WINDOWS, 50.0)
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

        # # Optional: refine with cuOpt if requested
        # if req.use_cuopt and solutions:
        #     try:
        #         cuopt_url = get_cuopt_url()
        #         refined = []
        #         for s in solutions[: req.max_solutions]:
        #             # Build one payload per candidate (you can also batch if your cuOpt supports)
        #             payload = build_cuopt_payload(
        #                 DATA=DATA,
        #                 request_trip=root_trip,
        #                 assignments_so_far=[a.dict() for a in s.assignments],
        #                 priorities=PRIORITY_MAP,
        #                 sla_windows=SLA_WINDOWS,
        #                 M=M,
        #                 new_req_window=[earliest, latest],  # pass down
        #             )
        #             raw = solve_with_cuopt(cuopt_url, payload)
        #             variants = extract_solutions_from_cuopt(raw, max_solutions=1)
        #             if variants:
        #                 v = variants[0]
        #                 s.objective_value = float(v.get("objective_value", s.objective_value))
        #                 s.details["backend"] = "cuopt"
        #         # resort after refinement
        #         solutions.sort(key=lambda s: (s.objective_value, s.drivers_touched))
        #         for r, s in enumerate(solutions, 1):
        #             s.rank = r
        #     except Exception as e:
        #         # Non-fatal: keep heuristic results
        #         pass

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
# Force reload
