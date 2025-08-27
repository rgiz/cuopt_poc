# backend/cuopt_vrp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math

# --- simple containers (no external deps) ---

@dataclass
class Vehicle:
    vehicle_id: str
    start_index: int
    end_index: int
    start_min: int
    end_min: int
    max_overtime_min: int = 0

@dataclass
class Shipment:
    pickup_index: int
    delivery_index: int
    pickup_tw: Tuple[int, int]     # minutes since 00:00 local (same day)
    delivery_tw: Tuple[int, int]
    service_time_pickup: int = 0
    service_time_delivery: int = 0
    priority: int = 3              # 1..5 (1 = fastest/SLA-tightest)

# --- translator API you call from CuOptModel ---

def build_vehicles_from_driver_states(
    driver_states: Dict[str, Any],
    weekday: str,                 # "Mon".."Sun"
    loc2idx: Dict[str, int],
    max_overtime_per_duty_min: int = 120,
) -> List[Vehicle]:
    """
    Simplified: one vehicle per duty_id with start at first element.from_id and end at last element.to_id.
    Uses daily window from driver_states['daily_windows'][weekday].
    """
    out: List[Vehicle] = []
    for duty_id, d in driver_states.items():
        days = set(d.get("days", []))
        if weekday not in days:
            continue

        elems = d.get("elements", [])
        if not elems:
            continue

        start_idx = elems[0].get("from_id")
        end_idx = elems[-1].get("to_id")
        if start_idx is None or end_idx is None:
            continue

        dw = (d.get("daily_windows") or {}).get(weekday)
        if not dw:
            continue

        start_min = int(dw.get("start_min", 0))
        end_min = int(dw.get("end_min", 24*60))
        # normalize to same-day range, ignore crosses_midnight for first pass
        if start_min < 0 or start_min >= 24*60:
            continue
        end_min = min(end_min, 24*60)

        out.append(Vehicle(
            vehicle_id=str(duty_id),
            start_index=int(start_idx),
            end_index=int(end_idx),
            start_min=start_min,
            end_min=end_min,
            max_overtime_min=int(max_overtime_per_duty_min)
        ))
    return out

def build_shipment_for_request(
    start_idx: int,
    end_idx: int,
    mode: str,                 # "depart_after" | "arrive_before"
    when_min: int,             # minutes since 00:00 local
    sla_by_priority: Dict[int, Dict[str, int]],
    priority: int,
) -> Shipment:
    """
    Derive simple pickup/delivery windows by SLA.
    Example sla_by_priority[2] = {"depart_after_slack_min": 90, "arrive_before_slack_min": 90}
    """
    p = int(priority)
    cfg = sla_by_priority.get(p, sla_by_priority.get(3, {"depart_after_slack_min": 90, "arrive_before_slack_min": 90}))
    if mode == "depart_after":
        # allow pickup any time between when_min .. when_min + slack
        pickup_tw = (when_min, min(when_min + int(cfg["depart_after_slack_min"]), 24*60))
        # delivery can be later; leave wide for now
        delivery_tw = (0, 24*60)
    else:
        # arrive_before: allow delivery up to when_min, with slack window before it
        delivery_tw = (max(0, when_min - int(cfg["arrive_before_slack_min"])), when_min)
        pickup_tw = (0, 24*60)

    return Shipment(
        pickup_index=start_idx,
        delivery_index=end_idx,
        pickup_tw=pickup_tw,
        delivery_tw=delivery_tw,
        service_time_pickup=0,
        service_time_delivery=0,
        priority=p
    )

def make_cuopt_payload(
    time_matrix: List[List[float]],
    distance_matrix: Optional[List[List[float]]],
    vehicles: List[Vehicle],
    shipment: Shipment,
    cost_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Construct a minimal routing payload. Different cuOpt builds expect slightly different field names.
    This structure is intentionally simple; adjust keys to match your server if it returns a 400 with a helpful message.
    """
    # vehicles
    vlist = []
    for v in vehicles:
        vlist.append({
            "id": v.vehicle_id,
            "start_index": v.start_index,
            "end_index": v.end_index,
            "time_window": [int(v.start_min), int(v.end_min)],
            "max_overtime": int(v.max_overtime_min),
        })

    # one shipment (pickup->delivery)
    tasks = [{
        "type": "shipment",
        "pickup": {
            "location_index": int(shipment.pickup_index),
            "time_window": [int(shipment.pickup_tw[0]), int(shipment.pickup_tw[1])],
            "service_time": int(shipment.service_time_pickup),
        },
        "delivery": {
            "location_index": int(shipment.delivery_index),
            "time_window": [int(shipment.delivery_tw[0]), int(shipment.delivery_tw[1])],
            "service_time": int(shipment.service_time_delivery),
        },
        "priority": int(shipment.priority),
    }]

    payload: Dict[str, Any] = {
        "model": "routing",
        "time_matrix": time_matrix,     # minutes
        # "distance_matrix": distance_matrix,  # optional
        "vehicles": vlist,
        "tasks": tasks,
        "objective": {
            "type": "minimize_time",    # keep simple; tune later
            "weights": (cost_weights or {}),
        },
        "settings": {
            "num_workers": 16
        }
    }
    return payload

def parse_cuopt_solution_to_assignments(
    result: Dict[str, Any],
    new_trip_id: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Map a typical cuOpt 'response' into our assignments list. Adjust the keys to match your server’s actual response.
    For now, fall back to zero if structure not present (caller will keep greedy result).
    """
    resp = result.get("response") if isinstance(result, dict) else None
    if not isinstance(resp, dict):
        return 0.0, []

    objective = float(resp.get("objective", 0.0))
    assignments: List[Dict[str, Any]] = []

    # Example structure (adjust if different):
    # resp["routes"] = [{"vehicle_id":"B415", "stops":[{"task_id": "...", ...}, ...]}]
    for r in resp.get("routes", []):
        vid = r.get("vehicle_id")
        # if this route includes our shipment, treat as reassigned
        assignments.append({
            "trip_id": new_trip_id,
            "type": "reassigned",
            "driver_id": vid,
            "candidate_id": f"{vid}::cuopt",
            "delay_minutes": 0.0,
            "uses_emergency_rest": False,
            "deadhead_miles": 0.0,
            "overtime_minutes": 0.0,
            "miles_delta": 0.0,
            "cost": 0.0,  # you can back-calculate with your cost model if you like
        })

    return objective, assignments
