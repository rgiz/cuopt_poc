from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

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
    cost_breakdown: Dict[str, float] = {}

class PlanSolveCascadeResponse(BaseModel):
    weekday: str
    trip_minutes: float
    trip_miles: float
    objective_value: float
    assignments: List[AssignmentOut]
    details: Dict[str, Any]
    candidates_considered: int
    cascades: List[Dict[str, Any]]
