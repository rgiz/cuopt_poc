from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Optional convenience constant (used by UI or logging if you want)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -----------------------------
# Core request/response models
# -----------------------------

class PlanRequest(BaseModel):
    start_location: str = Field(
        ...,
        description="Canonical site name (matches location_index.csv: name)",
    )
    end_location: str = Field(
        ...,
        description="Canonical site name",
    )
    mode: str = Field(
        ...,
        pattern="^(depart_after|arrive_before)$",
        description="depart_after or arrive_before",
    )
    when_local: str = Field(
        ...,
        description="Local datetime in Europe/London, e.g. 2025-08-18T21:30",
    )
    priority: int = Field(
        3,
        ge=1,
        le=5,
        description="1=highest urgency, 5=lowest",
    )
    trip_minutes: Optional[float] = Field(
        None,
        description="Override travel minutes; else use matrix",
    )
    trip_miles: Optional[float] = Field(
        None,
        description="Override miles; else use matrix",
    )
    top_n: int = Field(
        20,
        ge=1,
        le=200,
        description="Limit candidates returned",
    )


class PlanSolveCascadeRequest(PlanRequest):
    max_cascades: int = Field(2, ge=0, le=5)
    max_drivers_affected: int = Field(5, ge=1, le=50)


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
    reason: Optional[str] = None


class PlanCandidatesResponse(BaseModel):
    weekday: str                 # keep as str for consistency with backend helpers
    trip_minutes: float
    trip_miles: float
    candidates: List[CandidateOut]


class AssignmentOut(BaseModel):
    trip_id: str
    type: str  # "reassigned" | "outsourced" etc.
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
    weekday: str                # int (0=Mon .. 6=Sun), matches weekday_from_local
    trip_minutes: float
    trip_miles: float
    objective_value: float
    assignments: List[AssignmentOut]
    details: Dict[str, Any]
    candidates_considered: int
    cascades: List[Dict[str, Any]]
    schedules: List[DriverScheduleOut] = [] 


# -----------------------------
# Multi-solution (list of options)
# -----------------------------

class DriverScheduleOut(BaseModel):
    driver_id: str
    before: List[Dict[str, Any]]
    after: List[Dict[str, Any]]


class PlanSolutionOut(BaseModel):
    rank: int
    objective_value: float
    drivers_touched: int
    assignments: List["AssignmentOut"]   # forward ref to AssignmentOut above
    cascades: List[Dict[str, Any]] = []
    schedules: List[DriverScheduleOut] = []
    details: Dict[str, Any] = {}


class PlanSolveMultiRequest(BaseModel):
    start_location: str
    end_location: str
    when_local: str
    mode: str = "depart_after"           # or "arrive_before"
    priority: int = 3
    top_n_per_step: int = 3              # branch factor
    max_cascades: int = 2
    max_drivers_affected: int = 3
    max_solutions: int = 5
    trip_minutes: Optional[float] = None
    trip_miles: Optional[float] = None
    use_cuopt: bool = False              # flip to True to call cuOpt
    seed: Optional[int] = None


class PlanSolveMultiResponse(BaseModel):
    weekday: str
    trip_minutes: float
    trip_miles: float
    solutions: List[PlanSolutionOut]
    meta: Dict[str, Any] = {}
