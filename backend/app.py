# backend/app.py (only the relevant parts)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from src import io_rsl, matrix
from src.plan import build as plan_build
from src.plan import solve as plan_solve
from src.plan import candidates as plan_cand

app = FastAPI(title="cuOpt Rescheduler")

STATE: Dict[str,Any] = {"baseline": None, "problem": None, "matrices": None, "config": None, "valid_location_ids": set()}

class BaselineRequest(BaseModel):
    rsl_path: str
    locations_path: Optional[str] = None
    config: Dict[str, Any] = {}

class InsertJob(BaseModel):
    pickup_location_id: str
    drop_location_id: Optional[str] = None
    time_window: List[int]         # [start_min, end_min]
    service_time: int              # minutes
    priority: int = 3

@app.post("/solve/baseline")
def solve_baseline(req: BaselineRequest):
    locs, duties, drivers = io_rsl.load_and_normalize(req.rsl_path, req.locations_path, req.config)
    mats = matrix.load_or_build(locs, req.config)
    prob = plan_build.from_rsl(locs, duties, drivers, mats, req.config)
    baseline = plan_solve.solve_baseline(prob, req.config)

    STATE.update({
        "baseline": baseline,
        "problem": prob,
        "matrices": mats,
        "config": req.config,
        "valid_location_ids": set(locs["location_id"]),
    })
    return {"status": "ok", "drivers": int(drivers.shape[0]), "tasks": int(duties.shape[0])}

@app.post("/solve/insert")
def insert_job(job: InsertJob):
    if not STATE["baseline"]:
        raise HTTPException(status_code=400, detail="Run /solve/baseline first.")
    shortlist = plan_cand.shortlist(
        STATE["baseline"],
        job.model_dump(),
        STATE["valid_location_ids"],
        STATE["matrices"],
        STATE["config"],
    )
    result = plan_solve.solve_with_insertion(STATE["problem"], STATE["baseline"], job.model_dump(), shortlist, STATE["config"])
    return result


# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional, List, Dict, Any

# # Import your domain logic
# from src import io_rsl, matrix, model, solve, analysis

# app = FastAPI(title="cuOpt Rescheduler")

# # In-memory cache (replace with Redis later if needed)
# STATE = {
#     "baseline": None,   # solution dict
#     "problem": None,    # built problem dict
#     "matrices": None,   # distance/time matrices
#     "config": None
# }

# class BaselineRequest(BaseModel):
#     rsl_path: str
#     locations_path: str
#     config: Dict[str, Any] = {}

# class InsertJob(BaseModel):
#     pickup_location_id: str
#     drop_location_id: Optional[str] = None
#     time_window: List[int]         # [start_min, end_min]
#     service_time: int              # minutes
#     priority: int = 3

# @app.post("/solve/baseline")
# def solve_baseline(req: BaselineRequest):
#     locs, duties, drivers = io_rsl.load_and_normalize(
#         req.rsl_path, req.locations_path, req.config
#     )
#     mats = matrix.load_or_build(locs, req.config)
#     prob = model.build_problem(locs, duties, drivers, mats, req.config)
#     baseline = solve.solve_baseline(prob, req.config)
#     STATE.update({"baseline": baseline, "problem": prob, "matrices": mats, "config": req.config})
#     return {"status": "ok", "drivers": len(drivers), "tasks": len(duties)}

# @app.post("/solve/insert")
# def insert_job(job: InsertJob):
#     assert STATE["baseline"] and STATE["problem"], "Run /solve/baseline first"
#     shortlist = analysis.preselect_candidates(STATE["baseline"], job, STATE["matrices"], STATE["config"])
#     result = solve.solve_with_insertion(
#         STATE["problem"], STATE["baseline"], job, shortlist, STATE["config"]
#     )
#     return result