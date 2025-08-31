# src/plan/solve.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

# TODO: import your actual solver module(s)
# Example: from src.opt.cuopt_model_miles import solve_vrp, insert_with_pin
# Adjust these to your real entry points:
def _run_cuopt_baseline(problem: Dict[str,Any], time_limit_s: int) -> Dict[str,Any]:
    """
    Call your current baseline solve here and return a dict:
    {
      "routes": { "driver_id": [ { "order_id":..., "location_id":..., "eta_start": int, "eta_end": int }, ... ], ... },
      "metrics": { "total_dist": ..., "total_time": ... },
      "problem": problem  # include for later reference
    }
    """
    # --- TODO: replace this with your working function ---
    # return solve_vrp(problem, time_limit_s=time_limit_s)
    raise NotImplementedError("Wire _run_cuopt_baseline() to your existing solver function")

def _run_cuopt_with_pin(problem: Dict[str,Any], baseline: Dict[str,Any], job: Dict[str,Any], driver_id: str, time_limit_s: int) -> Dict[str,Any]:
    """
    Call your current 'insert with vehicle pinned' function and return a result dict.
    Should include delta metrics vs baseline if you can; else we compute later.
    """
    # --- TODO: replace this with your working function ---
    # return insert_with_pin(problem, baseline, job, driver_id, time_limit_s=time_limit_s)
    raise NotImplementedError("Wire _run_cuopt_with_pin() to your existing solver function")

def solve_baseline(problem: Dict[str,Any], cfg: Dict[str,Any]) -> Dict[str,Any]:
    tlim = int(os.getenv("CUOPT_SOLVER_TIMEOUT_SEC", str(cfg.get("time_limit_s", 120))))
    sol = _run_cuopt_baseline(problem, time_limit_s=tlim)
    if "problem" not in sol:
        sol["problem"] = problem
    return sol

def solve_with_insertion(problem: Dict[str,Any], baseline: Dict[str,Any], job: Dict[str,Any], shortlist: List[str], cfg: Dict[str,Any]) -> Dict[str,Any]:
    tlim_each = int(cfg.get("pin_time_limit_s", 30))
    results = []
    for d in shortlist:
        try:
            r = _run_cuopt_with_pin(problem, baseline, job, d, time_limit_s=tlim_each)
            r["driver_id"] = d
            results.append(r)
        except Exception as e:
            results.append({"driver_id": d, "error": str(e)})

    # Choose best (simple heuristic if your solver didn’t compute deltas)
    def key_fn(x):
        return (x.get("delta_km", 1e9), x.get("delta_min", 1e9))
    best = min([r for r in results if "error" not in r], key=key_fn) if any("error" not in r for r in results) else None

    return {"best": best, "alternatives": results}

