from typing import Dict, Any, List

def solve_baseline(problem: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: call cuOpt once, return solution dict (routes per driver with times)
    raise NotImplementedError

def solve_with_insertion(problem: Dict[str, Any], baseline: Dict[str, Any], job: Dict[str, Any],
                         shortlist: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: warm start, free/global assign + per-candidate assign runs, return ranked options
    raise NotImplementedError