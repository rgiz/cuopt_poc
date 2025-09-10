import responses
from src.plan.cuopt_adapter import solve_with_cuopt

@responses.activate
def test_cuopt_adapter_solve_ok():
    base = "http://localhost:5000"
    
    # Mock the NEW cuOpt 25.10.0a async endpoints
    responses.add(
        responses.POST, 
        base + "/cuopt/request",  # Updated endpoint
        json={"reqId": "test-123"}, 
        status=200
    )
    
    # Mock the polling response
    responses.add(
        responses.GET,
        base + "/cuopt/requests/test-123",
        json={
            "response": {
                "solver_response": {
                    "status": 0,
                    "solution_cost": 123.4,
                    "vehicle_data": {}
                }
            }
        },
        status=200
    )
    
    out = solve_with_cuopt(base, {"vehicles": [], "tasks": []})
    assert "solver_response" in out

