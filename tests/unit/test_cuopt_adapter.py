import responses
from src.plan.cuopt_adapter import solve_with_cuopt

@responses.activate
def test_cuopt_adapter_solve_ok():
    base = "http://dummy-cuopt:5000"
    responses.add(responses.POST, base + "/solve",
                  json={"solutions":[{"objective": 123.4}]}, status=200)
    out = solve_with_cuopt(base, {"vehicles": [], "tasks": []})
    assert "solutions" in out

