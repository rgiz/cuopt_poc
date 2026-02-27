import pytest


def test_solve_multi_returns_ranked_solutions(client):
    payload = {
        "start_location": "A",
        "end_location": "B",
        "mode": "depart_after",
        "when_local": "2025-09-02T10:30",
        "priority": 2,
        "top_n_per_step": 3,
        "max_cascades": 2,
        "max_drivers_affected": 3,
        "max_solutions": 5,
        "use_cuopt": False,
    }

    response = client.post("/plan/solve_multi", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "solutions" in body
    assert len(body["solutions"]) >= 1

    costs = [solution["objective_value"] for solution in body["solutions"]]
    assert costs == sorted(costs)


def test_solve_cascades_outsourced_fallback_when_no_candidates(client):
    payload = {
        "start_location": "A",
        "end_location": "C",
        "when_local": "2025-09-02T02:00",
        "priority": 5,
        "mode": "depart_after",
        "max_cascades": 1,
        "max_drivers_affected": 1,
    }

    response = client.post("/plan/solve_cascades", json=payload)
    assert response.status_code == 200

    body = response.json()
    assignments = body.get("assignments", [])
    assert len(assignments) >= 1

    first = assignments[0]
    assert first["type"] in {"outsourced", "reassigned"}

    if first["type"] == "outsourced":
        breakdown = first.get("cost_breakdown", {})
        assert "outsourcing_base" in breakdown
        assert "outsourcing_miles" in breakdown
        assert body.get("details", {}).get("fallback") == "outsourced"
