def test_solve_cascades_includes_performance_metrics(client):
    payload = {
        "start_location": "A",
        "end_location": "B",
        "when_local": "2025-09-02T10:30",
        "priority": 2,
        "mode": "depart_after",
        "max_cascades": 2,
        "max_drivers_affected": 3,
    }

    response = client.post("/plan/solve_cascades", json=payload)
    assert response.status_code == 200
    body = response.json()

    perf = body.get("details", {}).get("performance", {})
    for key in [
        "total_ms",
        "ensure_ready_ms",
        "candidate_generation_ms",
        "assignment_build_ms",
        "cascade_build_ms",
        "postprocess_ms",
    ]:
        assert key in perf
        assert perf[key] >= 0



def test_solve_multi_includes_performance_metrics(client):
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

    perf = body.get("meta", {}).get("performance", {})
    for key in [
        "total_ms",
        "ensure_ready_ms",
        "candidate_generation_ms",
        "solution_build_ms",
    ]:
        assert key in perf
        assert perf[key] >= 0
