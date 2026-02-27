def test_solve_cascades_includes_diagnostics_contract(client):
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
    details = body.get("details", {})
    assert "cascade_diagnostics" in details

    diag = details["cascade_diagnostics"]
    for key in [
        "candidates_total",
        "feasible_hard_count",
        "max_chain_depth",
        "avg_chain_depth",
        "unresolved_total",
        "uncovered_p4_total",
        "disposed_p5_total",
        "reason_code_counts",
    ]:
        assert key in diag

    cascades = body.get("cascades", [])
    for c in cascades:
        assert "reason_code" in c
        assert "reason_detail" in c
        assert "assigned_steps" in c
        assert "blocked_steps" in c



def test_solve_multi_includes_diagnostics_contract(client):
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
    assert "meta" in body
    assert "cascade_diagnostics" in body["meta"]

    for solution in body.get("solutions", []):
        details = solution.get("details", {})
        if details.get("backend") == "cascade-cuopt-enhanced":
            assert "cascade_diagnostics" in details
        for c in solution.get("cascades", []):
            assert "reason_code" in c
            assert "reason_detail" in c
