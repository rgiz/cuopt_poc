import json

def test_priority_map_loaded(client):
    r = client.get("/plan/priority_map")
    assert r.status_code == 200
    data = r.json()
    assert "DELIVERY" in data
    assert data["EMPTY"] == 5

def test_locations(client):
    r = client.get("/plan/locations")
    assert r.status_code == 200
    body = r.json()
    # Your router returns {'locations': [...], 'count': N, 'source': ...} in the first definition
    assert "count" in body

def test_candidates_smoke(client, monkeypatch):
    payload = {
        "start_location": "A",
        "end_location": "B",
        "mode": "depart_after",
        "when_local": "2025-09-02T10:30",
        "priority": 2,
        "top_n": 5
    }
    r = client.post("/plan/candidates", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "weekday" in body
    assert "candidates" in body

def test_solve_multi_heuristic(client):
    payload = {
        "start_location":"A",
        "end_location":"B",
        "mode":"depart_after",
        "when_local":"2025-09-02T10:30",
        "priority":1,
        "top_n_per_step":2,
        "max_cascades":1,
        "max_drivers_affected":2,
        "max_solutions":2,
        "use_cuopt": False
    }
    r = client.post("/plan/solve_multi", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "solutions" in body
    # schedules + cascades presence
    for sol in body["solutions"]:
        assert "assignments" in sol
        assert "schedules" in sol
