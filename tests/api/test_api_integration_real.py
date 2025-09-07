import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_DATASET", "toy").lower() != "real",
    reason="Only runs with TEST_DATASET=real"
)

def test_candidates_real_names(client):
    payload = {
        "start_location": "BIRMINGHAM MAIL CENTRE",
        "end_location":   "MIDLANDS SUPER HUB",
        "mode": "depart_after",
        "when_local": "2025-09-02T10:30",
        "priority": 2,
        "top_n": 5
    }
    r = client.post("/plan/candidates", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "candidates" in body
