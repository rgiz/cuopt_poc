from src.plan.config import load_priority_map

def test_load_priority_map_prefers_data_root():
    m = load_priority_map()
    assert m["DELIVERY"] == 1
    assert m["EMPTY"] == 5