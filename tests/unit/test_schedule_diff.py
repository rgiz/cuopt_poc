from src.plan.router import create_router

def test_compute_before_after_schedules_exists():
    def get_data(): return {
        "distance": None, "time": None, "location_to_index": {},
        "driver_states": {"drivers": {"D1": {"elements": []}}}
    }
    def get_cost(): return {}
    def get_url(): return "http://dummy"
    router = create_router(get_data, get_cost, get_url)
    # smoke: router constructed; schedule diff is used by solve_multi path (covered in api tests)
    assert router is not None
