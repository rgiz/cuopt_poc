from src.plan.router import create_router

def test_leg_helpers_smoke():
    # create_router needs callables; pass dummies
    def get_data(): return {"distance": None, "time": None, "location_to_index": {}}
    def get_cost(): return {}
    def get_url(): return "http://dummy"

    r = create_router(get_data, get_cost, get_url)
    # access closure functions for coverage via the router module if exposed,
    # or just ensure router builds without error.
    assert r is not None