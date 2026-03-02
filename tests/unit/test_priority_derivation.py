from src.priority_derivation import derive_priority, normalize_priority_map


def test_derive_priority_prefers_explicit_numeric_priority():
    priority_map = {"DELIVERY": 1, "DEFAULT": 4}
    assert derive_priority("2", "DELIVERY", priority_map, default=3) == 2



def test_derive_priority_falls_back_to_load_type_and_default_key():
    priority_map = {"DELIVERY": 1, "DEFAULT": 4}
    assert derive_priority(None, "DELIVERY", priority_map, default=3) == 1
    assert derive_priority(None, "UNKNOWN", priority_map, default=3) == 4



def test_derive_priority_clamps_out_of_range_values():
    priority_map = {"DELIVERY": 9, "DEFAULT": 0}
    assert derive_priority(None, "DELIVERY", priority_map, default=3) == 5
    assert derive_priority(None, "UNKNOWN", priority_map, default=3) == 1



def test_normalize_priority_map_uppercases_keys():
    mp = normalize_priority_map({"delivery": 2, "Default": 3})
    assert mp["DELIVERY"] == 2
    assert mp["DEFAULT"] == 3
