from src.plan.candidates import weekday_from_local

def test_weekday_from_local_label():
    assert weekday_from_local("2025-09-02T10:30") == "Tue"