import pandas as pd

from src.driver_states_builder import build_driver_states


def test_build_driver_states_accepts_due_to_convey_variant_and_sets_planz_code():
    df = pd.DataFrame(
        [
            {
                "Duty ID": "D1",
                "Element Type": "Travel",
                "Commencement Time": "10:30:00",
                "Ending Time": "11:15:00",
                "Mapped Name A": "BIRMINGHAM MC",
                "Mapped Name B": "MIDLANDS SUPER HUB",
                "Due to Convey": "EMPTY",
                "PLANZ Code": "X123",
                "Mon": "Y",
                "Tue": "Y",
                "Wed": "Y",
                "Thu": "Y",
                "Fri": "Y",
                "Sat": "N",
                "Sun": "N",
            }
        ]
    )

    states, _ = build_driver_states(df)
    elements = states["D1"]["elements"]
    assert len(elements) == 1
    assert elements[0]["load_type"] == "EMPTY"
    assert elements[0]["planz_code"] == "X123"


def test_build_driver_states_carries_post_midnight_rows_to_next_day_for_cross_midnight_duty():
    df = pd.DataFrame(
        [
            {
                "Duty ID": "D2",
                "Element Type": "Start Facility",
                "Commencement Time": "21:00:00",
                "Ending Time": "21:25:00",
                "Mapped Name A": "BIRMINGHAM MC",
                "Mapped Name B": "BIRMINGHAM MC",
                "Due to Convey": "",
                "Mon": "Y",
                "Tue": "N",
                "Wed": "N",
                "Thu": "N",
                "Fri": "N",
                "Sat": "N",
                "Sun": "N",
            },
            {
                "Duty ID": "D2",
                "Element Type": "Travel",
                "Commencement Time": "00:20:00",
                "Ending Time": "01:05:00",
                "Mapped Name A": "BIRMINGHAM MC",
                "Mapped Name B": "MIDLANDS SUPER HUB",
                "Due to Convey": "LOADED",
                "Mon": "Y",
                "Tue": "N",
                "Wed": "N",
                "Thu": "N",
                "Fri": "N",
                "Sat": "N",
                "Sun": "N",
            },
            {
                "Duty ID": "D2",
                "Element Type": "End Facility",
                "Commencement Time": "04:45:00",
                "Ending Time": "05:00:00",
                "Mapped Name A": "BIRMINGHAM MC",
                "Mapped Name B": "BIRMINGHAM MC",
                "Due to Convey": "",
                "Mon": "Y",
                "Tue": "N",
                "Wed": "N",
                "Thu": "N",
                "Fri": "N",
                "Sat": "N",
                "Sun": "N",
            },
        ]
    )

    states, _ = build_driver_states(df)
    travel = [e for e in states["D2"]["elements"] if e["element_type"] == "TRAVEL"][0]

    assert states["D2"]["daily_windows"]["Mon"]["crosses_midnight"] is True
    assert travel["Mon"] == 1
    assert travel["Tue"] == 1
    assert travel.get("overnight_day_carry") is True
