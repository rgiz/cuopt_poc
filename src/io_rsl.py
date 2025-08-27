import pandas as pd
from typing import Dict, Any, Tuple

def load_and_normalize(rsl_path: str, locations_path: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # TODO: read your RSL and locations; return (locations, duties, drivers)
    # locations: location_id, name, lat, lon, postcode
    # duties: duty_id, driver_id, sequence, location_id, service_start_plan, service_duration, priority, type
    # drivers: driver_id, shift_start, shift_end, breaks?
    raise NotImplementedError