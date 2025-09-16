import time
import requests
from typing import Dict, Any, Optional, List

class CuOptSolution:
    def __init__(self, objective_value: float, assignments: List[Dict[str, Any]], details: Dict[str, Any]):
        self.objective_value = objective_value
        self.assignments = assignments
        self.details = details

class CuOptModel:
    def __init__(
        self,
        driver_states: Dict[str, Any],
        distance_miles_matrix,
        time_minutes_matrix,
        location_to_index: Dict[str, int],
        cost_config: Dict[str, float],
        server_url: str,
        max_solve_time_seconds: int = 30,
        **kwargs
    ):
        # Handle case where driver_states is None or empty
        if driver_states is None:
            driver_states = {}
        
        # Ensure driver_states has the expected structure
        if "drivers" not in driver_states:
            driver_states = {"drivers": driver_states}
            
        self.driver_states = driver_states
        self.distance = distance_miles_matrix
        self.time = time_minutes_matrix
        self.loc2idx = {str(k).upper(): int(v) for k, v in location_to_index.items()}
        self.cost = cost_config or {}
        self.server_url = str(server_url or "").rstrip("/")
        self.max_time = int(max_solve_time_seconds)
        
        # Cost coefficients for objective function
        self.deadhead_cost_per_mile = self.cost.get("deadhead_cost_per_mile", 1.0)
        self.overtime_cost_per_minute = self.cost.get("overtime_cost_per_minute", 1.0)
        self.admin_cost = self.cost.get("reassignment_admin_cost", 10.0)

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information about the model's data"""
        drivers = self.driver_states.get("drivers", {})
        
        return {
            "driver_states_structure": {
                "has_drivers_key": "drivers" in self.driver_states,
                "num_drivers": len(drivers),
                "driver_ids": list(drivers.keys())[:5],
                "sample_driver": list(drivers.values())[0] if drivers else None
            },
            "matrices": {
                "distance_shape": self.distance.shape if hasattr(self.distance, 'shape') else "unknown",
                "time_shape": self.time.shape if hasattr(self.time, 'shape') else "unknown",
                "num_locations": len(self.loc2idx)
            },
            "cost_config": self.cost,
            "server_url": self.server_url
        }

    def basic_connectivity_test(self) -> Dict[str, Any]:
        """Test basic cuOpt connectivity without requiring driver data"""
        try:
            test_payload = {
                "cost_matrix_data": {
                    "data": {"0": [[0, 10], [10, 0]]}
                },
                "fleet_data": {
                    "vehicle_locations": [[0, 0]],
                    "vehicle_time_windows": [[0, 1440]],
                    "capacities": [[100]],
                    "vehicle_types": [1]
                },
                "task_data": {
                    "task_locations": [1],
                    "task_time_windows": [[0, 1440]],
                    "service_times": [5],
                    "demand": [[1]]
                },
                "solver_config": {
                    "time_limit": 5
                }
            }
            
            headers = {'Content-Type': 'application/json', 'CLIENT-VERSION': 'custom'}
            submit_url = f"{self.server_url}/cuopt/request"
            
            response = requests.post(submit_url, json=test_payload, headers=headers, timeout=10)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_preview": str(response.text)[:200] + "..." if len(response.text) > 200 else response.text,
                "url_tested": submit_url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url_tested": f"{self.server_url}/cuopt/request"
            }