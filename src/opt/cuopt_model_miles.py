import time
import requests
from typing import Dict, Any, Optional

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
        self.driver_states = driver_states
        self.distance = distance_miles_matrix
        self.time = time_minutes_matrix
        self.loc2idx = {str(k).upper(): int(v) for k, v in location_to_index.items()}
        self.cost = cost_config
        self.server_url = str(server_url or "").rstrip("/")
        self.max_time = int(max_solve_time_seconds)

    def _request_and_poll(self, problem_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Submit asynchronous request to cuOpt 25.10.0a and poll for results.
        """
        # Step 1: Submit request to new async endpoint
        headers = {
            'Content-Type': 'application/json',
            'CLIENT-VERSION': 'custom'  # Skip version compatibility check
        }
        
        submit_url = f"{self.server_url}/cuopt/request"
        
        try:
            response = requests.post(
                submit_url,
                json=problem_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 404:
                raise RuntimeError(f"Endpoint not found. Ensure cuOpt container is running and accessible at {submit_url}")
            
            response.raise_for_status()
            initial_response = response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to submit request to cuOpt: {e}")
        
        # Check if response contains immediate result (unlikely in 25.x)
        if 'response' in initial_response and 'solver_response' in initial_response['response']:
            return initial_response['response']
        
        # Step 2: Extract request ID for polling
        if 'reqId' not in initial_response:
            raise RuntimeError(f"No request ID in response: {initial_response}")
            
        request_id = initial_response['reqId']
        
        # Step 3: Poll for results
        poll_url = f"{self.server_url}/cuopt/requests/{request_id}"
        start_time = time.time()
        attempts = 0
        max_retries = timeout
        
        while attempts < max_retries and (time.time() - start_time) < timeout:
            try:
                poll_response = requests.get(
                    poll_url,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if poll_response.status_code == 404:
                    # Request ID not ready yet, continue polling
                    time.sleep(1)
                    attempts += 1
                    continue
                    
                poll_response.raise_for_status()
                result = poll_response.json()
                
                # Check if result contains the solution
                if 'response' in result:
                    solver_response = result['response'].get('solver_response', {})
                    
                    # Check solver status
                    if solver_response.get('status') == 0:  # 0 = feasible solution
                        return result['response']
                    elif solver_response.get('status') == 1:  # 1 = infeasible
                        raise RuntimeError(f"cuOpt found no feasible solution: {solver_response}")
                    else:
                        # Still processing
                        time.sleep(1)
                        attempts += 1
                        continue
                        
                # Check for explicit status field
                status = result.get('status', '').lower()
                if status == 'failed':
                    error_msg = result.get('error', 'Unknown error')
                    raise RuntimeError(f"cuOpt job failed: {error_msg}")
                    
                # Continue polling
                time.sleep(1)
                attempts += 1
                
            except requests.exceptions.RequestException as e:
                # Network error - retry
                time.sleep(1)
                attempts += 1
                continue
        
        # Timeout reached
        raise TimeoutError(f"Request {request_id} timed out after {timeout}s")
    
    def solve(self, disrupted_trips, candidates_per_trip, params=None) -> CuOptSolution:
        """Solve using NVIDIA cuOpt API"""
        
        if not disrupted_trips:
            return CuOptSolution(0.0, [], {"backend": "empty"})

        try:
            # Build proper cuOpt payload
            payload = self._build_cuopt_payload(disrupted_trips, candidates_per_trip)
            
            # Submit request and poll for result
            result = self._request_and_poll(payload, timeout=self.max_time)
            
            # Parse result
            objective = result.get('solver_response', {}).get('solution_cost', 0.0)
            routes = result.get('solver_response', {}).get('vehicle_data', {})
            
            # Convert routes to assignments
            assignments = []
            for vehicle_id, route_data in routes.items():
                route = route_data.get('route', [])
                for i, location in enumerate(route[1:-1], 1):  # Skip depot start/end
                    assignments.append({
                        "trip_id": f"task_{location}",
                        "driver_id": vehicle_id,
                        "cost": objective / len(routes) if routes else 0.0
                    })
            
            return CuOptSolution(
                objective_value=float(objective),
                assignments=assignments,
                details={"backend": "cuopt", "request_successful": True}
            )
            
        except Exception as e:
            # Fallback to greedy
            return self._greedy_fallback(disrupted_trips)

    def _build_cuopt_payload(self, disrupted_trips, candidates_per_trip) -> Dict[str, Any]:
        """Build proper NVIDIA cuOpt payload format"""
        
        # Build cost matrix from your distance matrix
        n_locations = self.distance.shape[0]
        cost_matrix = self.distance.tolist()
        
        # Build vehicles from driver states
        vehicles = []
        ds = self.driver_states.get("drivers", {})
        for i, (driver_id, meta) in enumerate(ds.items()):
            home_loc = meta.get("home_center_id", 0)
            if home_loc is None or home_loc < 0:
                home_loc = 0
            
            vehicles.append({
                "start_location": home_loc,
                "end_location": home_loc,
                "time_window": [
                    meta.get("start_min", 0),
                    meta.get("end_min", 1440)
                ]
            })

        # Build tasks from disrupted trips
        tasks = []
        for trip in disrupted_trips:
            start_loc = self.loc2idx.get(trip["start_location"].upper(), 0)
            end_loc = self.loc2idx.get(trip["end_location"].upper(), 0)
            
            tasks.append({
                "location": start_loc,  # Pickup location
                "time_window": [0, 1440],  # Wide time window
                "service_time": trip.get("duration_minutes", 0),
                "demand": 1  # Default demand
            })

        # Build the complete payload in NVIDIA cuOpt format
        return {
            "cost_matrix_data": {
                "data": {
                    "0": cost_matrix
                }
            },
            "fleet_data": {
                "vehicle_locations": [[v["start_location"], v["end_location"]] for v in vehicles],
                "vehicle_time_windows": [v["time_window"] for v in vehicles]
            },
            "task_data": {
                "task_locations": [t["location"] for t in tasks],
                "task_time_windows": [t["time_window"] for t in tasks],
                "service_times": [t["service_time"] for t in tasks],
                "demand": [[t["demand"]] for t in tasks]
            },
            "solver_config": {
                "time_limit": self.max_time
            }
        }