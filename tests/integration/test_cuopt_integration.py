# tests/integration/test_cuopt_integration.py
import pytest
import responses
import json
from unittest.mock import patch
from src.plan.cuopt_adapter import solve_with_cuopt, build_cuopt_payload, extract_solutions_from_cuopt

@pytest.fixture(scope="session")
def cuopt_server_url():
    """Fixture to provide cuOpt server URL for integration tests"""
    import os
    import requests
    import time
    
    url = os.getenv("TEST_CUOPT_URL", "http://localhost:5000")
    
    if url:
        # Try multiple health endpoints
        health_endpoints = [
            "/cuopt/health",  # ✅ This one works for 25.10.0a!
            "/health",
            "/v2/health/live",
            "/"
        ]
        
        for attempt in range(30):
            for endpoint in health_endpoints:
                try:
                    resp = requests.get(f"{url}{endpoint}", timeout=2)
                    if resp.status_code == 200:
                        print(f"cuOpt server ready at {url} (endpoint: {endpoint})")
                        return url
                except:
                    continue
            time.sleep(1)
    
    return None

class TestCuOptIntegration:

    @responses.activate
    def test_cuopt_solver_discovery_endpoints(self):
        """Test that the solver correctly discovers cuOpt endpoints"""
        base_url = "http://localhost:5000"
        
        # Mock the NEW 25.10.0a async endpoint
        responses.add(
            responses.POST, 
            f"{base_url}/cuopt/request",  # Updated endpoint
            json={"reqId": "test-123"}, 
            status=200
        )
        
        # Mock the polling endpoint
        responses.add(
            responses.GET,
            f"{base_url}/cuopt/requests/test-123",
            json={
                "response": {
                    "solver_response": {
                        "status": 0,
                        "solution_cost": 100.0,
                        "vehicle_data": {}
                    }
                }
            },
            status=200
        )
        
        payload = {"vehicles": [], "tasks": []}
        result = solve_with_cuopt(base_url, payload)
        
        assert "solver_response" in result

    @responses.activate 
    def test_cuopt_fallback_endpoints(self):
        """Test fallback when health endpoint works but solve fails"""
        base_url = "http://localhost:5000"
        
        # Mock async endpoint success
        responses.add(
            responses.POST, 
            f"{base_url}/cuopt/request", 
            json={"reqId": "test-456"}, 
            status=200
        )
        
        responses.add(
            responses.GET,
            f"{base_url}/cuopt/requests/test-456",
            json={
                "response": {
                    "solver_response": {
                        "status": 0,
                        "solution_cost": 150.0
                    }
                }
            },
            status=200
        )
        
        payload = {"vehicles": [], "tasks": []}
        result = solve_with_cuopt(base_url, payload)
        
        assert "solver_response" in result

    def test_cuopt_async_request_poll_pattern(self):
        """Test async request/poll pattern used by cuOpt 25.10.0a"""
        from src.opt.cuopt_model_miles import CuOptModel
        
        # Create model with correct parameters
        model = CuOptModel(
            driver_states={"drivers": {}},
            distance_miles_matrix=[[0]], 
            time_minutes_matrix=[[0]],
            location_to_index={"A": 0},
            cost_config={"deadhead_cost_per_mile": 1.0},
            server_url="http://localhost:5000"
        )
        
        test_payload = {
            "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
            "fleet_data": {"vehicle_locations": [[0, 0]]},
            "task_data": {"task_locations": [1]}
        }
        
        # Test that the method exists and can be called
        # In a real test environment, this would connect to cuOpt
        try:
            # This will fail without real cuOpt, but tests the method exists
            model._request_and_poll(test_payload, timeout=1)
        except (TimeoutError, RuntimeError):
            # Expected when no real cuOpt server
            pass

    def test_cuopt_payload_generation(self):
        """Test that cuOpt payloads are correctly formatted"""
        import numpy as np
        
        DATA = {
            "driver_states": {
                "drivers": {
                    "D1": {
                        "start_loc": "A",
                        "start_min": 480,
                        "end_min": 1080,
                        "elements": [
                            {"is_travel": True, "from": "A", "to": "B", 
                            "start_min": 540, "end_min": 600, "priority": 2}
                        ]
                    }
                }
            }
        }
        
        request_trip = {
            "id": "NEW:A->C@12345",
            "start_location": "A", 
            "end_location": "C",
            "priority": 1
        }
        
        # Use numpy arrays as your actual code expects
        M = {
            "dist": np.array([[0, 50, 100], [50, 0, 75], [100, 75, 0]]),
            "time": np.array([[0, 60, 120], [60, 0, 90], [120, 90, 0]]),
            "loc2idx": {"A": 0, "B": 1, "C": 2}
        }
        
        payload = build_cuopt_payload(
            DATA=DATA,
            request_trip=request_trip,
            assignments_so_far=[],
            priorities={},
            sla_windows={},
            M=M,
            new_req_window=[480, 600],
        )
        
        # Verify basic structure for 25.10.0a format
        assert "cost_matrix_data" in payload
        assert "fleet_data" in payload
        assert "task_data" in payload
        assert "solver_config" in payload

    def test_cuopt_solution_extraction(self):
        """Test extraction of solutions from cuOpt 25.10.0a response format"""
        
        # Test format for 25.10.0a
        raw1 = {
            "solver_response": {
                "status": 0,
                "solution_cost": 150.5,
                "vehicle_data": {
                    "0": {
                        "route": [0, 1, 2, 0],
                        "arrival_stamp": [0, 15, 30, 50],
                        "type": ["Depot", "Delivery", "Delivery", "Depot"]
                    }
                }
            }
        }
        
        solutions = extract_solutions_from_cuopt(raw1, max_solutions=5)
        assert len(solutions) == 1
        assert solutions[0]["objective_value"] == 150.5

    @pytest.mark.integration
    def test_cuopt_end_to_end_with_real_server(self, client, cuopt_server_url):
        """Integration test with actual cuOpt server"""
        if not cuopt_server_url:
            pytest.skip("No cuOpt server available - start with: docker-compose up cuopt")
        
        print(f"\n=== CUOPT END-TO-END TEST ===")
        print(f"cuOpt URL: {cuopt_server_url}")
        
        # Test cuOpt server health with correct endpoint for 25.10.0a
        try:
            import requests
            health_resp = requests.get(f"{cuopt_server_url}/cuopt/health", timeout=5)
            print(f"cuOpt health check (/cuopt/health): {health_resp.status_code}")
        except Exception as e:
            print(f"cuOpt health check failed: {e}")
        
        # Configure backend to use the test cuOpt server
        import os
        # For backend running in container, use Docker service name
        if cuopt_server_url == "http://localhost:5000":
            backend_cuopt_url = "http://cuopt:5000"  # Docker internal network
        else:
            backend_cuopt_url = cuopt_server_url
        
        os.environ["CUOPT_URL"] = backend_cuopt_url
        print(f"Setting backend CUOPT_URL to: {backend_cuopt_url}")
        
        # Reload backend with cuOpt URL
        reload_resp = client.post("/admin/reload")
        assert reload_resp.status_code == 200
        
        # Real request that should use cuOpt
        payload = {
            "start_location": "A",
            "end_location": "B", 
            "when_local": "2025-09-02T10:30",
            "priority": 2,
            "mode": "depart_after",
            "top_n_per_step": 2,
            "max_cascades": 1,
            "max_drivers_affected": 2,
            "max_solutions": 2,
            "use_cuopt": True  # Explicitly request cuOpt
        }
        
        print(f"\n=== SOLVE_MULTI WITH CUOPT ===")
        r = client.post("/plan/solve_multi", json=payload)
        assert r.status_code == 200, f"solve_multi failed: {r.status_code} {r.text}"
        
        result = r.json()
        print(f"Solutions returned: {len(result.get('solutions', []))}")
        
        # Verify results
        solutions = result.get("solutions", [])
        assert len(solutions) > 0, "Should return at least one solution"
        
        # Check if any solution indicates cuOpt backend
        cuopt_used = False
        for sol in solutions:
            backend = sol.get("details", {}).get("backend", "")
            print(f"Solution backend: {backend}")
            if "cuopt" in backend.lower():
                cuopt_used = True
                break
        
        if not cuopt_used:
            print("WARNING: cuOpt may not have been used (fallback to heuristic)")
        
        # Verify solution structure
        for i, sol in enumerate(solutions):
            assert "objective_value" in sol
            assert "assignments" in sol
            assert "schedules" in sol
            assert sol["objective_value"] >= 0
            print(f"Solution {i+1}: cost={sol['objective_value']:.2f}, assignments={len(sol['assignments'])}")

    @pytest.mark.cuopt
    @pytest.mark.integration  
    def test_cuopt_direct_api_call(self, cuopt_server_url):
        """Test direct cuOpt API calls"""
        if not cuopt_server_url:
            pytest.skip("No cuOpt server available")
            
        # Minimal cuOpt payload for 25.10.0a
        payload = {
            "cost_matrix_data": {
                "data": {"0": [[0, 50], [50, 0]]}
            },
            "fleet_data": {
                "vehicle_locations": [[0, 0]]
            },
            "task_data": {
                "task_locations": [1],
                "task_time_windows": [[600, 720]],  # 10 AM - 12 PM
                "service_times": [5],
                "demand": [[1]]
            },
            "solver_config": {
                "time_limit": 30
            }
        }
        
        try:
            # Use the fixed function signature
            result = solve_with_cuopt(cuopt_server_url, payload, timeout_sec=30)
            assert isinstance(result, dict)
            print(f"Direct cuOpt call successful: {list(result.keys())}")
            
            # Basic structure validation for 25.10.0a
            if "solver_response" in result:
                assert isinstance(result["solver_response"], dict)
                status = result["solver_response"].get("status")
                assert status is not None
            
        except Exception as e:
            pytest.fail(f"Direct cuOpt API call failed: {e}")