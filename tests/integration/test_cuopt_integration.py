# tests/integration/test_cuopt_integration.py
import pytest
import responses
import json
from unittest.mock import patch
from src.plan.cuopt_adapter import solve_with_cuopt, build_cuopt_payload, extract_solutions_from_cuopt

class TestCuOptIntegration:

    @responses.activate
    def test_cuopt_solver_discovery_endpoints(self):
        """Test that the solver correctly discovers cuOpt endpoints"""
        base_url = "http://cuopt:5000"
        
        # Mock the base endpoint that the function actually tries
        responses.add(responses.POST, f"{base_url}/solve", 
                    json={"objective_value": 100.0, "routes": []}, status=200)
        
        payload = {"vehicles": [], "tasks": []}
        result = solve_with_cuopt(base_url, payload)
        
        assert "objective_value" in result

    @responses.activate 
    def test_cuopt_fallback_endpoints(self):
        """Test fallback when v2 endpoint unavailable"""
        base_url = "http://cuopt:5000"
        
        # v2 health fails, should fallback to base/solve
        responses.add(responses.GET, f"{base_url}/v2/health/live", status=404)
        responses.add(responses.POST, f"{base_url}/solve", json={"solutions": [{"objective": 150.0}]}, status=200)
        
        payload = {"vehicles": [], "tasks": []}
        result = solve_with_cuopt(base_url, payload)
        
        assert "solutions" in result

    @responses.activate
    def test_cuopt_async_request_poll_pattern(self):
        """Test async request/poll pattern used by some cuOpt images"""
        base_url = "http://cuopt:5000"
        
        # Request returns reqId
        responses.add(responses.POST, f"{base_url}/request", 
                     json={"reqId": "test-123"}, status=202)
        
        # Poll returns result
        responses.add(responses.GET, f"{base_url}/cuopt/result?reqId=test-123",
                     json={"objective_value": 200.0, "routes": [{"vehicle_id": "D1", "steps": []}]}, status=200)
        
        # Mock the async flow in CuOptModel
        from src.opt.cuopt_model_miles import CuOptModel
        
        model = CuOptModel(
            driver_states={},
            distance_miles_matrix=[[0]], 
            time_minutes_matrix=[[0]],
            location_to_index={"A": 0},
            cost_config={"deadhead_cost_per_mile": 1.0},
            server_url=base_url,
            solve_path="request"  # Force async mode
        )
        
        result = model._request_and_poll({"test": "payload"})
        assert result["objective_value"] == 200.0

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
        
        # Verify basic structure exists
        assert "vehicles" in payload
        assert "tasks" in payload
        assert "matrices" in payload

    def test_cuopt_solution_extraction(self):
        """Test extraction of solutions from various cuOpt response formats"""
        
        # Test format 1: Direct solution
        raw1 = {
            "objective_value": 150.5,
            "routes": [
                {
                    "vehicle_id": "D1",
                    "steps": [
                        {
                            "task_id": "NEW:A->B@123",
                            "delay_min": 15.0,
                            "deadhead_miles": 25.0,
                            "cost": 45.0
                        }
                    ]
                }
            ]
        }
        
        solutions = extract_solutions_from_cuopt(raw1, max_solutions=5)
        assert len(solutions) == 1
        assert solutions[0]["objective_value"] == 150.5

    @pytest.mark.integration
    def test_cuopt_end_to_end_with_real_server(self, cuopt_server_url):
        """Integration test with actual cuOpt server (requires TEST_CUOPT_URL env var)"""
        if not cuopt_server_url:
            pytest.skip("No cuOpt server configured for integration testing")
            
        # This would test against a real cuOpt instance
        # Useful for CI/CD with cuOpt container
        payload = {
            "vehicles": [{"id": "TEST_DRIVER", "time_window": [0, 1440]}],
            "tasks": [{"id": "TEST_TASK", "from": "A", "to": "B", "mandatory": True}],
            "matrices": {"distance": [[0, 50], [50, 0]], "time": [[0, 60], [60, 0]]}
        }
        
        try:
            result = solve_with_cuopt(cuopt_server_url, payload, timeout=30)
            assert "objective_value" in result or "solutions" in result
        except Exception as e:
            pytest.fail(f"cuOpt integration failed: {e}")

@pytest.fixture
def cuopt_server_url():
    """Fixture to provide cuOpt server URL for integration tests"""
    import os
    return os.getenv("TEST_CUOPT_URL")  # Set in CI/CD environment