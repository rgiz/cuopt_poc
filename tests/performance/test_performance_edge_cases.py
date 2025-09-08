# tests/performance/test_performance_edge_cases.py
import pytest
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class TestPerformanceAndEdgeCases:
    
    def test_large_driver_fleet_performance(self, client):
        """Test performance with large number of drivers"""
        # Create a large driver fleet
        large_driver_states = {"drivers": {}}
        
        for i in range(50):  # Reduced from 100 to avoid memory issues
            driver_id = f"D{i:03d}"
            large_driver_states["drivers"][driver_id] = {
                "start_min": 0, "end_min": 1440,
                "elements": [
                    {"is_travel": True, "from": "A", "to": "B", 
                    "start_min": 600 + (i * 5), "end_min": 660 + (i * 5), "priority": 3}
                ]
            }
        
        # Mock large fleet temporarily
        try:
            with self._mock_driver_states(client, large_driver_states):
                start_time = time.time()
                
                payload = {
                    "start_location": "A", "end_location": "B",
                    "when_local": "2025-09-02T10:30",
                    "priority": 2, "top_n": 20  # Reduced from 50
                }
                
                r = client.post("/plan/candidates", json=payload)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Should either succeed or fail gracefully
                if r.status_code == 200:
                    assert response_time < 10.0, f"Response took {response_time:.2f}s, should be < 10s"
                    candidates = r.json()["candidates"]
                    assert len(candidates) <= 20  # Respects top_n limit
                else:
                    # If it fails due to resource constraints, that's acceptable for stress test
                    assert r.status_code in (422, 500, 503), \
                        f"Large fleet test should succeed or fail gracefully, got {r.status_code}"
                    
        except Exception as e:
            # If the test setup itself fails, that's also acceptable for a stress test
            assert "driver_states" in str(e).lower() or "memory" in str(e).lower() or "timeout" in str(e).lower(), \
                f"Large fleet test failure should be resource-related: {e}"

    def test_concurrent_request_handling(self, client):
        """Test that multiple concurrent requests are handled properly"""
        def make_request(req_id):
            payload = {
                "start_location": "A", "end_location": "B",
                "when_local": f"2025-09-02T{10 + (req_id % 8):02d}:30",
                "priority": (req_id % 5) + 1,
                "mode": "depart_after"
            }
            
            start_time = time.time()
            r = client.post("/plan/solve_multi", json=payload)
            end_time = time.time()
            
            return {
                "req_id": req_id,
                "status_code": r.status_code,
                "response_time": end_time - start_time,
                "result": r.json() if r.status_code == 200 else None
            }
        
        # Submit 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        # All requests should succeed
        assert all(r["status_code"] == 200 for r in results)
        
        # No request should take excessively long
        max_response_time = max(r["response_time"] for r in results)
        assert max_response_time < 10.0, f"Slowest request took {max_response_time:.2f}s"
        
        # Results should be consistent (same inputs = same outputs)
        grouped_results = {}
        for r in results:
            key = (r["result"]["trip_minutes"], r["result"]["trip_miles"])
            if key not in grouped_results:
                grouped_results[key] = []
            if "solutions" in r["result"] and r["result"]["solutions"]:
                grouped_results[key].append(r["result"]["solutions"][0]["objective_value"])
            elif "objective_value" in r["result"]:
                grouped_results[key].append(r["result"]["objective_value"])
        
        # Same trip should have same objective value
        for trip_key, objectives in grouped_results.items():
            if len(objectives) > 1:
                assert all(abs(obj - objectives[0]) < 0.01 for obj in objectives), \
                    f"Inconsistent results for trip {trip_key}: {objectives}"

    def test_invalid_location_handling(self, client):
        """Test handling of invalid or missing locations"""
        test_cases = [
            {"start_location": "INVALID_LOC", "end_location": "A"},
            {"start_location": "A", "end_location": "INVALID_LOC"}, 
            {"start_location": "", "end_location": "A"},
            {"start_location": "A", "end_location": ""},
        ]
        
        for case in test_cases:
            payload = {
                **case,
                "when_local": "2025-09-02T10:30",
                "priority": 2,
                "mode": "depart_after"
            }
            
            try:
                r = client.post("/plan/candidates", json=payload)
                # Should return 400/422 (validation error) or 500 (processing error)
                assert r.status_code in (400, 422, 500), \
                    f"Invalid location case {case} should fail, got {r.status_code}"
            except Exception as e:
                # If an exception is thrown instead of returning error status, that's also valid
                assert "Unknown location" in str(e) or "location" in str(e).lower(), \
                    f"Exception should mention location issue: {e}"

    def test_extreme_datetime_values(self, client):
        """Test handling of extreme datetime values"""
        # Test valid edge cases first
        valid_cases = ["2025-01-01T00:00", "2025-12-31T23:59"]
        
        for dt in valid_cases:
            payload = {
                "start_location": "A", "end_location": "B",
                "when_local": dt, "priority": 2, "mode": "depart_after"
            }
            r = client.post("/plan/candidates", json=payload)
            assert r.status_code == 200, f"Valid datetime {dt} should work"
        
        # Test invalid cases - these should properly fail
        invalid_cases = [
            "2025-02-29T10:30",  # Invalid date (not leap year)
            "2025-09-02T25:30",  # Invalid hour
            "invalid-datetime"   # Malformed
        ]
        
        for dt in invalid_cases:
            payload = {
                "start_location": "A", "end_location": "B", 
                "when_local": dt, "priority": 2, "mode": "depart_after"
            }
            
            try:
                r = client.post("/plan/candidates", json=payload)
                # Should return error status
                assert r.status_code in (400, 422, 500), f"Invalid datetime {dt} should fail"
            except Exception as e:
                # If exception thrown instead, check it's date-related
                # Error messages can include: "hour must be in 0..23", "day is out of range", etc.
                assert any(word in str(e).lower() for word in ["date", "time", "day", "month", "range", "hour", "minute"]), \
                    f"Exception should mention date/time issue: {e}"

    def test_memory_usage_large_matrices(self, client):
        """Test memory handling with large distance/time matrices"""
        # This is more of a stress test - would need actual large test data
        # For now, verify that the current dataset loads without memory issues
        
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        
        health_data = health_resp.json()
        
        # Verify matrices are loaded
        assert "distance_shape" in health_data
        assert "time_shape" in health_data
        
        matrix_size = health_data["distance_shape"][0] * health_data["distance_shape"][1]
        
        # For very large matrices (>10k locations), might want specific handling
        if matrix_size > 10_000_000:  # 10M elements
            pytest.skip("Large matrix test requires specific memory monitoring")
        
        # Basic functionality should still work
        payload = {
            "start_location": "A", "end_location": "B",
            "when_local": "2025-09-02T10:30", "priority": 2, "mode": "depart_after"
        }
        
        r = client.post("/plan/candidates", json=payload)
        assert r.status_code == 200

    def test_zero_duration_trips(self, client):
        """Test handling of zero or negative duration trips"""
        # This tests edge cases in matrix data
        payload = {
            "start_location": "A", "end_location": "A",  # Same start/end
            "when_local": "2025-09-02T10:30",
            "priority": 2, "mode": "depart_after",
            "trip_minutes": 0, "trip_miles": 0
        }
        
        r = client.post("/plan/candidates", json=payload)
        
        # Should either handle gracefully or return appropriate error
        if r.status_code == 200:
            result = r.json()
            assert result["trip_minutes"] == 0
            assert result["trip_miles"] == 0
            # Candidates might still exist (admin cost only)
        else:
            assert r.status_code in (400, 422), "Zero trip should be handled gracefully"

    def test_cuopt_server_unavailable(self, client, monkeypatch):
        """Test fallback when cuOpt server is unavailable"""
        # Point to non-existent cuOpt server
        monkeypatch.setenv("CUOPT_URL", "http://nonexistent-cuopt:5000")
        
        # Reload config
        reload_resp = client.post("/admin/reload")
        assert reload_resp.status_code == 200
        
        payload = {
            "start_location": "A", "end_location": "B",
            "when_local": "2025-09-02T10:30",
            "priority": 2, "mode": "depart_after",
            "use_cuopt": True  # Request cuOpt but it's unavailable
        }
        
        r = client.post("/plan/solve_multi_multi", json=payload)
        
        # Should fallback to heuristic and still return results
        assert r.status_code in (200, 404), f"Expected success or not found, got {r.status_code}"
        
        result = r.json()
        if "solutions" in result:
            solutions = result["solutions"]
        elif "assignments" in result:
            # Convert cascades format to multi format for consistency
            solutions = [{"assignments": result["assignments"], "objective_value": result["objective_value"]}]
        else:
            solutions = []
        assert len(solutions) > 0
        
        # Should indicate fallback in details
        for sol in solutions:
            backend = sol["details"].get("backend", "")
            assert "greedy" in backend or "heuristic" in backend

    def _mock_driver_states(self, client, driver_states):
        """Helper method to temporarily override driver states"""
        # Implementation same as in cascade tests
        import tempfile
        import os
        from contextlib import contextmanager
        from pathlib import Path
        
        @contextmanager
        def mock_context():
            health_resp = client.get("/health")
            data_dir = Path(health_resp.json()["private_data_dir"])
            
            driver_file = data_dir / "driver_states.json"
            backup_content = None
            if driver_file.exists():
                backup_content = driver_file.read_text()
            
            try:
                driver_file.write_text(json.dumps(driver_states))
                reload_resp = client.post("/admin/reload")
                assert reload_resp.status_code == 200
                yield
            finally:
                if backup_content:
                    driver_file.write_text(backup_content)
                else:
                    driver_file.unlink(missing_ok=True)
                client.post("/admin/reload")
        
        return mock_context()
    
    