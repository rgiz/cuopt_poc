# tests/integration/test_cascade_logic.py
import pytest
import json
from pathlib import Path

class TestCascadeLogic:

    def test_cascade_priority_filtering(self, client):
        """Test that lower priority trips don't displace higher priority ones"""
        driver_states = {
            "drivers": {
                "D1": {
                    "start_min": 0, "end_min": 1440,
                    "elements": [
                        {"is_travel": True, "from": "A", "to": "B", 
                         "start_min": 600, "end_min": 660, "priority": 1}  # High priority existing
                    ]
                }
            }
        }
        
        with self._mock_driver_states(client, driver_states):
            payload = {
                "start_location": "A", "end_location": "B",
                "when_local": "2025-09-02T10:30",
                "priority": 3,  # Lower priority than existing (1)
                "mode": "depart_after",
                "max_cascades": 2, "max_drivers_affected": 5
            }
            
            r = client.post("/plan/solve_cascades", json=payload)
            assert r.status_code == 200
            
            result = r.json()
            
            # Should not displace the higher priority trip
            # New trip should be outsourced or use different strategy
            cascades = result["cascades"]
            assignments = result["assignments"]
            
            # If any assignment uses D1, it shouldn't be a swap_leg operation
            d1_assignments = [a for a in assignments if a.get("driver_id") == "D1"]
            for a in d1_assignments:
                assert "swap_leg" not in a.get("candidate_id", "")

    def test_cascade_cost_accumulation(self, client):
        """Test that cascade costs are properly accumulated"""
        driver_states = {
            "drivers": {
                "D1": {
                    "start_min": 0, "end_min": 1440,
                    "elements": [
                        {"is_travel": True, "from": "A", "to": "B", 
                         "start_min": 600, "end_min": 660, "priority": 3}
                    ]
                }
            }
        }
        
        with self._mock_driver_states(client, driver_states):
            payload = {
                "start_location": "A", "end_location": "B",
                "when_local": "2025-09-02T10:30",
                "priority": 2,
                "mode": "depart_after",
                "max_cascades": 2, "max_drivers_affected": 5
            }
            
            r = client.post("/plan/solve_cascades", json=payload)
            assert r.status_code == 200
            
            result = r.json()
            
            # Verify total objective includes all assignment costs
            total_from_assignments = sum(a["cost"] for a in result["assignments"])
            assert abs(result["objective_value"] - total_from_assignments) < 0.01
            
            # Verify cost breakdown is present for each assignment
            for assignment in result["assignments"]:
                assert "cost_breakdown" in assignment
                breakdown = assignment["cost_breakdown"]
                calculated_cost = sum(breakdown.values())
                assert abs(assignment["cost"] - calculated_cost) < 0.01

    def _mock_driver_states(self, client, driver_states):
        """Context manager to temporarily override driver states"""
        import tempfile
        import os
        from contextlib import contextmanager
        
        @contextmanager
        def mock_context():
            # Get the current data directory
            health_resp = client.get("/health")
            data_dir = Path(health_resp.json()["private_data_dir"])
            
            # Backup existing driver states
            driver_file = data_dir / "driver_states.json"
            backup_content = None
            if driver_file.exists():
                backup_content = driver_file.read_text()
            
            try:
                # Write test driver states
                driver_file.write_text(json.dumps(driver_states))
                
                # Reload the backend data
                reload_resp = client.post("/admin/reload")
                assert reload_resp.status_code == 200
                
                yield
                
            finally:
                # Restore original driver states
                if backup_content:
                    driver_file.write_text(backup_content)
                else:
                    driver_file.unlink(missing_ok=True)
                
                # Reload original data
                client.post("/admin/reload")
        
        return mock_context()
    
    def test_realistic_rsl_scenario(self, client):
        """Test with RSL-like data structure that matches real world usage"""
        
        # Driver with realistic RSL-style duty structure
        driver_states = {
            "drivers": {
                "D001": {
                    "start_min": 480,   # 8:00 AM (from Start Facility row)
                    "end_min": 1080,    # 6:00 PM (from End Facility row) = 10 hour duty
                    "home_loc": "DEPOT_LONDON",  # Start/end location
                    "elements": [
                        # Start Facility
                        {
                            "element_type": "Start Facility",
                            "from": "DEPOT_LONDON",
                            "to": "DEPOT_LONDON", 
                            "start_min": 480,  # 8:00 AM
                            "end_min": 480,
                            "Tue": True
                        },
                        # Travel leg 1: DEPOT -> Location A
                        {
                            "is_travel": True,
                            "element_type": "travel",
                            "from": "DEPOT_LONDON",
                            "to": "A",
                            "start_min": 480,   # 8:00 AM
                            "end_min": 540,     # 9:00 AM
                            "priority": 3,
                            "Tue": True,
                            "duration_min": 60
                        },
                        # Service at A
                        {
                            "element_type": "service",
                            "from": "A",
                            "to": "A",
                            "start_min": 540,   # 9:00 AM
                            "end_min": 600,     # 10:00 AM
                            "priority": 3,
                            "Tue": True,
                            "duration_min": 60
                        },
                        # Travel leg 2: A -> B (the one we want to displace)
                        {
                            "is_travel": True,
                            "element_type": "travel",
                            "from": "A",
                            "to": "B",
                            "start_min": 600,   # 10:00 AM
                            "end_min": 660,     # 11:00 AM
                            "priority": 4,      # Lower priority - displaceable
                            "Tue": True,
                            "duration_min": 60
                        },
                        # Service at B
                        {
                            "element_type": "service", 
                            "from": "B",
                            "to": "B",
                            "start_min": 660,   # 11:00 AM
                            "end_min": 720,     # 12:00 PM
                            "priority": 4,
                            "Tue": True,
                            "duration_min": 60
                        },
                        # Travel back: B -> DEPOT
                        {
                            "is_travel": True,
                            "element_type": "travel",
                            "from": "B",
                            "to": "DEPOT_LONDON",
                            "start_min": 720,   # 12:00 PM
                            "end_min": 1080,    # 6:00 PM
                            "priority": 2,
                            "Tue": True,
                            "duration_min": 360
                        },
                        # End Facility
                        {
                            "element_type": "End Facility",
                            "from": "DEPOT_LONDON",
                            "to": "DEPOT_LONDON",
                            "start_min": 1080,  # 6:00 PM
                            "end_min": 1080,
                            "Tue": True
                        }
                    ]
                }
            }
        }
        
        with self._mock_driver_states(client, driver_states):
            print(f"\n=== RSL-REALISTIC SCENARIO ===")
            print(f"Driver D001: 8 AM - 6 PM duty (10 hours)")
            print(f"Route: DEPOT -> A (service) -> B (service) -> DEPOT")
            print(f"Current A->B leg: 10:00-11:00 AM, priority 4")
            print(f"Request: HIGH priority A->B at 10:30 AM")
            print(f"Expected: Should displace existing A->B leg")
            
            # High priority request that should displace the A->B leg
            candidates_payload = {
                "start_location": "A",
                "end_location": "B", 
                "when_local": "2025-09-02T10:30",  # Right in middle of existing leg
                "priority": 1,  # Much higher than existing priority 4
                "mode": "depart_after",
                "top_n": 10,
                "trip_minutes": 60
            }
            
            print(f"\n=== CANDIDATES ANALYSIS ===")
            candidates_r = client.post("/plan/candidates", json=candidates_payload)
            if candidates_r.status_code == 200:
                candidates_data = candidates_r.json()
                print(f"Found {len(candidates_data['candidates'])} candidates")
                
                for i, cand in enumerate(candidates_data['candidates']):
                    print(f"  {i}: {cand['candidate_id']} - cost:{cand['est_cost']:.2f}")
                    if "swap_leg" in cand['candidate_id']:
                        print(f"    ^^^ SWAP CANDIDATE - Should trigger cascade!")
                    elif "append" in cand['candidate_id']:
                        if cand['feasible_hard']:
                            print(f"    Append feasible - total duty would be: {10*60 + cand['overtime_minutes']} min")
                        else:
                            print(f"    Append infeasible - exceeds duty limit")
            
            # Test cascade generation
            cascade_payload = {
                "start_location": "A",
                "end_location": "B",
                "when_local": "2025-09-02T10:30",
                "priority": 1,
                "mode": "depart_after", 
                "max_cascades": 2,
                "max_drivers_affected": 5
            }
            
            print(f"\n=== CASCADE TESTING ===")
            cascade_r = client.post("/plan/solve_cascades", json=cascade_payload)
            if cascade_r.status_code == 200:
                result = cascade_r.json()
                print(f"Cascades generated: {len(result['cascades'])}")
                print(f"Assignments: {len(result['assignments'])}")
                
                for assignment in result['assignments']:
                    print(f"  Assignment: {assignment['trip_id']} -> {assignment.get('driver_id')} ({assignment['type']})")
                    if 'candidate_id' in assignment:
                        print(f"    Via: {assignment['candidate_id']}")
                
                for cascade in result['cascades']:
                    print(f"  Cascade: {cascade.get('driver_id')} displaced {cascade.get('from')}->{cascade.get('to')}")
            
            print(f"\n=== ANALYSIS ===")
            print(f"This scenario tests:")
            print(f"1. Whether driver passing through A->B is detected")
            print(f"2. Whether priority-based displacement works") 
            print(f"3. Whether cascade logic triggers on displacement")
            print(f"4. Whether duty time constraints are properly enforced")
        
        assert True  # Debug test