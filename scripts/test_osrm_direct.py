"""
Standalone test of OSRM Table API to isolate the issue.
"""

import requests
import numpy as np

def test_osrm_table():
    """Test OSRM table with the exact same coordinates that are failing."""
    
    # Use the same coordinates from your failing batch
    coordinates = [
        (-0.766641, 51.330863),
        (-1.108688, 51.272315), 
        (-1.125141, 52.785473),
        (-0.408345, 51.439532)
    ]
    
    osrm_url = "http://localhost:5001"
    
    # Format exactly like the working curl command
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
    url = f"{osrm_url}/table/v1/driving/{coord_str}"
    
    # ONLY annotations parameter - no sources/destinations
    params = {
        "annotations": "duration,distance"
    }
    
    print(f"Testing OSRM with {len(coordinates)} coordinates")
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Full request URL: {response.url}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok":
                distances = np.array(data["distances"])
                durations = np.array(data["durations"]) 
                print(f"SUCCESS! Got {distances.shape} matrix")
                print(f"Sample distance: {distances[0,1]:.1f} meters")
                print(f"Sample duration: {durations[0,1]:.1f} seconds")
                return True
            else:
                print(f"OSRM error: {data.get('message')}")
                return False
        else:
            print(f"HTTP error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_osrm_table()
    if success:
        print("\n✅ OSRM test PASSED - The API call works!")
    else:
        print("\n❌ OSRM test FAILED - API call broken")