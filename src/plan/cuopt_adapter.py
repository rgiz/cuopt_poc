from __future__ import annotations
import os
from typing import Any, Dict, Optional, Sequence, List
import requests
import time
import json
from urllib.parse import urljoin

DEFAULT_TIMEOUT = float(os.getenv("CUOPT_SOLVER_TIMEOUT_SEC", "120"))

def solve_with_cuopt(raw_base_url: Optional[str], payload: Dict[str, Any], timeout_sec: Optional[float] = None) -> Dict[str, Any]:
    """
    Call cuOpt server with a payload using the new async pattern for 25.10.0a.
    """
    timeout = float(timeout_sec or DEFAULT_TIMEOUT)
    base_env = raw_base_url or os.getenv("CUOPT_URL", "http://localhost:5000")
    base = base_env.rstrip("/")

    print(f"🔍 DEBUG: cuOpt Base URL: {base}")
    print(f"🔍 DEBUG: Timeout: {timeout}s")
    print(f"🔍 DEBUG: Payload size: {len(json.dumps(payload))} characters")

    # Use correct cuOpt 25.x endpoint
    headers = {
        'Content-Type': 'application/json',
        'CLIENT-VERSION': 'custom'
    }
    
    submit_url = f"{base}/cuopt/request"
    print(f"🔍 DEBUG: Submit URL: {submit_url}")
    
    try:
        # Step 1: Submit request to /cuopt/request (not /solve)
        print("🚀 DEBUG: Submitting request to cuOpt...")
        start_submit_time = time.time()
        
        response = requests.post(
            submit_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        submit_time = time.time() - start_submit_time
        print(f"✅ DEBUG: Submit completed in {submit_time:.2f}s")
        print(f"🔍 DEBUG: Response status: {response.status_code}")
        print(f"🔍 DEBUG: Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        initial = response.json()
        print(f"🔍 DEBUG: Initial response: {json.dumps(initial, indent=2)}")
        
        # Check for immediate response (rare in 25.x)
        if 'response' in initial and 'solver_response' in initial['response']:
            print("⚡ DEBUG: Got immediate response!")
            return initial['response']
            
        # Get request ID for polling
        request_id = initial.get('reqId')
        if not request_id:
            print(f"❌ DEBUG: No request ID in response: {initial}")
            raise ValueError(f"No request ID in response: {initial}")
            
        print(f"🔍 DEBUG: Got request ID: {request_id}")
        
        # Step 2: Poll for results
        poll_url = f"{base}/cuopt/requests/{request_id}"
        print(f"🔍 DEBUG: Poll URL: {poll_url}")
        
        start_time = time.time()
        poll_count = 0
        
        while time.time() - start_time < timeout:
            poll_count += 1
            elapsed = time.time() - start_time
            print(f"🔄 DEBUG: Poll attempt #{poll_count} (elapsed: {elapsed:.1f}s)")
            
            try:
                poll_response = requests.get(
                    poll_url,
                    headers={'Content-Type': 'application/json'},
                    timeout=10  # Shorter timeout for individual polls
                )
                
                print(f"🔍 DEBUG: Poll response status: {poll_response.status_code}")
                
                if poll_response.status_code == 200:
                    result = poll_response.json()
                    print(f"🔍 DEBUG: Poll response: {json.dumps(result, indent=2)}")
                    
                    if 'response' in result:
                        solver_response = result['response'].get('solver_response', {})
                        if solver_response:
                            status = solver_response.get('status')
                            print(f"✅ DEBUG: Got solver response with status: {status}")
                            return result['response']
                        else:
                            print("🔍 DEBUG: Response exists but no solver_response yet")
                    else:
                        print("🔍 DEBUG: No 'response' field in result yet")
                        
                elif poll_response.status_code == 404:
                    # Still processing, continue polling
                    print("⏳ DEBUG: Request still processing (404)")
                else:
                    print(f"⚠️ DEBUG: Unexpected poll status: {poll_response.status_code}")
                    print(f"⚠️ DEBUG: Poll response text: {poll_response.text}")
                    poll_response.raise_for_status()
            
            except requests.exceptions.RequestException as e:
                print(f"⚠️ DEBUG: Poll request failed: {e}")
                # Continue polling on network errors
                
            print(f"⏳ DEBUG: Sleeping 2s before next poll...")
            time.sleep(2)  # Poll every 2 seconds
            
        # Timeout reached
        final_elapsed = time.time() - start_time  
        print(f"⏰ DEBUG: Timeout reached after {final_elapsed:.1f}s ({poll_count} polls)")
        raise TimeoutError(f"cuOpt request {request_id} timed out after {timeout}s")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ DEBUG: HTTP Error: {e}")
        print(f"❌ DEBUG: Response status: {e.response.status_code}")
        print(f"❌ DEBUG: Response text: {e.response.text}")