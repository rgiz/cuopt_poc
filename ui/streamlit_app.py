import os, requests, pandas as pd, streamlit as st
import json

API = os.getenv("API_BASE_URL", "http://backend:8000")

st.set_page_config(page_title="Dynamic Trip Rescheduling", layout="wide")
st.title("Dynamic Trip Rescheduling")

# Helper function to format minutes to HH:MM
def format_time(minutes):
    """Convert minutes since midnight to HH:MM format"""
    if pd.isna(minutes) or minutes is None:
        return "—"
    try:
        total_minutes = int(minutes)
        hours = total_minutes // 60
        mins = total_minutes % 60
        return f"{hours:02d}:{mins:02d}"
    except (ValueError, TypeError):
        return str(minutes)

# Initialize session state
if 'selected_candidate' not in st.session_state:
    st.session_state.selected_candidate = None
if 'candidates_data' not in st.session_state:
    st.session_state.candidates_data = None
if 'trip_info' not in st.session_state:
    st.session_state.trip_info = None

# --- SECTION 1: INPUTS ---
st.header("1. Trip Details")

# Fetch location metadata
try:
    locs_resp = requests.get(f"{API}/plan/locations", timeout=10).json()
    locations = locs_resp.get("locations", [])
    locations = sorted(locations, key=lambda x: x["name"])
    names = [loc["name"] for loc in locations]
    name_to_postcode = {loc["name"]: loc.get("postcode", "N/A") for loc in locations}
except Exception as e:
    st.error(f"Failed to load locations: {e}")
    names = []
    name_to_postcode = {}

# Location inputs
col1, col2 = st.columns(2)
with col1:
    start = st.selectbox(
        "Start location", 
        names, 
        key="start_location",
        help="Type to search locations"
    )
    if start:
        st.text(f"📍 {name_to_postcode.get(start, 'No postcode')}")

with col2:
    end = st.selectbox(
        "End location", 
        names, 
        key="end_location",
        help="Type to search locations"
    )
    if end:
        st.text(f"📍 {name_to_postcode.get(end, 'No postcode')}")

# Trip parameters
col3, col4, col5 = st.columns(3)
with col3:
    mode = st.selectbox("Mode", ["depart_after", "arrive_before"], key="mode")
with col4:
    dt = st.text_input("When (Europe/London)", "2025-09-02T10:30", key="datetime")
with col5:
    priority = st.slider("Priority (1=highest, 5=lowest)", 1, 5, 3, key="priority")

# Advanced options (collapsible)
with st.expander("Advanced Options"):
    topn = st.slider("Max candidates to show", 3, 20, 10, key="topn")

# --- SECTION 2: FIND CANDIDATES ---
st.header("2. Find Candidates")

if st.button("Find Candidates", disabled=not (start and end), type="primary", key="find_candidates"):

    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n": topn
    }
    
    with st.spinner("Searching for candidates..."):
        try:
            r = requests.post(f"{API}/plan/candidates", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            
            # Store in session state
            st.session_state.candidates_data = data
            st.session_state.trip_info = {
                "trip_minutes": data['trip_minutes'],
                "trip_miles": data['trip_miles'],
                "weekday": data['weekday'],
                "start": start,
                "end": end,
                "mode": mode,
                "datetime": dt,
                "priority": priority
            }
            st.session_state.selected_candidate = None  # Reset selection
            
            st.success(f"Found {len(data.get('candidates', []))} candidates")
            
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.session_state.candidates_data = None

# # Display candidates if available
# if st.session_state.candidates_data:
#     data = st.session_state.candidates_data
#     trip_info = st.session_state.trip_info
    
#     st.info(f"Trip: {trip_info['trip_minutes']:.1f} mins, {trip_info['trip_miles']:.1f} miles, {trip_info['weekday']}")
    
#     candidates = data.get("candidates", [])
#     if candidates:
#         st.subheader("Available Candidates")
        
#         # Create a more user-friendly display
#         for i, candidate in enumerate(candidates):
#             candidate_summary = f"{candidate.get('driver_id', 'N/A')} - "
            
#             # Parse candidate type
#             cid = candidate.get('candidate_id', '')
#             if 'take_empty' in cid:
#                 candidate_summary += f"Take empty slot (Cost: £{candidate.get('est_cost', 0):.2f})"
#             elif 'swap_leg' in cid:
#                 candidate_summary += f"Swap existing leg (Cost: £{candidate.get('est_cost', 0):.2f})"
#             elif 'append' in cid:
#                 candidate_summary += f"Add to end of duty (Cost: £{candidate.get('est_cost', 0):.2f}, +{candidate.get('overtime_minutes', 0):.0f}min overtime)"
#             elif 'slack' in cid:
#                 candidate_summary += f"Use slack time (Cost: £{candidate.get('est_cost', 0):.2f})"
#             else:
#                 candidate_summary += f"Other solution (Cost: £{candidate.get('est_cost', 0):.2f})"
            
#             # Add feasibility indicator
#             if not candidate.get('feasible_hard', True):
#                 candidate_summary += " ⚠️ Not feasible"
            
#             # Create button for selection
#             if st.button(
#                 candidate_summary, 
#                 key=f"candidate_{i}",
#                 help=f"Deadhead: {candidate.get('deadhead_miles', 0):.1f} miles, Delay: {candidate.get('delay_minutes', 0):.0f} mins"
#             ):
#                 st.session_state.selected_candidate = candidate
#                 st.rerun()
#     else:
#         st.warning("No candidates found. Try adjusting the priority or time window.")

# SECTION 2: CANDIDATES LIST - Just show summary, NO schedules
if st.session_state.candidates_data:
    st.subheader("2. Available Candidates")
    
    candidates = st.session_state.candidates_data.get("candidates", [])
    
    if not candidates:
        st.warning("No viable candidates found.")
    else:
        st.info(f"Found {len(candidates)} potential solutions")
        
        # Simple list view with selection buttons
        for i, cand in enumerate(candidates):
            driver_id = cand.get("driver_id")
            cost = cand.get("est_cost", 0)
            cand_type = cand.get("type", "unknown")
            reason = cand.get("reason", "")
            
            # Create a concise one-line summary
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"**{i+1}. Driver {driver_id}** - £{cost:.2f} ({cand_type})")
                if reason:
                    st.caption(reason)
            
            with col2:
                if st.button("Select", key=f"select_{i}"):
                    st.session_state.selected_candidate = cand
                    st.rerun()

elif st.session_state.trip_info:
    st.info("👆 Click 'Find Candidates' to see available drivers")


# SECTION 3: SOLUTION DETAILS - Show schedules HERE
st.header("3. Solution Details")

if st.session_state.selected_candidate:
    candidate = st.session_state.selected_candidate
    
    st.success(f"Selected: Driver {candidate.get('driver_id')} - {candidate.get('candidate_id')}")
    
    # Show candidate metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cost", f"£{candidate.get('est_cost', 0):.2f}")
    with col2:
        st.metric("Deadhead Miles", f"{candidate.get('deadhead_miles', 0):.1f}")
    with col3:
        st.metric("Overtime Minutes", f"{candidate.get('overtime_minutes', 0):.0f}")
    
    # Button to show full cascade solution with schedules
    if st.button("Show Full Solution (with cascades)", type="primary"):
        
        # Get the schedule for the selected candidate from already-loaded data
        schedules_data = st.session_state.candidates_data.get("schedules", [])
        candidate_schedule = next(
            (s for s in schedules_data if s.get("driver_id") == candidate.get("driver_id")), 
            None
        )
        
        if candidate_schedule:
            # Build solution data directly without calling backend again
            trip_info = st.session_state.get('trip_info', {})
            
            solution_data = {
                "objective_value": candidate.get("est_cost", 0),
                "weekday": st.session_state.candidates_data.get("weekday"),
                "trip_minutes": st.session_state.candidates_data.get("trip_minutes"),
                "trip_miles": st.session_state.candidates_data.get("trip_miles"),
                "assignments": [{
                    "trip_id": f"NEW-{trip_info.get('start')}→{trip_info.get('end')}",
                    "type": "reassigned",
                    "driver_id": candidate.get("driver_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "cost": candidate.get("est_cost", 0),
                    "deadhead_miles": candidate.get("deadhead_miles", 0),
                    "overtime_minutes": candidate.get("overtime_minutes", 0),
                    "delay_minutes": candidate.get("delay_minutes", 0),
                }],
                "cascades": [{
                    "depth": 1,
                    "displaced_by": "NEW_SERVICE",
                    "driver_id": candidate.get("driver_id"),
                    "from": trip_info.get('start'),
                    "to": trip_info.get('end'),
                    "priority": trip_info.get('priority'),
                    "reason": candidate.get("reason", "")
                }],
                "schedules": [candidate_schedule]
            }
            
            # Store and display immediately - no backend call needed!
            st.session_state.solution_data = solution_data
            st.rerun()
        else:
            st.error("Schedule data not available for this candidate")

# if st.session_state.selected_candidate:
#     candidate = st.session_state.selected_candidate
    
#     st.success(f"Selected: Driver {candidate.get('driver_id')} - {candidate.get('candidate_id')}")
    
#     # Show candidate metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Cost", f"£{candidate.get('est_cost', 0):.2f}")
#     with col2:
#         st.metric("Deadhead Miles", f"{candidate.get('deadhead_miles', 0):.1f}")
#     with col3:
#         st.metric("Overtime Minutes", f"{candidate.get('overtime_minutes', 0):.0f}")
    
#     # Button to get full cascade solution with schedules
#     if st.button("Show Full Solution (with cascades)", type="primary"):
        
#         # Get trip details from session state
#         trip_info = st.session_state.get('trip_info', {})
        
#         cascade_payload = {
#             "start_location": trip_info.get('start'),
#             "end_location": trip_info.get('end'),
#             "mode": trip_info.get('mode', 'depart_after'),
#             "when_local": trip_info.get('datetime'),
#             "priority": trip_info.get('priority', 3),
#             "trip_minutes": st.session_state.candidates_data.get('trip_minutes'),  # Required field
#             "trip_miles": st.session_state.candidates_data.get('trip_miles'),      # Required field
#             "max_cascades": 2,
#             "max_drivers_affected": 5,
#             "preferred_candidate_id": candidate.get('candidate_id'),
#             "preferred_driver_id": candidate.get('driver_id')
#         }
        
#         with st.spinner("Computing full cascade solution..."):
#             try:
#                 r = requests.post(f"{API}/plan/solve_cascades", json=cascade_payload, timeout=120)
#                 r.raise_for_status()
#                 solution_data = r.json()
                
#                 # Store solution in session state
#                 st.session_state.solution_data = solution_data
#                 st.rerun()
                
#             except Exception as e:
#                 st.error(f"Failed to compute solution: {e}")
    
    # Display solution if available
    if 'solution_data' in st.session_state and st.session_state.solution_data:
        solution_data = st.session_state.solution_data
        
        st.divider()
        st.subheader("Complete Solution")
        
        # Summary metrics
        st.success(f"Total Solution Cost: £{solution_data.get('objective_value', 0):.2f}")
        
        # Trip assignments
        assignments = solution_data.get("assignments", [])
        if assignments:
            st.subheader("Trip Assignments")
            assign_df = pd.DataFrame(assignments)
            display_cols = ['trip_id', 'driver_id', 'type', 'cost', 'delay_minutes', 'overtime_minutes']
            available_cols = [col for col in display_cols if col in assign_df.columns]
            st.dataframe(assign_df[available_cols], use_container_width=True)
        
        # Cascade effects
        cascades = solution_data.get("cascades", [])
        if cascades:
            st.subheader("Cascade Effects")
            st.info(f"Found {len(cascades)} displaced trips that need reassignment")
            casc_df = pd.DataFrame(cascades)
            st.dataframe(casc_df, use_container_width=True)
        
        # NOW show the detailed schedules
        schedules = solution_data.get("schedules", [])
        
        if schedules:
            st.subheader("Driver Schedule Changes")
            
            for sch in schedules:
                driver_id = sch.get('driver_id')
                
                with st.expander(f"Driver {driver_id} Schedule", expanded=True):
                    col_before, col_after = st.columns(2)
                    
                    with col_before:
                        st.markdown("**Before:**")
                        before_data = sch.get("before", [])
                        if before_data:
                            before_df = pd.DataFrame(before_data)
                            
                            # Format time columns if they exist
                            if 'start_min' in before_df.columns:
                                before_df['start_time'] = before_df['start_min'].apply(
                                    lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}" if pd.notna(x) else "—"
                                )
                            if 'end_min' in before_df.columns:
                                before_df['end_time'] = before_df['end_min'].apply(
                                    lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}" if pd.notna(x) else "—"
                                )
                            
                            # Show relevant columns
                            display_cols = ['element_type', 'from', 'to', 'start_time', 'end_time', 'priority']
                            available_cols = [col for col in display_cols if col in before_df.columns]
                            
                            if available_cols:
                                st.dataframe(before_df[available_cols], use_container_width=True, hide_index=True)
                            else:
                                st.write("No schedule data available")
                    
                    with col_after:
                        st.markdown("**After:**")
                        after_data = sch.get("after", [])
                        if after_data:
                            after_df = pd.DataFrame(after_data)
                            
                            # Format time columns
                            if 'start_min' in after_df.columns:
                                after_df['start_time'] = after_df['start_min'].apply(
                                    lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}" if pd.notna(x) else "—"
                                )
                            if 'end_min' in after_df.columns:
                                after_df['end_time'] = after_df['end_min'].apply(
                                    lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}" if pd.notna(x) else "—"
                                )
                            
                            # Show columns with changes highlighted
                            display_cols = ['element_type', 'from', 'to', 'start_time', 'end_time', 'priority', 'changes']
                            available_cols = [col for col in display_cols if col in after_df.columns]
                            
                            if available_cols:
                                # Highlight changed rows
                                def highlight_changes(row):
                                    changes = row.get('changes', '')
                                    if changes and changes != '':
                                        return ['background-color: #90EE90'] * len(row)
                                    return [''] * len(row)
                                
                                styled_df = after_df[available_cols].style.apply(highlight_changes, axis=1)
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("No schedule data available")
        else:
            st.info("No schedule changes to display")

elif st.session_state.candidates_data:
    st.info("👆 Select a candidate above to see the full solution with cascades")
else:
    st.info("👆 Find candidates first to see solution options")

# --- SIDEBAR: SYSTEM STATUS ---
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
        st.success("✅ Backend connected")
        st.text(f"Locations: {health.get('locations', 'Unknown')}")
        st.text(f"cuOpt: {'Available' if 'cuopt_url' in health else 'Unavailable'}")
        
        if health.get('status') != 'ok':
            st.warning("⚠️ Backend needs data reload")
    except:
        st.error("❌ Backend unavailable")
    
    # Debug info (collapsible)
    with st.expander("Debug Info"):
        st.text(f"API URL: {API}")
        st.text(f"Session state keys: {list(st.session_state.keys())}")
        if st.session_state.candidates_data:
            st.text(f"Candidates loaded: {len(st.session_state.candidates_data.get('candidates', []))}")

# import os, requests, pandas as pd, streamlit as st
# import json

# API = os.getenv("API_BASE_URL", "http://backend:8000")

# st.set_page_config(page_title="Dynamic Trip Rescheduling", layout="wide")
# st.title("Dynamic Trip Rescheduling")

# # Helper function to format minutes to HH:MM
# def format_time(minutes):
#     """Convert minutes since midnight to HH:MM format"""
#     if pd.isna(minutes) or minutes is None:
#         return "—"
#     try:
#         total_minutes = int(minutes)
#         hours = total_minutes // 60
#         mins = total_minutes % 60
#         return f"{hours:02d}:{mins:02d}"
#     except (ValueError, TypeError):
#         return str(minutes)

# # Initialize session state
# if 'selected_candidate' not in st.session_state:
#     st.session_state.selected_candidate = None
# if 'candidates_data' not in st.session_state:
#     st.session_state.candidates_data = None
# if 'trip_info' not in st.session_state:
#     st.session_state.trip_info = None

# # --- SECTION 1: INPUTS ---
# st.header("1. Trip Details")

# # Fetch location metadata
# try:
#     locs_resp = requests.get(f"{API}/plan/locations", timeout=10).json()
#     locations = locs_resp.get("locations", [])
#     locations = sorted(locations, key=lambda x: x["name"])
#     names = [loc["name"] for loc in locations]
#     name_to_postcode = {loc["name"]: loc.get("postcode", "N/A") for loc in locations}
# except Exception as e:
#     st.error(f"Failed to load locations: {e}")
#     names = []
#     name_to_postcode = {}

# # Location inputs
# col1, col2 = st.columns(2)
# with col1:
#     start = st.selectbox(
#         "Start location", 
#         names, 
#         key="start_location",
#         help="Type to search locations"
#     )
#     if start:
#         st.text(f"📍 {name_to_postcode.get(start, 'No postcode')}")

# with col2:
#     end = st.selectbox(
#         "End location", 
#         names, 
#         key="end_location",
#         help="Type to search locations"
#     )
#     if end:
#         st.text(f"📍 {name_to_postcode.get(end, 'No postcode')}")

# # Trip parameters
# col3, col4, col5 = st.columns(3)
# with col3:
#     mode = st.selectbox("Mode", ["depart_after", "arrive_before"], key="mode")
# with col4:
#     dt = st.text_input("When (Europe/London)", "2025-09-02T10:30", key="datetime")
# with col5:
#     priority = st.slider("Priority (1=highest, 5=lowest)", 1, 5, 3, key="priority")

# # Advanced options (collapsible)
# with st.expander("Advanced Options"):
#     topn = st.slider("Max candidates to show", 5, 50, 20, key="topn")

# # --- SECTION 2: FIND CANDIDATES ---
# st.header("2. Find Candidates")

# if st.button("Find Candidates", disabled=not (start and end), type="primary", key="find_candidates"):
#     payload = {
#         "start_location": start,
#         "end_location": end,
#         "mode": mode,
#         "when_local": dt,
#         "priority": priority,
#         "top_n": topn
#     }
    
#     with st.spinner("Searching for candidates..."):
#         try:
#             r = requests.post(f"{API}/plan/candidates", json=payload, timeout=30)
#             r.raise_for_status()
#             data = r.json()
            
#             # Store in session state
#             st.session_state.candidates_data = data
#             st.session_state.trip_info = {
#                 "trip_minutes": data['trip_minutes'],
#                 "trip_miles": data['trip_miles'],
#                 "weekday": data['weekday']
#             }
#             st.session_state.selected_candidate = None  # Reset selection
            
#             st.success(f"Found {len(data.get('candidates', []))} candidates")
            
#         except Exception as e:
#             st.error(f"Request failed: {e}")
#             st.session_state.candidates_data = None

# # Display candidates if available
# if st.session_state.candidates_data:
#     data = st.session_state.candidates_data
#     trip_info = st.session_state.trip_info
    
#     st.info(f"Trip: {trip_info['trip_minutes']:.1f} mins, {trip_info['trip_miles']:.1f} miles, {trip_info['weekday']}")
    
#     candidates = data.get("candidates", [])
#     if candidates:
#         st.subheader("Available Candidates")
        
#         # Create a more user-friendly display
#         for i, candidate in enumerate(candidates):
#             candidate_summary = f"{candidate.get('driver_id', 'N/A')} - "
            
#             # Parse candidate type
#             cid = candidate.get('candidate_id', '')
#             if 'take_empty' in cid:
#                 candidate_summary += f"Take empty slot (Cost: £{candidate.get('est_cost', 0):.2f})"
#             elif 'swap_leg' in cid:
#                 candidate_summary += f"Swap existing leg (Cost: £{candidate.get('est_cost', 0):.2f})"
#             elif 'append' in cid:
#                 candidate_summary += f"Add to end of duty (Cost: £{candidate.get('est_cost', 0):.2f}, +{candidate.get('overtime_minutes', 0):.0f}min overtime)"
#             elif 'slack' in cid:
#                 candidate_summary += f"Use slack time (Cost: £{candidate.get('est_cost', 0):.2f})"
#             else:
#                 candidate_summary += f"Other solution (Cost: £{candidate.get('est_cost', 0):.2f})"
            
#             # Add feasibility indicator
#             if not candidate.get('feasible_hard', True):
#                 candidate_summary += " ⚠️ Not feasible"
            
#             # Create button for selection
#             if st.button(
#                 candidate_summary, 
#                 key=f"candidate_{i}",
#                 help=f"Deadhead: {candidate.get('deadhead_miles', 0):.1f} miles, Delay: {candidate.get('delay_minutes', 0):.0f} mins"
#             ):
#                 st.session_state.selected_candidate = candidate
#                 st.rerun()
#     else:
#         st.warning("No candidates found. Try adjusting the priority or time window.")

# # --- SECTION 3: SOLUTION PANEL ---
# st.header("3. Solution Details")

# if st.session_state.selected_candidate:
#     candidate = st.session_state.selected_candidate
    
#     st.success(f"Selected: Driver {candidate.get('driver_id')} - {candidate.get('candidate_id')}")
    
#     # Show candidate details
#     col_details1, col_details2, col_details3 = st.columns(3)
#     with col_details1:
#         st.metric("Cost", f"£{candidate.get('est_cost', 0):.2f}")
#     with col_details2:
#         st.metric("Deadhead Miles", f"{candidate.get('deadhead_miles', 0):.1f}")
#     with col_details3:
#         st.metric("Overtime Minutes", f"{candidate.get('overtime_minutes', 0):.0f}")
    
#     # Get full solution with cascades
#     if st.button("Show Full Solution (with cascades)", type="primary", key=f"solve_{candidate.get('driver_id', 'unknown')}"):
#         st.write("DEBUG: Button was clicked!")  
#         cascade_payload = {
#             "start_location": start,
#             "end_location": end,
#             "mode": mode,
#             "when_local": dt,
#             "priority": priority,
#             "max_cascades": 2,
#             "max_drivers_affected": 5,
#             "force_candidate": candidate.get('candidate_id')  # Force this specific candidate
#         }
        
#         with st.spinner("Computing cascades..."):
#             try:
#                 r = requests.post(f"{API}/plan/solve_cascades", json=cascade_payload, timeout=60)
#                 r.raise_for_status()
#                 solution_data = r.json()
                
#                 st.success(f"Total Solution Cost: £{solution_data.get('objective_value', 0):.2f}")
                
#                 # Show assignments
#                 assignments = solution_data.get("assignments", [])
#                 if assignments:
#                     st.subheader("Trip Assignments")
#                     assign_df = pd.DataFrame(assignments)
#                     # Clean up for display
#                     display_cols = ['trip_id', 'driver_id', 'type', 'cost', 'delay_minutes', 'overtime_minutes']
#                     available_cols = [col for col in display_cols if col in assign_df.columns]
#                     st.dataframe(assign_df[available_cols], use_container_width=True)
                
#                 # Show cascades if any
#                 cascades = solution_data.get("cascades", [])
#                 if cascades:
#                     st.subheader("Cascade Effects")
#                     st.info(f"Found {len(cascades)} displaced trips that need reassignment")
#                     casc_df = pd.DataFrame(cascades)
#                     st.dataframe(casc_df, use_container_width=True)
                
#                 # Get driver schedules directly from solve_cascades response
#                 schedules_from_cascades = solution_data.get("schedules", [])
                
#                 if schedules_from_cascades:
#                     st.subheader("Driver Schedule Changes")
                    
#                     for sch in schedules_from_cascades:
#                         driver_id = sch.get('driver_id')
                        
#                         with st.expander(f"Driver {driver_id} Schedule", expanded=True):
#                             col_before, col_after = st.columns(2)
                            
#                             with col_before:
#                                 st.markdown("**Before:**")
#                                 before_data = sch.get("before", [])
#                                 if before_data:
#                                     before_df = pd.DataFrame(before_data)
                                    
#                                     # Format time columns if they exist
#                                     if 'start_min' in before_df.columns:
#                                         before_df['start_time'] = before_df['start_min'].apply(format_time)
#                                     if 'end_min' in before_df.columns:
#                                         before_df['end_time'] = before_df['end_min'].apply(format_time)
                                    
#                                     # Show relevant columns
#                                     display_cols = []
#                                     for col in ['element_type', 'from', 'to', 'start_time', 'end_time', 'priority']:
#                                         if col in before_df.columns:
#                                             display_cols.append(col)
                                    
#                                     if display_cols:
#                                         st.dataframe(before_df[display_cols], use_container_width=True)
#                                     else:
#                                         st.dataframe(before_df.head(10), use_container_width=True)
#                                 else:
#                                     st.write("No scheduled activities")
                            
#                             with col_after:
#                                 st.markdown("**After:**")
#                                 after_data = sch.get("after", [])
#                                 if after_data:
#                                     after_df = pd.DataFrame(after_data)
                                    
#                                     # Format time columns if they exist
#                                     if 'start_min' in after_df.columns:
#                                         after_df['start_time'] = after_df['start_min'].apply(format_time)
#                                     if 'end_min' in after_df.columns:
#                                         after_df['end_time'] = after_df['end_min'].apply(format_time)
                                    
#                                     # Add load type display
#                                     if 'load_type' in after_df.columns:
#                                         after_df['load_type_display'] = after_df['load_type'].fillna('UNKNOWN')
#                                     elif 'planz_code' in after_df.columns:
#                                         after_df['load_type_display'] = after_df['planz_code'].fillna('UNKNOWN')
#                                     else:
#                                         after_df['load_type_display'] = 'UNKNOWN'
                                    
#                                     # Show priority
#                                     if 'priority' in after_df.columns:
#                                         after_df['priority_display'] = after_df['priority'].fillna(3).astype(int)
#                                     else:
#                                         after_df['priority_display'] = 3
                                    
#                                     # Mark changes
#                                     if 'note' in after_df.columns:
#                                         after_df['changes'] = after_df['note'].fillna('')
#                                     else:
#                                         after_df['changes'] = ''
                                    
#                                     # Display columns
#                                     col_mapping = {
#                                         'element_type': 'Type',
#                                         'from': 'From', 
#                                         'to': 'To', 
#                                         'start_time': 'Start', 
#                                         'end_time': 'End',
#                                         'load_type_display': 'Load Type',
#                                         'priority_display': 'Priority',
#                                         'changes': 'Changes'
#                                     }
                                    
#                                     display_cols = []
#                                     for col in col_mapping.keys():
#                                         if col in after_df.columns:
#                                             display_cols.append(col)
                                    
#                                     if display_cols:
#                                         display_df = after_df[display_cols].rename(columns=col_mapping)
#                                         st.dataframe(display_df.head(10), use_container_width=True)
#                                     else:
#                                         st.dataframe(after_df.head(10), use_container_width=True)
#                                 else:
#                                     st.write("No scheduled activities")
#                 else:
#                     # Fallback: if solve_cascades doesn't return schedules, show assignment info
#                     st.subheader("Affected Drivers")
#                     affected_drivers = set()
#                     for assignment in assignments:
#                         if assignment.get('driver_id'):
#                             affected_drivers.add(assignment['driver_id'])
                    
#                     if affected_drivers:
#                         for driver_id in affected_drivers:
#                             st.info(f"Driver {driver_id} - schedule changes applied (detailed view not available)")
#                     else:
#                         st.warning("No driver assignments found")
                
#             except Exception as e:
#                 st.error(f"Failed to get full solution: {e}")
#                 st.text("Try selecting a different candidate or check backend connectivity")

# elif st.session_state.candidates_data:
#     st.info("👆 Select a candidate above to see the full solution with cascades")
# else:
#     st.info("👆 Find candidates first to see solution options")

# # --- SIDEBAR: SYSTEM STATUS ---
# with st.sidebar:
#     st.header("System Status")
#     try:
#         health = requests.get(f"{API}/health", timeout=5).json()
#         st.success("✅ Backend connected")
#         st.text(f"Locations: {health.get('locations', 'Unknown')}")
#         st.text(f"cuOpt: {'Available' if 'cuopt_url' in health else 'Unavailable'}")
        
#         if health.get('status') != 'ok':
#             st.warning("⚠️ Backend needs data reload")
#     except:
#         st.error("❌ Backend unavailable")
    
#     # Debug info (collapsible)
#     with st.expander("Debug Info"):
#         st.text(f"API URL: {API}")
#         st.text(f"Session state keys: {list(st.session_state.keys())}")
#         if st.session_state.candidates_data:
#             st.text(f"Candidates loaded: {len(st.session_state.candidates_data.get('candidates', []))}")