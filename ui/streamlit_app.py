
import os, requests, pandas as pd, streamlit as st
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

# --- Fetch location metadata (name + postcode) ---
try:
    locs_resp = requests.get(f"{API}/plan/locations", timeout=10).json()
    locations = locs_resp.get("locations", [])
    # Sort locations alphabetically by name
    locations = sorted(locations, key=lambda x: x["name"])
    names = [loc["name"] for loc in locations]
    name_to_postcode = {loc["name"]: loc.get("postcode", "N/A") for loc in locations}
except Exception as e:
    st.error(f"Failed to load locations: {e}")
    names = []
    name_to_postcode = {}

# --- UI Inputs ---
col1, col2 = st.columns(2)
with col1:
    start = st.selectbox(
        "Start location", 
        names, 
        key="start_location_unique",
        help="Type to search locations"
    )
    st.text_input("Start postcode", name_to_postcode.get(start, ""), disabled=True, key="start_pc_display")

with col2:
    end = st.selectbox(
        "End location", 
        names, 
        key="end_location_unique",
        help="Type to search locations"
    )
    st.text_input("End postcode", name_to_postcode.get(end, ""), disabled=True, key="end_pc_display")

# Trip parameters
col3, col4, col5 = st.columns(3)
with col3:
    mode = st.selectbox("Mode", ["depart_after", "arrive_before"], key="mode_unique")
with col4:
    dt = st.text_input("When (Europe/London)", "2025-09-02T10:30", key="datetime_unique")
with col5:
    priority = st.slider("Priority (1=highest, 5=lowest)", 1, 5, 3, key="priority_unique")

# Solution parameters
st.subheader("Solution Options")
col6, col7, col8 = st.columns(3)
with col6:
    max_solutions = st.slider("Max solutions", 1, 5, 3, key="max_sol_unique")
with col7:
    max_cascades = st.slider("Max cascade depth", 0, 2, 1, key="max_casc_unique")
with col8:
    max_drivers = st.slider("Max drivers affected", 1, 5, 3, key="max_drv_unique")

use_cuopt = st.checkbox("Use cuOpt optimization", value=False, key="cuopt_unique")

# --- Single Solve Button ---
if st.button("Find Solutions", disabled=not (start and end), type="primary", key="solve_unique"):
    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n_per_step": 3,
        "max_cascades": max_cascades,
        "max_drivers_affected": max_drivers,
        "max_solutions": max_solutions,
        "use_cuopt": use_cuopt
    }
    
    with st.spinner("Finding optimal solutions..."):
        try:
            r = requests.post(f"{API}/plan/solve_multi", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            
            st.success(f"Found {len(data.get('solutions', []))} solutions")
            st.caption(f"Trip: {data['trip_minutes']:.1f} mins, {data['trip_miles']:.1f} miles, {data['weekday']}")
            
            sols = data.get("solutions", [])
            if not sols:
                st.warning("No feasible solutions found. Try relaxing constraints.")
            else:
                for i, s in enumerate(sols):
                    cost = s.get('objective_value', 0)
                    drivers_affected = s.get('drivers_touched', 0)
                    
                    with st.expander(f"Solution {i+1}: Cost £{cost:.2f} ({drivers_affected} drivers affected)", expanded=(i==0)):
                        
                        # Show assignments
                        assignments = s.get("assignments", [])
                        if assignments:
                            st.markdown("**Trip Assignments:**")
                            assign_df = pd.DataFrame(assignments)
                            cols_to_drop = ["type", "candidate_id", "cost_breakdown"]
                            assign_df = assign_df.drop(columns=cols_to_drop, errors="ignore")
                            st.dataframe(assign_df, use_container_width=True)
                        
                        # Show cascades
                        cascades = s.get("cascades", [])
                        if cascades:
                            st.markdown("**Cascade Effects:**")
                            casc_df = pd.DataFrame(cascades)
                            st.dataframe(casc_df, use_container_width=True)
                        
                        # Show ONLY affected driver schedules
                        schedules = s.get("schedules", [])
                        if assignments:
                            assigned_drivers = {a.get('driver_id') for a in assignments if a.get('driver_id')}
                            affected_schedules = [sch for sch in schedules if sch.get('driver_id') in assigned_drivers]
                        else:
                            affected_schedules = schedules[:3]  # Limit to first 3 if no specific assignments
                        
                        if affected_schedules:
                            st.markdown("**Affected Driver Schedules:**")
                            
                            for sch in affected_schedules:
                                driver_id = sch.get('driver_id')
                                
                                # Create expandable section for each driver
                                with st.expander(f"Driver {driver_id} Schedule Changes"):
                                    col_before, col_after = st.columns(2)
                                    
                                    with col_before:
                                        st.markdown("**Before:**")
                                        before_data = sch.get("before", [])
                                        if before_data:
                                            # Convert to DataFrame and format times
                                            before_df = pd.DataFrame(before_data)
                                            
                                            # Format time columns if they exist
                                            if 'start_min' in before_df.columns:
                                                before_df['start_time'] = before_df['start_min'].apply(format_time)
                                            if 'end_min' in before_df.columns:
                                                before_df['end_time'] = before_df['end_min'].apply(format_time)
                                            
                                            # Show relevant columns
                                            display_cols = []
                                            for col in ['element_type', 'from', 'to', 'start_time', 'end_time', 'priority']:
                                                if col in before_df.columns:
                                                    display_cols.append(col)
                                            
                                            if display_cols:
                                                st.dataframe(before_df[display_cols], use_container_width=True)
                                            else:
                                                st.dataframe(before_df.head(10), use_container_width=True)
                                        else:
                                            st.write("No scheduled activities")
                                    
                                    with col_after:
                                        st.markdown("**After:**")
                                        after_data = sch.get("after", [])
                                        if after_data:
                                            # Convert to DataFrame and format times
                                            after_df = pd.DataFrame(after_data)
                                            
                                            # Format time columns if they exist
                                            if 'start_min' in after_df.columns:
                                                after_df['start_time'] = after_df['start_min'].apply(format_time)
                                            if 'end_min' in after_df.columns:
                                                after_df['end_time'] = after_df['end_min'].apply(format_time)
                                            
                                            # Add Due To Convey display if available
                                            if 'due_to_convey' in after_df.columns:
                                                after_df['load_type_display'] = after_df['due_to_convey'].fillna('UNKNOWN')
                                            elif 'load_type' in after_df.columns:
                                                after_df['load_type_display'] = after_df['load_type'].fillna('UNKNOWN')
                                            elif 'planz_code' in after_df.columns:
                                                after_df['load_type_display'] = after_df['planz_code'].fillna('UNKNOWN')
                                            else:
                                                after_df['load_type_display'] = 'UNKNOWN'
                                            
                                            # Show priority with Load Type
                                            if 'priority' in after_df.columns:
                                                after_df['priority_display'] = after_df['priority'].fillna(3).astype(int)
                                            else:
                                                after_df['priority_display'] = 3
                                            
                                            # Mark new/changed items
                                            if 'note' in after_df.columns:
                                                after_df['changes'] = after_df['note'].fillna('')
                                            else:
                                                after_df['changes'] = ''
                                            
                                            # Show relevant columns with Load Type and Priority
                                            display_cols = []
                                            col_mapping = {
                                                'element_type': 'Type',
                                                'from': 'From', 
                                                'to': 'To', 
                                                'start_time': 'Start', 
                                                'end_time': 'End',
                                                'load_type_display': 'Load Type',
                                                'priority_display': 'Priority',
                                                'changes': 'Changes'
                                            }
                                            
                                            for col, display_name in col_mapping.items():
                                                if col in after_df.columns:
                                                    display_cols.append(col)
                                            
                                            if display_cols:
                                                display_df = after_df[display_cols].rename(columns=col_mapping)
                                                st.dataframe(display_df.head(10), use_container_width=True)
                                            else:
                                                st.dataframe(after_df.head(10), use_container_width=True)
                                        else:
                                            st.write("No scheduled activities")
                        
                        # Solution details
                        if s.get("details"):
                            with st.expander("Technical Details"):
                                st.json(s["details"])
                        
        except requests.exceptions.Timeout:
            st.error("Request timed out. Try reducing max solutions or cascade depth.")
        except Exception as e:
            st.error(f"Solution failed: {e}")
            st.text("Please check if the backend is running and try again.")

# System status in sidebar
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
        st.success("✅ Backend connected")
        st.text(f"Locations: {health.get('locations', 'Unknown')}")
        st.text(f"cuOpt: {'Available' if 'cuopt_url' in health else 'Unavailable'}")
    except:
        st.error("❌ Backend unavailable")

# mode = st.selectbox("Mode", ["depart_after", "arrive_before"], key="trip_mode")
# dt = st.text_input("When (Europe/London)", "2025-09-02T10:30", key="trip_when")
# priority = st.slider("Priority (1=highest urgency, 5=lowest)", 1, 5, 3, key="priority_slider")
# topn = st.slider("Top N", 1, 50, 10, key="topn_slider")

# --- Submit ---
if st.button("Find candidates", disabled=not (start and end), key="find_candidates_btn1"):
    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n": topn
    }
    r = requests.post(f"{API}/plan/candidates", json=payload, timeout=30)
    if r.status_code != 200:
        st.error(r.text)
    else:
        data = r.json()
        st.caption(f"Trip mins={data['trip_minutes']:.1f}, miles={data['trip_miles']:.1f}, weekday={data['weekday']}")
        df = pd.DataFrame(data.get("candidates", []))
        if not df.empty:
            df = df.drop(columns=["type"], errors="ignore")  # Remove 'type' column
            st.dataframe(df)
        else:
            st.info("No candidates.")

# Other input options
mode = st.selectbox("Mode", ["depart_after", "arrive_before"])
dt = st.text_input("When (Europe/London)", "2025-09-02T10:30")
priority = st.slider("Priority (1=highest urgency, 5=lowest)", 1, 5, 3)
topn = st.slider("Top N candidates", 1, 50, 10)
min_slack = st.slider("Min slack (mins)", 0, 180, 0)

# Submit to backend and show results
if st.button("Find candidates", disabled=not (start and end), key="find_candidates_btn2"):
    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n": topn,
        "min_slack": min_slack,
    }
    try:
        r = requests.post(f"{API}/plan/candidates", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.caption(f"Trip mins={data['trip_minutes']:.1f}, miles={data['trip_miles']:.1f}, weekday={data['weekday']}")

        df = pd.DataFrame(data.get("candidates", []))
        if not df.empty:
            df = df.drop(columns=["type"], errors="ignore")
            st.dataframe(df)
        else:
            st.info("No candidates found.")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.subheader("Solve with multiple options (cascades)")

colA, colB, colC = st.columns(3)
with colA:
    max_solutions = st.slider("Max solutions", 1, 10, 5, key="max_solutions")
with colB:
    top_n_per_step = st.slider("Branch factor (per step)", 1, 5, 3, key="branch_factor")
with colC:
    max_cascades = st.slider("Max cascade depth", 0, 4, 2, key="max_cascades")

use_cuopt = st.checkbox("Refine with cuOpt (if configured)", value=False)

if st.button("Solve (multi)", disabled=not (start and end), key="solve_multi_btn"):
    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n_per_step": top_n_per_step,
        "max_cascades": max_cascades,
        "max_drivers_affected": 3,
        "max_solutions": max_solutions,
        "use_cuopt": use_cuopt
    }
    try:
        r = requests.post(f"{API}/plan/solve_multi", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        st.caption(f"Trip mins={data['trip_minutes']:.1f}, miles={data['trip_miles']:.1f}, weekday={data['weekday']}")
        sols = data.get("solutions", [])
        if not sols:
            st.info("No feasible solutions found.")
        else:
            for s in sols:
                with st.expander(f"#{s['rank']}: cost={s['objective_value']:.2f}, drivers touched={s['drivers_touched']}"):
                    # assignments table
                    assign_df = pd.DataFrame(s.get("assignments", []))
                    if not assign_df.empty:
                        assign_df = assign_df.drop(columns=["type"], errors="ignore")
                        st.markdown("**Assignments**")
                        st.dataframe(assign_df)

                    # cascades table
                    casc_df = pd.DataFrame(s.get("cascades", []))
                    if not casc_df.empty:
                        st.markdown("**Cascades**")
                        st.dataframe(casc_df)

                    # schedules (before/after) per driver
                    schedules = s.get("schedules", [])
                    for sch in schedules:
                        st.markdown(f"**Driver {sch['driver_id']}**")
                        b = pd.DataFrame(sch.get("before", []))
                        a = pd.DataFrame(sch.get("after", []))
                        st.markdown("_Before_")
                        if b.empty:
                            st.write("—")
                        else:
                            st.dataframe(b)
                        st.markdown("_After_")
                        if a.empty:
                            st.write("—")
                        else:
                            st.dataframe(a)
    except Exception as e:
        st.error(f"Solve failed: {e}")

