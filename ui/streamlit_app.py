import os, requests, pandas as pd, streamlit as st

API = os.getenv("API_BASE_URL", "http://backend:8000")
CASCADE_SOLVE_TIMEOUT_SEC = int(os.getenv("CASCADE_SOLVE_TIMEOUT_SEC", "180"))

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


def classify_movement(element):
    """Classify schedule element as LOADED/EMPTY/NON_TRAVEL/UNKNOWN for UI clarity."""
    if not bool(element.get("is_travel", False)):
        return "NON_TRAVEL"

    load_type = str(element.get("load_type", "")).upper()
    planz_code = str(element.get("planz_code", element.get("Planz Code", ""))).upper()

    if (
        "LOADED" in load_type
        or "DELIVERY" in planz_code
        or "SERVICE" in load_type
    ):
        return "LOADED"

    if (
        "EMPTY" in load_type
        or "DEADHEAD" in load_type
        or "RETURN" in load_type
        or "EMPTY" in planz_code
        or "DEADHEAD" in planz_code
        or bool(element.get("is_empty", False))
        or int(element.get("priority", 3) or 3) == 5
    ):
        return "EMPTY"

    return "TRAVEL_UNKNOWN"


def parse_reason_detail(detail):
    parsed = {}
    for part in str(detail or "").split(";"):
        token = part.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = int(value)
            continue
        except Exception:
            pass
        parsed[key] = value
    return parsed


def parse_unresolved_task_blob(value):
    rows = []
    blob = str(value or "").strip()
    if not blob or blob.lower() == "none":
        return rows

    for item in blob.split("|"):
        token = item.strip()
        if not token:
            continue
        route, _, reason = token.partition(":")
        leg, _, priority_part = route.partition("@")
        task_from, _, task_to = leg.partition(">")

        priority = None
        if priority_part.upper().startswith("P"):
            try:
                priority = int(priority_part[1:])
            except Exception:
                priority = None

        rows.append(
            {
                "from": task_from or "UNKNOWN",
                "to": task_to or "UNKNOWN",
                "priority": priority,
                "reason": reason or "unspecified",
            }
        )
    return rows

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
    geography_radius_miles = st.slider(
        "Pickup geography radius (miles)",
        0,
        80,
        15,
        key="geography_radius_miles",
        help="First-pass distance from pickup location used to shortlist drivers",
    )
    home_base_radius_miles = st.slider(
        "Home-base to delivery radius (miles)",
        0,
        80,
        30,
        key="home_base_radius_miles",
        help="First-pass distance from driver home base to delivery location",
    )

# --- SECTION 2: FIND CANDIDATES ---
st.header("2. Find Candidates")

if st.button("Find Candidates", disabled=not (start and end), type="primary", key="find_candidates"):

    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n": topn,
        "geography_radius_miles": geography_radius_miles,
        "home_base_radius_miles": home_base_radius_miles,
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
                "priority": priority,
                "geography_radius_miles": geography_radius_miles,
                "home_base_radius_miles": home_base_radius_miles,
            }
            st.session_state.selected_candidate = None  # Reset selection
            
            st.success(f"Found {len(data.get('candidates', []))} candidates")
            
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.session_state.candidates_data = None

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
        trip_info = st.session_state.get('trip_info', {})
        solve_payload = {
            "start_location": trip_info.get("start"),
            "end_location": trip_info.get("end"),
            "mode": trip_info.get("mode", "depart_after"),
            "when_local": trip_info.get("datetime"),
            "priority": int(trip_info.get("priority", 3)),
            "geography_radius_miles": float(trip_info.get("geography_radius_miles", 15)),
            "home_base_radius_miles": float(trip_info.get("home_base_radius_miles", 30)),
            "max_cascades": 2,
            "max_drivers_affected": 5,
            "preferred_driver_id": candidate.get("driver_id"),
            "preferred_candidate_id": candidate.get("candidate_id"),
        }

        try:
            solve_resp = requests.post(
                f"{API}/plan/solve_cascades",
                json=solve_payload,
                timeout=CASCADE_SOLVE_TIMEOUT_SEC,
            )
        except requests.exceptions.ReadTimeout:
            solve_resp = requests.post(
                f"{API}/plan/solve_cascades",
                json=solve_payload,
                timeout=CASCADE_SOLVE_TIMEOUT_SEC,
            )

        try:
            solve_resp.raise_for_status()
            st.session_state.solution_data = solve_resp.json()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to build full cascade solution: {e}")

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

        # Explicit unresolved displaced services panel
        diagnostics = (solution_data.get("details") or {}).get("cascade_diagnostics") or {}
        unresolved_total = int(diagnostics.get("unresolved_total", 0) or 0)
        uncovered_p4_total = int(diagnostics.get("uncovered_p4_total", 0) or 0)
        disposed_p5_total = int(diagnostics.get("disposed_p5_total", 0) or 0)

        badge_count = unresolved_total
        st.markdown(
            f"### Unassigned Displaced Services <span style='background-color:#d32f2f;color:white;padding:2px 8px;border-radius:10px;font-size:0.8em;'>{badge_count}</span>",
            unsafe_allow_html=True,
        )
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            st.metric("Total unresolved", unresolved_total)
        with col_u2:
            st.metric("Uncovered P1-P4", uncovered_p4_total)
        with col_u3:
            st.metric("Disposed P5", disposed_p5_total)

        unresolved_rows = []
        for row in cascades:
            parsed = parse_reason_detail(row.get("reason_detail"))
            p4_tasks = parse_unresolved_task_blob(parsed.get("uncovered_p4_tasks", ""))
            p5_tasks = parse_unresolved_task_blob(parsed.get("disposed_p5_tasks", ""))

            for task in p4_tasks:
                unresolved_rows.append(
                    {
                        "driver_id": row.get("driver_id"),
                        "severity": "UNRESOLVED_P1_P4",
                        "from": task.get("from"),
                        "to": task.get("to"),
                        "priority": task.get("priority"),
                        "reason": task.get("reason"),
                        "reason_code": row.get("reason_code"),
                    }
                )

            for task in p5_tasks:
                unresolved_rows.append(
                    {
                        "driver_id": row.get("driver_id"),
                        "severity": "DISPOSED_P5",
                        "from": task.get("from"),
                        "to": task.get("to"),
                        "priority": task.get("priority"),
                        "reason": task.get("reason"),
                        "reason_code": row.get("reason_code"),
                    }
                )

        unresolved_df = pd.DataFrame(unresolved_rows)

        if unresolved_df.empty:
            st.success("No unresolved displaced services in this solution.")
        else:
            st.error("⚠️ Unresolved displaced services detected. Review and manually intervene if needed.")
            detail_cols = [
                "driver_id",
                "severity",
                "from",
                "to",
                "priority",
                "reason",
                "reason_code",
            ]
            detail_cols = [c for c in detail_cols if c in unresolved_df.columns]
            st.dataframe(unresolved_df[detail_cols], use_container_width=True)
        
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
                                before_df['start_time'] = before_df['start_min'].apply(format_time)
                            if 'end_min' in before_df.columns:
                                before_df['end_time'] = before_df['end_min'].apply(format_time)
                            before_df['movement_kind'] = before_df.apply(classify_movement, axis=1)
                            
                            # Show relevant columns
                            display_cols = ['element_type', 'movement_kind', 'from', 'to', 'start_time', 'end_time', 'priority', 'load_type', 'planz_code']
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
                                after_df['start_time'] = after_df['start_min'].apply(format_time)
                            if 'end_min' in after_df.columns:
                                after_df['end_time'] = after_df['end_min'].apply(format_time)
                            after_df['movement_kind'] = after_df.apply(classify_movement, axis=1)
                            
                            # Show columns with changes highlighted
                            display_cols = ['element_type', 'movement_kind', 'from', 'to', 'start_time', 'end_time', 'priority', 'load_type', 'planz_code', 'changes']
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
