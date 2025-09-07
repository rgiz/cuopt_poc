import os, requests, pandas as pd, streamlit as st
API = os.getenv("API_BASE_URL", "http://backend:8000")

st.set_page_config(page_title="Dynamic Trip Rescheduling", layout="wide")
st.title("Dynamic Trip Rescheduling")

# --- Fetch location metadata (name + postcode) ---
try:
    locs_resp = requests.get(f"{API}/plan/locations", timeout=10).json()
    locations = locs_resp.get("locations", [])
    names = [loc["name"] for loc in locations]
    name_to_postcode = {loc["name"]: loc.get("postcode", "N/A") for loc in locations}
except Exception as e:
    st.error(f"Failed to load locations: {e}")
    names = []
    name_to_postcode = {}

# --- UI Inputs ---
col1, col2 = st.columns(2)
with col1:
    start = st.selectbox("Start location", names, key="start_location")
    st.text_input("Start postcode", name_to_postcode.get(start, ""), key="start_postcode", disabled=True)

with col2:
    end = st.selectbox("End location", names, key="end_location")
    st.text_input("End postcode", name_to_postcode.get(end, ""), key="end_postcode", disabled=True)

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

