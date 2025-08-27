import os, io, json
from datetime import date, time as dtime
from typing import Dict, List, Tuple

import requests
import pandas as pd
import streamlit as st

API_BASE_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# -------------------------- HTTP helper --------------------------
def api(method: str, path: str, **kwargs):
    url = f"{st.session_state['api_base']}{path}"
    r = requests.request(method, url, timeout=60, **kwargs)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return {"_raw": r.text}

# -------------------------- Data fetchers --------------------------
def fetch_site_names() -> List[str]:
    """
    Try several likely endpoints and parse various shapes to get site names.
    Falls back to a minimal list if nothing works.
    """
    probe_paths = [
        "/locations?dataset_id=active",
        "/locations",
        "/plan/locations",
        "/admin/locations",
        "/config",
    ]
    for p in probe_paths:
        try:
            data = api("GET", p)
        except Exception:
            continue

        # Accept various shapes
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            names = [x.strip() for x in data if x and isinstance(x, str)]
            if names:
                return sorted(set(names))

        if isinstance(data, dict):
            if "names" in data and isinstance(data["names"], list):
                names = [str(x).strip() for x in data["names"] if x]
                if names:
                    return sorted(set(names))
            if "location_to_index" in data and isinstance(data["location_to_index"], dict):
                names = [str(k).strip() for k in data["location_to_index"].keys()]
                if names:
                    return sorted(set(names))
            if "location_index" in data:
                li = data["location_index"]
                if isinstance(li, list) and li and isinstance(li[0], dict) and "name" in li[0]:
                    names = [str(x["name"]).strip() for x in li if x.get("name")]
                    if names:
                        return sorted(set(names))

    # Fallback: keep your current defaults available, but show a warning in UI.
    return sorted(set(["BIRMINGHAM MAIL CENTRE", "MIDLANDS SUPER HUB"]))

def fetch_priority_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get a mapping of label -> numeric priority from backend; otherwise use a reasonable default.
    Returns: (label_to_num, num_to_label)
    """
    probe_paths = ["/plan/priority_map", "/priority_map", "/admin/priority_map"]
    for p in probe_paths:
        try:
            data = api("GET", p)
        except Exception:
            continue

        # Accept shapes like {"RM TRACKED 24": 1, ...}
        if isinstance(data, dict) and data:
            label_to_num = {}
            for k, v in data.items():
                try:
                    label = str(k).strip()
                    num = int(v)
                    if label:
                        label_to_num[label] = num
                except Exception:
                    continue
            if label_to_num:
                num_to_label = {v: k for k, v in label_to_num.items()}
                return label_to_num, num_to_label

        # Accept shapes like {"1":"RM TRACKED 24", ...}
        if isinstance(data, dict) and all(str(k).isdigit() for k in data.keys()):
            num_to_label = {int(k): str(v).strip() for k, v in data.items() if str(v).strip()}
            label_to_num = {v: k for k, v in num_to_label.items()}
            if label_to_num:
                return label_to_num, num_to_label

    # Default mapping (override when backend exposes the real one)
    default_label_to_num = {
        "RM Tracked 24": 1,
        "RM 24": 2,
        "RM 48": 3,
        "Economy": 4,
        "Empty": 5,  # use this for empty legs if you have it
    }
    default_num_to_label = {v: k for k, v in default_label_to_num.items()}
    return default_label_to_num, default_num_to_label

# -------------------------- UI setup --------------------------
st.set_page_config(page_title="Dynamic Rescheduling Demo", layout="wide")
st.title("Dynamic Trip Rescheduling — Demo")

with st.sidebar:
    st.header("Backend")
    if "api_base" not in st.session_state:
        st.session_state["api_base"] = API_BASE_DEFAULT
    st.session_state["api_base"] = st.text_input(
        "API base URL",
        value=st.session_state["api_base"]
    ).rstrip("/")
    st.caption("Example: http://localhost:8000  or  http://api:8000 (inside Docker)")
    if st.button("Check API health"):
        try:
            st.json(api("GET", "/health"))
        except Exception as e:
            st.error(f"Health check failed: {e}")

# Cache site list & priority maps in session
if "site_names" not in st.session_state:
    st.session_state["site_names"] = fetch_site_names()
if "priority_maps" not in st.session_state:
    st.session_state["priority_maps"] = fetch_priority_maps()

site_names = st.session_state["site_names"]
label_to_num, num_to_label = st.session_state["priority_maps"]

if site_names == ["BIRMINGHAM MAIL CENTRE", "MIDLANDS SUPER HUB"]:
    st.warning("Could not load full site list from backend. Using a minimal list. Check your backend for a locations endpoint.")

st.divider()

# -------------------------- Disruption Manager --------------------------
st.header("Disruption Manager")

# --- Start / End with dropdowns + Not-in-RSL toggles ---
colA, colB = st.columns(2)
with colA:
    st.subheader("Start")
    start_not_in_rsl = st.checkbox("Not in RSL (use postcode instead)", value=False, key="start_not_in_rsl")
    if start_not_in_rsl:
        start_postcode = st.text_input("Start postcode (UK)", value="", placeholder="e.g., B1 1AA", key="start_postcode")
        # we still let you select a name (optional) for convenience; not sent to backend
        start_location = st.selectbox("Start location (optional)", options=site_names, index=site_names.index("BIRMINGHAM MAIL CENTRE") if "BIRMINGHAM MAIL CENTRE" in site_names else 0, key="start_location")
    else:
        start_location = st.selectbox("Start location", options=site_names, index=site_names.index("BIRMINGHAM MAIL CENTRE") if "BIRMINGHAM MAIL CENTRE" in site_names else 0, key="start_location")

with colB:
    st.subheader("End")
    end_not_in_rsl = st.checkbox("Not in RSL (use postcode instead)", value=False, key="end_not_in_rsl")
    if end_not_in_rsl:
        end_postcode = st.text_input("End postcode (UK)", value="", placeholder="e.g., B40 1NT", key="end_postcode")
        end_location = st.selectbox("End location (optional)", options=site_names, index=site_names.index("MIDLANDS SUPER HUB") if "MIDLANDS SUPER HUB" in site_names else min(1, len(site_names)-1), key="end_location")
    else:
        end_location = st.selectbox("End location", options=site_names, index=site_names.index("MIDLANDS SUPER HUB") if "MIDLANDS SUPER HUB" in site_names else min(1, len(site_names)-1), key="end_location")

# --- Time / Mode / Priority (label-based) ---
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1, 1])
with col1:
    from datetime import date as _date, time as _time
    d = st.date_input("Date (Europe/London)", value=_date(2025, 8, 18))
with col2:
    t = st.time_input("Time (24h)", value=_time(21, 30))
with col3:
    mode = st.radio("Constraint", options=["depart_after", "arrive_before"], horizontal=True, index=0)
with col4:
    pr_label = st.selectbox("Priority", options=list(label_to_num.keys()), index=0)
    priority = label_to_num[pr_label]

when_local = f"{d.isoformat()}T{t.strftime('%H:%M')}"

# Prepare payload strictly as backend expects (no postcode fields!)
def make_payload(top_n: int) -> Dict:
    return {
        "start_location": start_location.strip(),
        "end_location": end_location.strip(),
        "mode": mode,
        "when_local": when_local,
        "priority": int(priority),
        "top_n": int(top_n),
    }

# Guard: if user ticked “Not in RSL”, postcode must be provided (client-side for now)
def postcode_ready() -> bool:
    if st.session_state.get("start_not_in_rsl") and not st.session_state.get("start_postcode"):
        st.error("Start is marked 'Not in RSL' but postcode is empty.")
        return False
    if st.session_state.get("end_not_in_rsl") and not st.session_state.get("end_postcode"):
        st.error("End is marked 'Not in RSL' but postcode is empty.")
        return False
    return True

st.caption("Postcodes are captured client-side only. Calls to the backend still use site names to preserve current API contracts.")

# -------------------------- Candidate Search --------------------------
st.subheader("1) Find candidates")
c1, c2 = st.columns([1, 3])
with c1:
    top_n = st.slider("Top candidates", min_value=5, max_value=100, value=20, step=5)
if st.button("🔎 Search"):
    if postcode_ready():
        try:
            res = api("POST", "/plan/candidates", json=make_payload(top_n))
            st.session_state["cand_res"] = res
            st.success("Candidates loaded.")
        except Exception as e:
            st.error(f"Candidate search failed: {e}")

cand_res = st.session_state.get("cand_res")
if cand_res:
    df = pd.DataFrame(cand_res.get("candidates", []))
    st.caption(f"Trip: {cand_res.get('weekday','?')} — {cand_res.get('trip_minutes','?')} min — {cand_res.get('trip_miles','?')} miles — Priority: {pr_label} ({priority})")
    if not df.empty:
        preferred = ["candidate_id","driver_id","type","feasible_hard","est_cost",
                     "deadhead_miles","deadhead_minutes","overtime_minutes","miles_delta","delay_minutes","route_id"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download candidates (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="candidates.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No candidates found.")

st.divider()

# -------------------------- Solve (Cascades) --------------------------
st.subheader("2) Solve (with cascades)")
sc1, sc2 = st.columns(2)
with sc1:
    max_cascades = st.slider("Max cascades", min_value=0, max_value=5, value=2)
with sc2:
    max_drivers_affected = st.slider("Max drivers affected", min_value=1, max_value=50, value=5)

if st.button("🧩 Run solve (cascades)"):
    if postcode_ready():
        solve_payload = make_payload(top_n)  # top_n not used by solve; harmless if present
        solve_payload["max_cascades"] = int(max_cascades)
        solve_payload["max_drivers_affected"] = int(max_drivers_affected)
        try:
            res = api("POST", "/plan/solve_cascades", json=solve_payload)
            st.session_state["solve_res"] = res
            st.success("Solve complete.")
        except Exception as e:
            st.error(f"Solve failed: {e}")

solve_res = st.session_state.get("solve_res")
if solve_res:
    st.write(f"Objective value: **{solve_res.get('objective_value', 0)}**")
    st.write(f"Candidates considered: {solve_res.get('candidates_considered', 0)}")

    assignments = pd.DataFrame(solve_res.get("assignments", []))
    cascades = pd.DataFrame(solve_res.get("cascades", []))
    if not assignments.empty:
        if "cost_breakdown" in assignments.columns:
            cb = pd.json_normalize(assignments["cost_breakdown"]).fillna(0.0)
            cb.columns = [f"cost_{c}" for c in cb.columns]
            assignments = pd.concat([assignments.drop(columns=["cost_breakdown"]), cb], axis=1)

        st.subheader("Assignments")
        st.dataframe(assignments, use_container_width=True, hide_index=True)

        # Per-assignment detail expanders
        for i, row in assignments.iterrows():
            title = f"{row['type'].upper()} — {row.get('trip_id','N/A')} — driver {row.get('driver_id') or 'N/A'} — cost {row.get('cost')}"
            with st.expander(title):
                st.json({k: (v if not isinstance(v, (pd.Series, pd.DataFrame)) else v.to_dict()) for k, v in row.items()})

        # XLSX download (Assignments + Cascades)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            assignments.to_excel(xw, sheet_name="Assignments", index=False)
            if not cascades.empty:
                cascades.to_excel(xw, sheet_name="Cascades", index=False)

        st.download_button(
            "⬇️ Download solution (XLSX)",
            data=buf.getvalue(),
            file_name="solution.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No assignments returned.")

st.divider()

# -------------------------- Optional upload (unchanged) --------------------------
st.header("Optional: upload private data to backend")
uploaded = st.file_uploader("Upload CSV/NPZ to backend (/upload)", type=["csv","npz"], accept_multiple_files=False)
if uploaded is not None:
    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
        resp = api("POST", "/upload", files=files)
        st.write("Upload status:")
        st.json(resp)
    except Exception as e:
        st.error(f"Upload failed (is /upload enabled on backend?): {e}")
