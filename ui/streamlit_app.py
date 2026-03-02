import os
import json
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://backend:8000")
CASCADE_SOLVE_TIMEOUT_SEC = int(os.getenv("CASCADE_SOLVE_TIMEOUT_SEC", "180"))

st.set_page_config(page_title="Dynamic Trip Rescheduling", layout="wide")
st.title("Dynamic Trip Rescheduling")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().upper()


def _extract_candidate_schedules(candidate: Dict[str, Any], candidates_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected_candidate_id = _normalize_token(candidate.get("candidate_id"))
    selected_driver_id = _normalize_token(candidate.get("driver_id"))
    preview_schedules = ((candidates_data or {}).get("schedules", []) or [])

    schedules: List[Dict[str, Any]] = []
    if selected_candidate_id:
        schedules = [
            s for s in preview_schedules
            if _normalize_token(s.get("candidate_id")) == selected_candidate_id
        ]

    if not schedules and selected_driver_id:
        schedules = [
            s for s in preview_schedules
            if _normalize_token(s.get("driver_id")) == selected_driver_id
        ]

    return schedules


def _build_selected_option_detail(
    candidate: Dict[str, Any],
    candidates_data: Dict[str, Any],
    trip_info: Dict[str, Any],
) -> Dict[str, Any]:
    schedules = _extract_candidate_schedules(candidate, candidates_data)

    parsed_reason = parse_reason_detail(candidate.get("reason_detail"))
    uncovered_p4_total = int(parsed_reason.get("uncovered_p4", 0) or 0)
    disposed_p5_total = int(parsed_reason.get("disposed_p5", 0) or 0)
    unresolved_total = int(uncovered_p4_total + disposed_p5_total)

    cascades = [
        {
            "depth": int(parsed_reason.get("chain_depth", 0) or 0),
            "displaced_by": "NEW_SERVICE",
            "driver_id": candidate.get("driver_id"),
            "from": trip_info.get("start"),
            "to": trip_info.get("end"),
            "priority": int(trip_info.get("priority", 3) or 3),
            "reason": candidate.get("reason") or "Candidate option",
            "reason_code": candidate.get("reason_code"),
            "reason_detail": candidate.get("reason_detail"),
            "assigned_steps": int(parsed_reason.get("assigned_steps", 0) or 0),
            "blocked_steps": int(parsed_reason.get("blocked_steps", 0) or 0),
        }
    ]

    return {
        "schedules": schedules,
        "cascades": cascades,
        "details": {
            "backend": "candidates-single-pass",
            "cascade_diagnostics": {
                "candidates_total": 1,
                "feasible_hard_count": int(1 if bool(candidate.get("feasible_hard", True)) else 0),
                "max_chain_depth": int(parsed_reason.get("chain_depth", 0) or 0),
                "avg_chain_depth": float(parsed_reason.get("chain_depth", 0) or 0),
                "unresolved_total": unresolved_total,
                "uncovered_p4_total": uncovered_p4_total,
                "disposed_p5_total": disposed_p5_total,
                "reason_code_counts": {
                    str(candidate.get("reason_code") or "UNKNOWN"): 1,
                },
            },
        },
    }


def _candidate_cache_key(candidate: Dict[str, Any]) -> str:
    return f"{candidate.get('candidate_id', '')}::{candidate.get('driver_id', '')}"

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
        try:
            parsed[key] = float(value)
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


def build_location_lookup(locations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for loc in locations:
        name = str(loc.get("name", "")).strip()
        if not name:
            continue
        lookup[_normalize_name(name)] = {
            "name": name,
            "postcode": loc.get("postcode"),
            "lat": _safe_float(loc.get("lat")),
            "lon": _safe_float(loc.get("lon")),
        }
    return lookup


def _coords_for_name(name: Any, location_lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    key = _normalize_name(name)
    loc = location_lookup.get(key)
    if not loc:
        return None
    lat = _safe_float(loc.get("lat"))
    lon = _safe_float(loc.get("lon"))
    if lat is None or lon is None:
        return None
    return {"name": loc.get("name", str(name)), "lat": lat, "lon": lon}


def _build_path_points(elements: List[Dict[str, Any]], location_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for element in elements or []:
        from_name = element.get("from")
        to_name = element.get("to")
        from_loc = _coords_for_name(from_name, location_lookup)
        to_loc = _coords_for_name(to_name, location_lookup)

        if from_loc:
            points.append(from_loc)
        if to_loc:
            if not points or points[-1].get("name") != to_loc.get("name"):
                points.append(to_loc)
    return points


def _is_travel_element(element: Dict[str, Any]) -> bool:
    element_type = str(element.get("element_type", "")).upper().strip()
    return bool(element.get("is_travel", False)) or ("TRAVEL" in element_type)


def _travel_signature(element: Dict[str, Any]) -> str:
    from_key = _normalize_name(element.get("from"))
    to_key = _normalize_name(element.get("to"))
    return f"{from_key}::{to_key}"


def _classify_travel_segments(
    before_elements: List[Dict[str, Any]],
    after_elements: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    before_travel = [e for e in (before_elements or []) if _is_travel_element(e)]
    after_travel = [e for e in (after_elements or []) if _is_travel_element(e)]

    before_counter = Counter(_travel_signature(e) for e in before_travel)
    after_counter = Counter(_travel_signature(e) for e in after_travel)

    unchanged_counter = Counter()
    for signature in before_counter:
        unchanged_counter[signature] = min(before_counter[signature], after_counter.get(signature, 0))

    unchanged_segments: List[Dict[str, Any]] = []
    deprecated_segments: List[Dict[str, Any]] = []
    working_unchanged = Counter(unchanged_counter)

    for elem in before_travel:
        sig = _travel_signature(elem)
        if working_unchanged[sig] > 0:
            unchanged_segments.append(elem)
            working_unchanged[sig] -= 1
        else:
            deprecated_segments.append(elem)

    new_segments: List[Dict[str, Any]] = []
    working_new = Counter(after_counter)
    for sig, cnt in unchanged_counter.items():
        working_new[sig] -= cnt

    for elem in after_travel:
        sig = _travel_signature(elem)
        if working_new[sig] > 0:
            new_segments.append(elem)
            working_new[sig] -= 1

    return {
        "unchanged": unchanged_segments,
        "deprecated": deprecated_segments,
        "new": new_segments,
    }


def _segment_to_points(element: Dict[str, Any], location_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    from_loc = _coords_for_name(element.get("from"), location_lookup)
    to_loc = _coords_for_name(element.get("to"), location_lookup)
    if not from_loc or not to_loc:
        return []
    return [from_loc, to_loc]


def _build_marker_rows(
    before_points: List[Dict[str, Any]],
    after_points: List[Dict[str, Any]],
    start_loc: str,
    end_loc: str,
    location_lookup: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for point in before_points:
        rows.append(
            {
                "name": point["name"],
                "lat": point["lat"],
                "lon": point["lon"],
                "group": "BEFORE",
                "color": [60, 120, 216],
            }
        )

    for point in after_points:
        rows.append(
            {
                "name": point["name"],
                "lat": point["lat"],
                "lon": point["lon"],
                "group": "AFTER",
                "color": [244, 134, 66],
            }
        )

    req_start = _coords_for_name(start_loc, location_lookup)
    req_end = _coords_for_name(end_loc, location_lookup)
    if req_start:
        rows.append(
            {
                "name": f"REQ START: {req_start['name']}",
                "lat": req_start["lat"],
                "lon": req_start["lon"],
                "group": "REQUEST",
                "color": [33, 33, 33],
            }
        )
    if req_end:
        rows.append(
            {
                "name": f"REQ END: {req_end['name']}",
                "lat": req_end["lat"],
                "lon": req_end["lon"],
                "group": "REQUEST",
                "color": [33, 33, 33],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["name", "lat", "lon", "group"])
    return df


def fetch_route_geometry(points: List[Dict[str, Any]], route_mode: str) -> Dict[str, Any]:
    payload_points = [{"lat": float(p["lat"]), "lon": float(p["lon"]), "name": p.get("name")} for p in points]
    cache_key = json.dumps(
        {
            "route_mode": route_mode,
            "points": [{"lat": p["lat"], "lon": p["lon"]} for p in payload_points],
        },
        sort_keys=True,
    )
    cache = st.session_state.get("route_geometry_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    if len(payload_points) < 2:
        result = {
            "source": "straight_fallback",
            "geometry": [[p["lon"], p["lat"]] for p in payload_points],
        }
        cache[cache_key] = result
        st.session_state.route_geometry_cache = cache
        return result

    if route_mode == "straight_only":
        result = {
            "source": "straight_forced",
            "geometry": [[p["lon"], p["lat"]] for p in payload_points],
        }
        cache[cache_key] = result
        st.session_state.route_geometry_cache = cache
        return result

    request_payload = {
        "waypoints": payload_points,
        "profile": "driving",
        "fallback_straight": True,
    }
    try:
        resp = requests.post(f"{API}/plan/route_geometry", json=request_payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except Exception:
        result = {
            "source": "straight_fallback",
            "geometry": [[p["lon"], p["lat"]] for p in payload_points],
        }

    cache[cache_key] = result
    st.session_state.route_geometry_cache = cache
    return result


def _build_segment_path_rows(
    elements: List[Dict[str, Any]],
    location_lookup: Dict[str, Dict[str, Any]],
    route_mode: str,
    color: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for element in elements or []:
        points = _segment_to_points(element, location_lookup)
        if len(points) < 2:
            continue
        geometry = fetch_route_geometry(points, route_mode=route_mode)
        path = geometry.get("geometry") or [[p["lon"], p["lat"]] for p in points]
        if not path:
            continue
        rows.append(
            {
                "path": path,
                "color": color,
                "from": str(element.get("from", "")),
                "to": str(element.get("to", "")),
                "source": geometry.get("source", "unknown"),
            }
        )
    return rows


def _build_departure_bubbles(
    after_elements: List[Dict[str, Any]],
    location_lookup: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen_by_loc: Dict[str, int] = {}

    ordered = sorted(after_elements or [], key=lambda e: int(e.get("start_min", 0) or 0))
    for element in ordered:
        loc = _coords_for_name(element.get("from"), location_lookup)
        if not loc:
            continue

        start_min = element.get("start_min")
        if start_min is None:
            continue

        loc_key = _normalize_name(loc.get("name"))
        slot = seen_by_loc.get(loc_key, 0)
        seen_by_loc[loc_key] = slot + 1

        offset_lat = float(loc["lat"]) + (0.0023 * (slot + 1))
        offset_lon = float(loc["lon"]) + (0.0012 if slot % 2 == 0 else -0.0012)

        rows.append(
            {
                "name": loc["name"],
                "lat": offset_lat,
                "lon": offset_lon,
                "label": format_time(start_min),
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render_map_overlay(
    before_elements: List[Dict[str, Any]],
    after_elements: List[Dict[str, Any]],
    start_loc: str,
    end_loc: str,
    location_lookup: Dict[str, Dict[str, Any]],
    route_mode: str,
    focus_task: Optional[Dict[str, Any]] = None,
    route_view_mode: str = "both",
) -> None:
    route_view_mode = str(route_view_mode or "both").strip().lower()
    if route_view_mode not in {"before", "after", "both"}:
        route_view_mode = "both"

    before_points = _build_path_points(before_elements, location_lookup)
    after_points = _build_path_points(after_elements, location_lookup)
    visible_before_points = before_points if route_view_mode in {"before", "both"} else []
    visible_after_points = after_points if route_view_mode in {"after", "both"} else []
    points_df = _build_marker_rows(visible_before_points, visible_after_points, start_loc, end_loc, location_lookup)

    if points_df.empty:
        st.info("Map unavailable for this cascade tab (missing coordinates for these locations).")
        return

    layers: List[Any] = []

    classified = _classify_travel_segments(before_elements, after_elements)
    unchanged_paths = _build_segment_path_rows(
        classified.get("unchanged", []),
        location_lookup,
        route_mode,
        [128, 0, 128],
    )
    deprecated_paths = _build_segment_path_rows(
        classified.get("deprecated", []),
        location_lookup,
        route_mode,
        [158, 158, 158],
    )
    new_paths = _build_segment_path_rows(
        classified.get("new", []),
        location_lookup,
        route_mode,
        [0, 191, 255],
    )

    if route_view_mode == "before":
        new_paths = []
    elif route_view_mode == "after":
        deprecated_paths = []

    # Fallback: if segment classification yields no drawable paths, still draw before/after paths
    # so operators always see route lines.
    if not unchanged_paths and not deprecated_paths and not new_paths:
        if route_view_mode in {"before", "both"} and len(before_points) >= 2:
            before_geo = fetch_route_geometry(before_points, route_mode=route_mode)
            unchanged_paths = [{"path": before_geo.get("geometry") or [[p["lon"], p["lat"]] for p in before_points], "color": [128, 0, 128]}]
        if route_view_mode in {"after", "both"} and len(after_points) >= 2:
            after_geo = fetch_route_geometry(after_points, route_mode=route_mode)
            new_paths = [{"path": after_geo.get("geometry") or [[p["lon"], p["lat"]] for p in after_points], "color": [0, 191, 255]}]

    focus_from_loc = None
    focus_to_loc = None
    focus_geo: Optional[Dict[str, Any]] = None
    if focus_task:
        focus_from_loc = _coords_for_name(focus_task.get("from"), location_lookup)
        focus_to_loc = _coords_for_name(focus_task.get("to"), location_lookup)
        if focus_from_loc and focus_to_loc:
            focus_geo = fetch_route_geometry([focus_from_loc, focus_to_loc], route_mode=route_mode)

    if unchanged_paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=unchanged_paths,
                get_path="path",
                get_width=120,
                get_color="color",
                width_min_pixels=4,
                pickable=False,
            )
        )
    if deprecated_paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=deprecated_paths,
                get_path="path",
                get_width=100,
                get_color="color",
                width_min_pixels=3,
                pickable=False,
            )
        )
    if new_paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=new_paths,
                get_path="path",
                get_width=120,
                get_color="color",
                width_min_pixels=4,
                pickable=False,
            )
        )
    if focus_geo and focus_geo.get("geometry"):
        focus_path = [{"path": focus_geo.get("geometry"), "color": [255, 140, 0]}]
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=focus_path,
                get_path="path",
                get_width=140,
                get_color="color",
                width_min_pixels=5,
                pickable=False,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=points_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=2400,
            pickable=True,
        )
    )
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=points_df,
            get_position="[lon, lat]",
            get_text="name",
            get_size=12,
            get_color=[30, 30, 30],
            get_alignment_baseline="bottom",
        )
    )

    if focus_from_loc or focus_to_loc:
        focus_rows = []
        if focus_from_loc:
            focus_rows.append(
                {
                    "name": f"FOCUS FROM: {focus_from_loc['name']}",
                    "lat": focus_from_loc["lat"],
                    "lon": focus_from_loc["lon"],
                    "color": [255, 140, 0],
                }
            )
        if focus_to_loc:
            focus_rows.append(
                {
                    "name": f"FOCUS TO: {focus_to_loc['name']}",
                    "lat": focus_to_loc["lat"],
                    "lon": focus_to_loc["lon"],
                    "color": [255, 140, 0],
                }
            )
        focus_df = pd.DataFrame(focus_rows)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=focus_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=3600,
                pickable=True,
            )
        )

    bubble_source = after_elements if route_view_mode in {"after", "both"} else before_elements
    bubbles_df = _build_departure_bubbles(bubble_source, location_lookup)
    if not bubbles_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=bubbles_df,
                get_position="[lon, lat]",
                get_fill_color=[255, 255, 255, 240],
                get_line_color=[30, 30, 30, 220],
                get_line_width=2,
                stroked=True,
                get_radius=2600,
                pickable=False,
            )
        )
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=bubbles_df,
                get_position="[lon, lat]",
                get_text="label",
                get_size=18,
                get_color=[5, 5, 5],
                size_units="pixels",
                size_min_pixels=12,
                get_alignment_baseline="center",
            )
        )

    if focus_from_loc or focus_to_loc:
        focus_lats = [x["lat"] for x in [focus_from_loc, focus_to_loc] if x]
        focus_lons = [x["lon"] for x in [focus_from_loc, focus_to_loc] if x]
        center_lat = float(sum(focus_lats) / len(focus_lats))
        center_lon = float(sum(focus_lons) / len(focus_lons))
        zoom_level = 8
    else:
        center_lat = float(points_df["lat"].mean())
        center_lon = float(points_df["lon"].mean())
        zoom_level = 6

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom_level,
            pitch=0,
        ),
        layers=layers,
        tooltip={"text": "{name}"},
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.markdown("Legend: 🟣 Unchanged | ◻️ Gray Deprecated | 🔵 Cyan New | 🟠 Orange Uncovered Selected | ⚫ Requested endpoints")


def render_duty_table(elements: List[Dict[str, Any]], is_after: bool) -> None:
    if not elements:
        st.write("No schedule data available")
        return

    df = pd.DataFrame(elements)
    if "start_min" in df.columns:
        df["start_time"] = df["start_min"].apply(format_time)
    if "end_min" in df.columns:
        df["end_time"] = df["end_min"].apply(format_time)
    df["movement_kind"] = df.apply(classify_movement, axis=1)

    if is_after:
        display_cols = [
            "element_type",
            "movement_kind",
            "from",
            "to",
            "start_time",
            "end_time",
            "priority",
            "load_type",
            "planz_code",
            "changes",
        ]
    else:
        display_cols = [
            "element_type",
            "movement_kind",
            "from",
            "to",
            "start_time",
            "end_time",
            "priority",
            "load_type",
            "planz_code",
        ]

    available_cols = [col for col in display_cols if col in df.columns]
    if not available_cols:
        st.write("No schedule data available")
        return

    if not is_after:
        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
        return

    def highlight_changes(row):
        changes = row.get("changes", "")
        if changes and changes != "":
            return ["background-color: #214F43; color: #72F2B0; font-weight: 600;"] * len(row)
        return [""] * len(row)

    st.dataframe(df[available_cols].style.apply(highlight_changes, axis=1), use_container_width=True, hide_index=True)


# Initialize session state
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None
if "candidates_data" not in st.session_state:
    st.session_state.candidates_data = None
if "trip_info" not in st.session_state:
    st.session_state.trip_info = None
if "route_geometry_cache" not in st.session_state:
    st.session_state.route_geometry_cache = {}

# --- TOP INPUT PANEL ---
st.header("Inputs")

# Fetch location metadata
try:
    locs_resp = requests.get(f"{API}/plan/locations", timeout=10).json()
    locations = locs_resp.get("locations", [])
    locations = sorted(locations, key=lambda x: x["name"])
    names = [loc["name"] for loc in locations]
    name_to_postcode = {loc["name"]: loc.get("postcode", "N/A") for loc in locations}
    location_lookup = build_location_lookup(locations)
except Exception as e:
    st.error(f"Failed to load locations: {e}")
    names = []
    name_to_postcode = {}
    location_lookup = {}

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
    recovery_minutes = st.slider(
        "Recovery minutes (same driver late allowance)",
        0,
        240,
        0,
        key="recovery_minutes",
        help="Allow displaced downstream work to be recovered by the same driver up to this many minutes late",
    )
    map_routing_mode = st.selectbox(
        "Map routing mode",
        options=["osrm_preferred", "straight_only"],
        index=0,
        key="map_routing_mode",
        help="osrm_preferred uses road geometry via OSRM when available; straight_only draws direct line segments",
    )

if st.button("Find Options", disabled=not (start and end), type="primary", key="find_candidates"):

    payload = {
        "start_location": start,
        "end_location": end,
        "mode": mode,
        "when_local": dt,
        "priority": priority,
        "top_n": topn,
        "geography_radius_miles": geography_radius_miles,
        "home_base_radius_miles": home_base_radius_miles,
        "recovery_minutes": recovery_minutes,
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
                "recovery_minutes": recovery_minutes,
                "map_routing_mode": map_routing_mode,
            }
            st.session_state.selected_candidate = None
            st.session_state.route_geometry_cache = {}
            
            st.success(f"Found {len(data.get('candidates', []))} candidates")
            
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.session_state.candidates_data = None

main_left, main_right = st.columns([1.2, 5.0])

with main_left:
    st.subheader("List of Options")
    candidates = (st.session_state.candidates_data or {}).get("candidates", [])

    if not st.session_state.candidates_data:
        st.info("Run Find Options to load candidate choices.")
    elif not candidates:
        st.warning("No viable options found.")
    else:
        for i, cand in enumerate(candidates):
            selected = (
                st.session_state.selected_candidate is not None
                and st.session_state.selected_candidate.get("candidate_id") == cand.get("candidate_id")
                and st.session_state.selected_candidate.get("driver_id") == cand.get("driver_id")
            )
            label = f"{i + 1}. Driver {cand.get('driver_id')}"
            if st.button(
                label,
                key=f"option_btn_{i}_{cand.get('candidate_id', '')}_{cand.get('driver_id', '')}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state.selected_candidate = cand
                st.rerun()

with main_right:
    st.subheader("Option Detail")

    selected_candidate = st.session_state.selected_candidate
    if not selected_candidate:
        st.info("Select an option from the left panel to view cascade tabs, map overlay, and duty tables.")
    else:
        st.caption(
            f"Selected option: Driver {selected_candidate.get('driver_id')} | Candidate {selected_candidate.get('candidate_id')}"
        )

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Deadhead Miles", f"{selected_candidate.get('deadhead_miles', 0):.1f}")
        with metric_col2:
            st.metric("Overtime Minutes", f"{selected_candidate.get('overtime_minutes', 0):.0f}")

        cache_key = _candidate_cache_key(selected_candidate)
        solution_data = _build_selected_option_detail(
            selected_candidate,
            st.session_state.candidates_data or {},
            st.session_state.trip_info or {},
        )

        if solution_data:
            schedules = solution_data.get("schedules", []) or []
            cascades = solution_data.get("cascades", []) or []
            diagnostics = (solution_data.get("details") or {}).get("cascade_diagnostics") or {}
            unresolved_total = int(diagnostics.get("unresolved_total", 0) or 0)
            uncovered_p4_total = int(diagnostics.get("uncovered_p4_total", 0) or 0)
            disposed_p5_total = int(diagnostics.get("disposed_p5_total", 0) or 0)

            unresolved_rows: List[Dict[str, Any]] = []
            for row in cascades:
                parsed = parse_reason_detail(row.get("reason_detail"))
                p4_tasks = parse_unresolved_task_blob(parsed.get("uncovered_p4_tasks", ""))
                p5_tasks = parse_unresolved_task_blob(parsed.get("disposed_p5_tasks", ""))

                for task in p4_tasks:
                    unresolved_rows.append(
                        {
                            "severity": "UNRESOLVED_P1_P4",
                            "from": task.get("from"),
                            "to": task.get("to"),
                            "priority": task.get("priority"),
                            "reason": task.get("reason"),
                        }
                    )

                for task in p5_tasks:
                    unresolved_rows.append(
                        {
                            "severity": "DISPOSED_P5",
                            "from": task.get("from"),
                            "to": task.get("to"),
                            "priority": task.get("priority"),
                            "reason": task.get("reason"),
                        }
                    )

            deduped: Dict[tuple, Dict[str, Any]] = {}
            for task in unresolved_rows:
                dedupe_key = (
                    str(task.get("severity") or ""),
                    str(task.get("from") or "").strip().upper(),
                    str(task.get("to") or "").strip().upper(),
                    int(task.get("priority") or 99),
                    str(task.get("reason") or "").strip().lower(),
                )
                if dedupe_key not in deduped:
                    deduped[dedupe_key] = task

            severity_rank = {"UNRESOLVED_P1_P4": 0, "DISPOSED_P5": 1}
            unresolved_rows = sorted(
                deduped.values(),
                key=lambda task: (
                    severity_rank.get(str(task.get("severity") or ""), 9),
                    int(task.get("priority") or 99),
                    str(task.get("from") or "").upper(),
                    str(task.get("to") or "").upper(),
                ),
            )

            focus_state_key = f"focus_task::{cache_key}"
            if focus_state_key not in st.session_state:
                st.session_state[focus_state_key] = None

            if schedules:
                tab_labels = [f"Cascade Level {idx}" for idx in range(len(schedules))]
                cascade_tabs = st.tabs(tab_labels)
                for idx, tab in enumerate(cascade_tabs):
                    schedule = schedules[idx]
                    before_data = schedule.get("before", [])
                    after_data = schedule.get("after", [])

                    with tab:
                        st.markdown("### Map Overlay (Before vs After)")
                        view_mode_key = f"route_view_mode::{cache_key}::{idx}"
                        selected_view_mode = st.radio(
                            "Show routes",
                            options=["both", "before", "after"],
                            index=0,
                            horizontal=True,
                            key=view_mode_key,
                            format_func=lambda value: value.title(),
                        )
                        render_map_overlay(
                            before_elements=before_data,
                            after_elements=after_data,
                            start_loc=(st.session_state.trip_info or {}).get("start", ""),
                            end_loc=(st.session_state.trip_info or {}).get("end", ""),
                            location_lookup=location_lookup,
                            route_mode=(st.session_state.trip_info or {}).get("map_routing_mode", "osrm_preferred"),
                            focus_task=st.session_state.get(focus_state_key),
                            route_view_mode=selected_view_mode,
                        )

                        if unresolved_total > 0:
                            st.error(
                                f"Warning: uncovered displaced services remain (total={unresolved_total}, uncovered P1-P4={uncovered_p4_total}, disposed P5={disposed_p5_total})."
                            )

                            if unresolved_rows:
                                st.markdown("#### Uncovered Services (click to highlight on map)")
                                for row_idx, task in enumerate(unresolved_rows):
                                    left_col, right_col = st.columns([6, 1])
                                    with left_col:
                                        st.write(
                                            f"{task.get('severity')} | {task.get('from')} → {task.get('to')} | P{task.get('priority') or '?'} | {task.get('reason')}"
                                        )
                                    with right_col:
                                        if st.button("Highlight", key=f"highlight_{cache_key}_{idx}_{row_idx}"):
                                            st.session_state[focus_state_key] = {
                                                "from": task.get("from"),
                                                "to": task.get("to"),
                                            }
                                            st.rerun()

                                if st.button("Clear Highlight", key=f"clear_highlight_{cache_key}_{idx}"):
                                    st.session_state[focus_state_key] = None
                                    st.rerun()
                        else:
                            st.success("No uncovered services for this selected option.")

                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.markdown("### Option Duty (Before)")
                            render_duty_table(before_data, is_after=False)
                        with col_after:
                            st.markdown("### Option Duty (After)")
                            render_duty_table(after_data, is_after=True)
            else:
                st.info("No cascade schedules were returned for this option.")

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
