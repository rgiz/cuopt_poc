#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds driver_states.json + distance/time matrices from an RSL 'clean' CSV.

ENHANCED VERSION with OSRM integration for comprehensive UK routing.

Inputs:
  - RSL CSV with columns similar to your sample (case-insensitive match):
      "Duty ID", "Element Type", "Commencement Time", "Ending Time",
      "Mapped Name A", "Mapped Name B", "Leg Mileage",
      (optional) "From Lat", "From Long", "To Lat", "To Long",
      (optional) weekday columns: Mon, Tue, Wed, Thu, Fri, Sat, Sun with 'Y'/'N'
      (optional) "Driver Grade", "Vehicle Type", "Service Type" or "Due To Convey"

  - location_index.csv (canonical names + center_id, optional postcode/lat/lon)
    (auto-extends if new names are found in the RSL)

Outputs (by default to data/private/active/):
  - distance_miles_matrix.npz         (key: "matrix")
  - time_minutes_matrix.npz           (key: "matrix")
  - location_index.csv                (updated if new names discovered)
  - driver_states.json

Usage:
  # Original RSL-only approach:
  python3 scripts/build_dataset_from_rsl.py \
      --rsl data/private/active/df_rsl_clean.csv \
      --loc-index data/private/active/location_index.csv \
      --outdir data/private/active \
      --mph 45 \
      --fill-missing haversine

  # Enhanced OSRM approach:
  python3 scripts/build_dataset_from_rsl.py \
      --rsl data/private/active/df_rsl_clean.csv \
      --loc-index data/private/active/location_index.csv \
      --outdir data/private/active \
      --use-osrm \
      --osrm-url http://localhost:5001
"""

from __future__ import annotations
import argparse, math, re, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import time

import pandas as pd
import numpy as np

from src.priority_derivation import derive_priority

WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_col(df: pd.DataFrame, *cands) -> Optional[str]:
    """Enhanced column detection that handles spaces and case variations"""
    cols = set(df.columns)
    
    # First try exact matches
    for c in cands:
        if c and c in cols:
            return c
    
    # Then try lowercase matches
    for c in cands:
        if c and c.lower() in {col.lower() for col in cols}:
            # Find the actual column name
            for actual_col in cols:
                if actual_col.lower() == c.lower():
                    return actual_col
    
    # Then try matches with spaces converted to underscores
    for c in cands:
        if c:
            # Convert candidate to space format
            space_version = c.replace("_", " ")
            for actual_col in cols:
                if actual_col.lower() == space_version.lower():
                    return actual_col
    
    # Finally, try substring matches
    for c in cands:
        if c:
            for actual_col in cols:
                if c.lower() in actual_col.lower():
                    return actual_col
    
    return None

def hhmmss_to_min(s: str) -> int:
    """'05:06:00' -> minutes since 00:00 (same day)"""
    try:
        h, m, *_ = [int(x) for x in str(s).split(":")]
        return h*60 + m
    except Exception:
        return None  # caller should handle
    
def load_priority_map(path: Path) -> Dict[str, int]:
    """Load priority_map.json: {"LOAD TYPE": int}. Keys matched in UPPERCASE."""
    if not path.exists():
        # sensible defaults if file missing
        return {}
    try:
        mp = json.loads(path.read_text(encoding="utf-8"))
        return {str(k).upper(): int(v) for k, v in mp.items()}
    except Exception as e:
        print(f"[warn] failed to read priority map at {path}: {e}")
        return {}

def median_safe(vals: List[float]) -> float:
    v = [x for x in vals if pd.notna(x)]
    if not v:
        return 0.0
    return float(np.median(v))

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.7613  # Earth radius in miles
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def ensure_loc_index(loc_idx_path: Path) -> pd.DataFrame:
    if not loc_idx_path.exists():
        raise SystemExit(f"location_index.csv not found at {loc_idx_path}")
    li = pd.read_csv(loc_idx_path)
    li = norm_cols(li)
    if "name" not in li.columns or "center_id" not in li.columns:
        raise SystemExit("location_index.csv must have columns: name, center_id")
    # optional columns
    for c in ["postcode","lat","lon"]:
        if c not in li.columns: li[c] = pd.NA
    # normalize
    li["name"] = li["name"].astype(str).str.upper().str.strip()
    li["center_id"] = li["center_id"].astype(int)
    li = li.drop_duplicates(subset=["name"]).sort_values("center_id").reset_index(drop=True)
    return li

def extend_location_index(li: pd.DataFrame, names: List[str], rsl_rows: pd.DataFrame) -> pd.DataFrame:
    """Auto-extend the location index with any RSL names not present yet."""
    existing = set(li["name"])
    add = [n for n in sorted(set(names)) if n not in existing and n != "NO_DATA"]
    if not add:
        return li
    start_id = int(li["center_id"].max()) + 1 if len(li) else 0
    # try capture lat/lon/postcode from RSL if present
    df = rsl_rows.copy()
    df = norm_cols(df)
    for c in ["mapped_name_a","mapped_name_b"]:
        if c not in df.columns:
            df[c] = None
    # assemble new rows
    new_rows = []
    for n in add:
        # search any row where A or B equals this name
        rows = df[(df["mapped_name_a"].astype(str).str.upper()==n) | (df["mapped_name_b"].astype(str).str.upper()==n)]
        lat = lon = None
        pc = None
        # prefer from/to lat/lon when the name matches A or B
        for _, r in rows.iterrows():
            # try A side
            if str(r.get("mapped_name_a","")).upper() == n:
                lat = r.get("from_lat", lat)
                lon = r.get("from_long", lon)
                pc  = r.get("from_postcode", pc)
            # try B side
            if str(r.get("mapped_name_b","")).upper() == n:
                lat = r.get("to_lat", lat)
                lon = r.get("to_long", lon)
                pc  = r.get("to_postcode", pc)
            if pd.notna(lat) and pd.notna(lon): break
        new_rows.append({
            "center_id": start_id, "name": n,
            "postcode": (None if pd.isna(pc) else pc),
            "lat": (None if pd.isna(lat) else lat),
            "lon": (None if pd.isna(lon) else lon),
        })
        start_id += 1
    li2 = pd.concat([li, pd.DataFrame(new_rows)], ignore_index=True)
    li2 = li2.drop_duplicates(subset=["name"]).sort_values("center_id").reset_index(drop=True)
    return li2

# =============================================================================
# OSRM Integration Functions
# =============================================================================

def osrm_health_check(base_url: str) -> bool:
    """Check if OSRM service is available using a simple route request."""
    try:
        # Use a simple route request as health check since OSRM root endpoint returns 400
        test_url = f"{base_url.rstrip('/')}/route/v1/driving/-0.1,51.5;-0.2,51.6"
        response = requests.get(test_url, timeout=5)
        # OSRM returns 200 for valid requests, even if no route found
        return response.status_code == 200
    except Exception:
        return False

def get_osrm_table(coordinates: List[Tuple[float, float]], osrm_url: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get distance and duration matrix from OSRM Table API.
    
    Args:
        coordinates: List of (longitude, latitude) tuples
        osrm_url: OSRM service URL
        
    Returns:
        Tuple of (distance_matrix_miles, duration_matrix_minutes)
    """
    if not coordinates:
        return np.array([]), np.array([])
    
    n = len(coordinates)
    if n == 1:
        return np.array([[0.0]]), np.array([[0.0]])
    
    # Format coordinates for OSRM: "lon1,lat1;lon2,lat2;..."
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
    
    url = f"{osrm_url.rstrip('/')}/table/v1/driving/{coord_str}"
    params = {
        "annotations": "duration,distance"
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM error: {data.get('message', 'Unknown error')}")
        
        # Extract matrices
        distances = np.array(data["distances"], dtype=float)  # meters
        durations = np.array(data["durations"], dtype=float)  # seconds
        
        # Convert units: meters to miles, seconds to minutes
        distance_miles = distances * 0.000621371
        duration_minutes = durations / 60.0
        
        return distance_miles, duration_minutes
        
    except Exception as e:
        print(f"[error] OSRM Table API call failed: {e}")
        raise

def build_matrices_with_osrm(li: pd.DataFrame, osrm_url: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build matrices using OSRM routing service with automatic batching for large datasets.
    
    Args:
        li: Location index DataFrame with columns [center_id, name, lat, lon]
        osrm_url: OSRM service URL
        
    Returns:
        Tuple of (distance_matrix, time_matrix)
    """
    print(f"[info] Building matrices with OSRM from {osrm_url}")
    
    # Prepare coordinate list aligned with center_id order
    li_sorted = li.sort_values("center_id").reset_index(drop=True)
    max_id = int(li_sorted["center_id"].max())
    
    # Create full-size matrices
    K = max_id + 1
    D = np.zeros((K, K), dtype=float)
    T = np.zeros((K, K), dtype=float)
    
    # Extract valid coordinates
    valid_coords = []
    valid_indices = []
    
    for _, row in li_sorted.iterrows():
        center_id = int(row["center_id"])
        lat, lon = row.get("lat"), row.get("lon")
        
        if pd.notna(lat) and pd.notna(lon):
            valid_coords.append((float(lon), float(lat)))  # OSRM expects (lon, lat)
            valid_indices.append(center_id)
    
    if not valid_coords:
        print("[warn] No valid coordinates found, returning zero matrices")
        return D, T
    
    n_coords = len(valid_coords)
    print(f"[info] Processing {n_coords} locations with coordinates")
    
    # Determine batch size based on number of coordinates
    if n_coords <= 5:
        batch_size = n_coords  # Single batch
    else:
        # For large datasets, use very small batches - we know 4 works, 10+ fails
        batch_size = 4  # Conservative batch size that we know works
        print(f"[info] Using batched approach with batch size {batch_size}")
        print(f"[info] This will take approximately {(n_coords // batch_size) * 0.2:.0f} seconds")
    
    # Build matrix in batches
    for i in range(0, n_coords, batch_size):
        end_i = min(i + batch_size, n_coords)
        batch_coords = valid_coords[i:end_i]
        batch_indices = valid_indices[i:end_i]
        
        print(f"[info] Processing batch {i//batch_size + 1}: locations {i}-{end_i-1} ({len(batch_coords)} coords)")
        
        try:
            # Get OSRM matrix for this batch
            batch_dist, batch_time = get_osrm_table(batch_coords, osrm_url)
            
            # Map batch results back to full matrix
            for local_i, global_i in enumerate(batch_indices):
                for local_j, global_j in enumerate(batch_indices):
                    D[global_i, global_j] = batch_dist[local_i, local_j]
                    T[global_i, global_j] = batch_time[local_i, local_j]
            
            # Small delay between batches to be respectful
            if end_i < n_coords:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"[error] Batch {i}-{end_i} failed: {e}")
            print(f"[info] Continuing with remaining batches...")
            continue
    
    # Check coverage
    valid_pairs = np.sum(D > 0) - D.shape[0]  # Exclude diagonal
    total_pairs = len(valid_indices) * (len(valid_indices) - 1)  # Exclude diagonal
    coverage_pct = (valid_pairs / total_pairs * 100) if total_pairs > 0 else 100
    
    print(f"[info] OSRM matrix complete. Shape: {D.shape}")
    print(f"[info] Coverage: {coverage_pct:.1f}% ({valid_pairs}/{total_pairs} valid coordinate pairs)")
    if np.sum(D > 0) > 0:
        print(f"[info] Distance range: {np.min(D[D>0]):.1f} - {np.max(D):.1f} miles")
        print(f"[info] Time range: {np.min(T[T>0]):.1f} - {np.max(T):.1f} minutes")
    
    return D, T

def build_matrices_from_rsl_original(rsl: pd.DataFrame, li: pd.DataFrame, mph: float, fill_missing: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build directional OD matrices from TRAVEL elements (median of observed legs) - ORIGINAL METHOD."""
    li_map = dict(zip(li["name"], li["center_id"]))
    # Extract TRAVEL legs
    et_col = detect_col(rsl, "element_type") or "element_type"
    legs = rsl[rsl[et_col].astype(str).str.upper().str.contains("TRAVEL")].copy()

    # Canonical names
    a_col = detect_col(legs, "mapped_name_a") or "mapped_name_a"
    b_col = detect_col(legs, "mapped_name_b") or "mapped_name_b"
    legs[a_col] = legs[a_col].astype(str).str.upper().str.strip()
    legs[b_col] = legs[b_col].astype(str).str.upper().str.strip()

    # Map to ids (filter unknowns later)
    legs["i"] = legs[a_col].map(li_map)
    legs["j"] = legs[b_col].map(li_map)

    # Duration (minutes) and miles
    dur_col = detect_col(legs, "element_time") or "element_time"
    # parse "HH:MM:SS"
    def dur_to_min(s):
        try:
            h,m, *_ = [int(x) for x in str(s).split(":")]
            return h*60 + m
        except Exception:
            return np.nan

    legs["minutes"] = legs[dur_col].map(dur_to_min)
    miles_col = detect_col(legs, "leg_mileage") or "leg_mileage"
    legs["miles"] = pd.to_numeric(legs[miles_col], errors="coerce")

    # group median by i->j
    K = int(li["center_id"].max()) + 1
    D = np.zeros((K,K), dtype=float)
    T = np.zeros((K,K), dtype=float)

    valid = legs.dropna(subset=["i","j"])
    grp = valid.groupby(["i","j"])
    med = grp.agg({"miles": median_safe, "minutes": median_safe}).reset_index()
    for _, r in med.iterrows():
        i, j = int(r["i"]), int(r["j"])
        if i == j:
            continue
        D[i,j] = float(r["miles"] or 0.0)
        T[i,j] = float(r["minutes"] or 0.0)

    # fill_missing options
    if fill_missing.lower() == "haversine":
        # need lat/lon by center_id
        by_id = li.set_index("center_id")[["lat","lon"]].to_dict(orient="index")
        for i in range(K):
            for j in range(K):
                if i == j: 
                    continue
                if D[i,j] <= 0.0 or T[i,j] <= 0.0:
                    a, b = by_id.get(i), by_id.get(j)
                    if a and b and pd.notna(a["lat"]) and pd.notna(a["lon"]) and pd.notna(b["lat"]) and pd.notna(b["lon"]):
                        miles = haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])
                        D[i,j] = miles
                        T[i,j] = miles / mph * 60.0
    # else: leave zeros (strict mode)

    return D, T

def build_matrices_from_rsl_enhanced(rsl: pd.DataFrame, li: pd.DataFrame, mph: float, 
                                   fill_missing: str, use_osrm: bool = False, 
                                   osrm_url: str = None, osrm_fallback: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced matrix building with OSRM integration.
    
    This function can:
    1. Use OSRM for comprehensive routing (if use_osrm=True)
    2. Fall back to original RSL+haversine approach
    3. Combine both approaches for maximum coverage
    """
    
    if use_osrm and osrm_url:
        # Check OSRM availability
        if not osrm_health_check(osrm_url):
            if not osrm_fallback:
                raise RuntimeError(f"OSRM service unavailable at {osrm_url} and fallback disabled")
            print(f"[warn] OSRM service unavailable at {osrm_url}, falling back to RSL method")
            use_osrm = False
        else:
            print(f"[info] OSRM service available at {osrm_url}")
    
    if use_osrm and osrm_url:
        try:
            # Primary: Use OSRM for comprehensive routing
            D_osrm, T_osrm = build_matrices_with_osrm(li, osrm_url)
            
            # Check coverage
            total_pairs = D_osrm.size - D_osrm.shape[0]  # Exclude diagonal
            osrm_coverage = np.sum(D_osrm > 0) - D_osrm.shape[0]  # Exclude diagonal
            coverage_pct = (osrm_coverage / total_pairs * 100) if total_pairs > 0 else 100
            
            print(f"[info] OSRM coverage: {coverage_pct:.1f}% ({osrm_coverage}/{total_pairs} pairs)")
            
            if coverage_pct >= 90:  # Good coverage
                print("[info] Using OSRM matrices (good coverage)")
                return D_osrm, T_osrm
            elif osrm_fallback:
                print("[info] OSRM coverage low, combining with RSL data")
                # Get RSL matrices for supplementary data
                D_rsl, T_rsl = build_matrices_from_rsl_original(rsl, li, mph, fill_missing)
                
                # Combine: OSRM where available, RSL where not
                D_combined = D_osrm.copy()
                T_combined = T_osrm.copy()
                
                # Fill gaps with RSL data
                osrm_missing = (D_osrm == 0) & (D_rsl > 0)
                D_combined[osrm_missing] = D_rsl[osrm_missing]
                T_combined[osrm_missing] = T_rsl[osrm_missing]
                
                combined_coverage = np.sum(D_combined > 0) - D_combined.shape[0]
                combined_pct = (combined_coverage / total_pairs * 100) if total_pairs > 0 else 100
                print(f"[info] Combined coverage: {combined_pct:.1f}% ({combined_coverage}/{total_pairs} pairs)")
                
                return D_combined, T_combined
            else:
                print("[info] Using OSRM matrices despite low coverage (fallback disabled)")
                return D_osrm, T_osrm
                
        except Exception as e:
            if not osrm_fallback:
                raise RuntimeError(f"OSRM matrix building failed: {e}")
            print(f"[warn] OSRM matrix building failed: {e}, falling back to RSL method")
    
    # Fallback: Original RSL-based method
    print("[info] Using original RSL-based matrix building")
    return build_matrices_from_rsl_original(rsl, li, mph, fill_missing)

# =============================================================================
# Original Functions (unchanged)
# =============================================================================

def compute_duty_windows(rsl: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Per duty_id, build weekday windows from START/END FACILITY times (cross-midnight aware)."""
    df = rsl.copy()
    df = norm_cols(df)

    duty_col = detect_col(df, "duty_id") or "duty_id"
    et_col   = detect_col(df, "element_type") or "element_type"
    comm_col = detect_col(df, "commencement_time") or "commencement_time"
    end_col  = detect_col(df, "ending_time") or "ending_time"

    # Map of weekday -> column name (accept "mon", "monday", etc.)
    def find_weekday_cols(dframe: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str,str] = {}
        cols = list(dframe.columns)
        patterns = {
            "Mon": ("mon", "monday"),
            "Tue": ("tue", "tues", "tuesday"),
            "Wed": ("wed", "weds", "wednesday"),
            "Thu": ("thu", "thur", "thurs", "thursday"),
            "Fri": ("fri", "friday"),
            "Sat": ("sat", "saturday"),
            "Sun": ("sun", "sunday"),
        }
        for wd, pats in patterns.items():
            for c in cols:
                lc = c.lower()
                if any(lc.startswith(p) for p in pats):
                    mapping[wd] = c
                    break
        return mapping

    week_cols = find_weekday_cols(df)

    def val_true(x) -> bool:
        s = str(x).strip().lower()
        return s in {"y", "yes", "true", "1"}

    windows: Dict[str, Dict[str, Any]] = {}

    for duty_id, grp in df.groupby(duty_col, dropna=False):
        # pick first START and END for this duty
        g2 = grp.copy()
        srow = g2[g2[et_col].astype(str).str.upper().str.contains("START FACILITY")].head(1)
        erow = g2[g2[et_col].astype(str).str.upper().str.contains("END FACILITY")].head(1)
        if srow.empty or erow.empty:
            continue

        smin = hhmmss_to_min(str(srow.iloc[0][comm_col]))
        emin = hhmmss_to_min(str(erow.iloc[0][end_col]))
        if smin is None or emin is None:
            continue

        # days_active: if weekday columns exist, OR across all rows in this duty
        if week_cols:
            days_active = {wd: any(val_true(v) for v in g2[week_cols[wd]].values) if wd in week_cols else False
                           for wd in WEEKDAYS}
        else:
            # fallback: assume active all days (will be refined later if needed)
            days_active = {wd: True for wd in WEEKDAYS}

        # cross-midnight if end < start
        crosses = emin < smin

        daily = {}
        for wd in WEEKDAYS:
            if days_active.get(wd, False):
                daily[wd] = {
                    "start_min": int(smin),
                    "end_min": int(emin if not crosses else emin + 24*60),
                    "crosses_midnight": bool(crosses),
                }

        windows[str(duty_id)] = {
            "duty_id": str(duty_id),
            "days": [wd for wd, on in days_active.items() if on],
            "daily_windows": daily,
        }

    return windows

def condense_elements(rsl: pd.DataFrame, li_map: Dict[str,int], priority_map: Dict[str,int]) -> Dict[str, List[Dict[str,Any]]]:
    """Condense schedule elements per duty (for UI/debug export). Tag priorities."""
    df = rsl.copy()
    df = norm_cols(df)

    duty_col = detect_col(df, "duty_id") or "duty_id"
    et_col   = detect_col(df, "element_type") or "element_type"
    comm_col = detect_col(df, "commencement_time") or "commencement_time"
    end_col  = detect_col(df, "ending_time") or "ending_time"
    a_col    = detect_col(df, "mapped_name_a") or "mapped_name_a"
    b_col    = detect_col(df, "mapped_name_b") or "mapped_name_b"
    miles_col= detect_col(df, "leg_mileage") or "leg_mileage"
    svc_col  = detect_col(df, "due_to_convey") or detect_col(df, "service_type") or None

    def time_to_min(s: str) -> Optional[int]:
        try:
            h, m, *_ = [int(x) for x in str(s).split(":")]
            return h*60+m
        except Exception:
            return None

    out: Dict[str, List[Dict[str,Any]]] = {}
    for duty_id, grp in df.groupby(duty_col):
        rows = []
        for _, r in grp.iterrows():
            et = str(r.get(et_col,"")).upper()
            s  = str(r.get(comm_col,""))
            e  = str(r.get(end_col,""))
            a  = str(r.get(a_col,"")).upper().strip()
            b  = str(r.get(b_col,"")).upper().strip()
            miles = r.get(miles_col, None)
            dur = r.get("element_time", None)
            duration_min = None
            try:
                if isinstance(dur, str) and ":" in dur:
                    h,m,*_ = [int(x) for x in dur.split(":")]
                    duration_min = h*60+m
            except Exception:
                pass
            load_type = (str(r.get(svc_col," ")).upper().strip() if svc_col else "NO_DATA")
            prio = derive_priority(None, load_type, priority_map, default=3)
            rows.append({
                "element_type": et,
                "is_travel": bool("TRAVEL" in et),
                "start": s,
                "start_min": time_to_min(s),
                "end": e,
                "end_min": time_to_min(e),
                "from": a,
                "to": b,
                "from_id": li_map.get(a),
                "to_id": li_map.get(b),
                "miles": None if pd.isna(miles) else float(miles),
                "duration_min": duration_min,
                "load_type": load_type,
                "priority": prio,
                "Mon": int(r.get("mon", "0")),
                "Tue": int(r.get("tue", "0")), 
                "Wed": int(r.get("wed", "0")),
                "Thu": int(r.get("thu", "0")),
                "Fri": int(r.get("fri", "0")),
                "Sat": int(r.get("sat", "0")),
                "Sun": int(r.get("sun", "0")),
            })
        out[str(duty_id)] = rows
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsl", required=True, help="Path to df_rsl_clean CSV")
    ap.add_argument("--loc-index", required=True, help="Path to location_index.csv")
    ap.add_argument("--outdir", default="data/private/active", help="Output directory (dataset)")
    ap.add_argument("--mph", type=float, default=45.0, help="Fallback mph for haversine time")
    ap.add_argument("--fill-missing", choices=["haversine","zero"], default="haversine",
                    help="How to fill OD pairs not seen in RSL travel legs")
    
    # OSRM Integration Arguments
    ap.add_argument("--use-osrm", action="store_true", 
                    help="Use OSRM routing service for comprehensive distance/time matrices")
    ap.add_argument("--osrm-url", default="http://localhost:5001",
                    help="OSRM service URL (default: http://localhost:5001)")
    ap.add_argument("--osrm-fallback", action="store_true", default=True,
                    help="Fall back to RSL+haversine if OSRM fails")
    
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    li = ensure_loc_index(Path(args.loc_index))
    rsl = pd.read_csv(args.rsl)
    rsl = norm_cols(rsl)
    prio_path = (Path(args.outdir) / "priority_map.json")
    priority_map = load_priority_map(prio_path)

    # Canonicalize names in RSL
    for col in ["mapped_name_a","mapped_name_b"]:
        c = detect_col(rsl, col) or col
        rsl[c] = rsl[c].astype(str).str.upper().str.strip()

    # Extend location_index with any new names found in A/B
    all_names = pd.concat([
        rsl[detect_col(rsl,"mapped_name_a") or "mapped_name_a"],
        rsl[detect_col(rsl,"mapped_name_b") or "mapped_name_b"],
    ], ignore_index=True).astype(str).str.upper().str.strip().tolist()
    li2 = extend_location_index(li, all_names, rsl)

    # Save updated location_index if changed
    if len(li2) != len(li) or set(li2["name"]) != set(li["name"]):
        # Keep existing center_id; only added rows got new IDs in extend_location_index()
        li2 = li2.drop_duplicates(subset=["name"]).copy()
        li2["center_id"] = li2["center_id"].astype(int)
        li2 = li2.sort_values("center_id").reset_index(drop=True)
        li2 = li2[["center_id","name","postcode","lat","lon"]]
        (outdir/"location_index.csv").parent.mkdir(parents=True, exist_ok=True)
        li2.to_csv(outdir/"location_index.csv", index=False, encoding="utf-8")
        print(f"[info] location_index.csv updated with new sites: {outdir/'location_index.csv'}")
    else:
        # ensure the one in outdir exists with correct column order
        if not (outdir/"location_index.csv").exists():
            li2 = li2[["center_id","name","postcode","lat","lon"]]
            (outdir/"location_index.csv").parent.mkdir(parents=True, exist_ok=True)
            li2.to_csv(outdir/"location_index.csv", index=False, encoding="utf-8")

    # Use the saved/updated index
    li2 = pd.read_csv(outdir/"location_index.csv")
    li2 = norm_cols(li2)
    li2["center_id"] = li2["center_id"].astype(int)
    li_map = dict(zip(li2["name"], li2["center_id"]))

    # Build matrices: ENHANCED VERSION with OSRM integration
    D, T = build_matrices_from_rsl_enhanced(
        rsl, li2, 
        mph=args.mph, 
        fill_missing=args.fill_missing,
        use_osrm=args.use_osrm,
        osrm_url=args.osrm_url if args.use_osrm else None,
        osrm_fallback=args.osrm_fallback
    )
    
    np.savez_compressed(outdir/"distance_miles_matrix.npz", matrix=D)
    np.savez_compressed(outdir/"time_minutes_matrix.npz",  matrix=T)
    print("[ok] wrote matrices:",
          outdir/"distance_miles_matrix.npz",
          outdir/"time_minutes_matrix.npz")

    # Driver duty windows
    windows = compute_duty_windows(rsl)

    # Condensed elements (for UI display / export)
    elements = condense_elements(rsl, li_map, priority_map)

    # Assemble driver_states.json
    drivers = {}
    for duty_id, win in windows.items():
        drivers[duty_id] = {
            "duty_id": duty_id,
            "days": win.get("days", []),
            "daily_windows": win.get("daily_windows", {}),
            "elements": elements.get(duty_id, []),
            # slots for later:
            "weekly_emergency_rest_quota": 2,
            "grade": None,
            "vehicle_type": None,
        }
    # Optional: fill grade / vehicle_type columns if present
    grade_col = detect_col(rsl, "driver_grade")
    vtype_col = detect_col(rsl, "vehicle_type")
    if grade_col or vtype_col:
        meta = rsl.groupby(detect_col(rsl,"duty_id") or "duty_id").agg({
            (grade_col or "duty_id"): "first",
            (vtype_col or "duty_id"): "first",
        }).reset_index()
        meta = norm_cols(meta)
        for _, r in meta.iterrows():
            d = str(r.get("duty_id"))
            if d in drivers:
                if grade_col:
                    drivers[d]["grade"] = r.get(grade_col)
                if vtype_col:
                    drivers[d]["vehicle_type"] = r.get(vtype_col)

    out = {"drivers": drivers, "location_index_size": int(li2["center_id"].max())+1}
    (outdir/"driver_states.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[ok] wrote", outdir/"driver_states.json")

if __name__ == "__main__":
    main()
