#!/usr/bin/env python3
"""
Phase 1: Proximity-based OSRM matrix building.

This script builds comprehensive distance/time matrices by:
1. Starting with RSL baseline data
2. Using haversine to identify location pairs within proximity thresholds
3. Using OSRM to compute real routing for proximate pairs
4. Two-tier distance thresholds: 100mi for hubs, 50mi for regular locations

Usage:
  python3 scripts/build_proximity_matrices.py \
      --rsl data/df_rsl_clean.csv \
      --loc-index data/location_index.csv \
      --outdir data/private/active \
      --osrm-url http://localhost:5001
"""

import argparse
import json
from typing import Optional
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import math

import pandas as pd
import numpy as np
import requests

from build_dataset_from_rsl import build_matrices_from_rsl_original, norm_cols, detect_col, median_safe

# Add path for imports from existing script
sys.path.insert(0, str(Path(__file__).parent.parent))

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in miles."""
    R = 3958.7613  # Earth radius in miles
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def get_distance_threshold(location_name: str) -> int:
    """Get distance threshold based on location type."""
    hub_keywords = ["AIRPORT", "VOC", "MAIL CENTRE", "MC", "HUB"]
    name_upper = str(location_name).upper()
    
    if any(keyword in name_upper for keyword in hub_keywords):
        return 100  # miles for major transport hubs
    return 50  # miles for regular locations

def osrm_health_check(base_url: str) -> bool:
    """Check if OSRM service is available using a simple route request."""
    try:
        test_url = f"{base_url.rstrip('/')}/route/v1/driving/-0.1,51.5;-0.2,51.6"
        response = requests.get(test_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_osrm_route(from_coords: Tuple[float, float], to_coords: Tuple[float, float], osrm_url: str) -> Tuple[float, float]:
    """
    Get single route from OSRM Route API.
    
    Returns:
        Tuple of (distance_miles, duration_minutes)
    """
    lon1, lat1 = from_coords
    lon2, lat2 = to_coords
    
    url = f"{osrm_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"steps": "false", "geometries": "geojson"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM route error: {data.get('message', 'Unknown error')}")
        
        route = data["routes"][0]
        distance_miles = route["distance"] * 0.000621371  # meters to miles
        duration_minutes = route["duration"] / 60.0  # seconds to minutes
        
        return distance_miles, duration_minutes
        
    except Exception as e:
        print(f"[debug] OSRM route failed for {from_coords} -> {to_coords}: {e}")
        raise

def identify_proximity_pairs(locations_df: pd.DataFrame) -> List[Tuple[int, int, float, int, int]]:
    """
    Identify location pairs within proximity thresholds using haversine distance.
    
    Returns:
        List of tuples: (from_id, to_id, haversine_distance, from_threshold, to_threshold)
    """
    print("[info] Identifying proximity pairs using haversine distances...")
    
    proximity_pairs = []
    n = len(locations_df)
    
    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle, avoid duplicates
            row_i = locations_df.iloc[i]
            row_j = locations_df.iloc[j]
            
            # Skip if either location missing coordinates
            if pd.isna(row_i['lat']) or pd.isna(row_i['lon']) or pd.isna(row_j['lat']) or pd.isna(row_j['lon']):
                continue
            
            # Calculate haversine distance
            haversine_dist = haversine_miles(
                row_i['lat'], row_i['lon'],
                row_j['lat'], row_j['lon']
            )
            
            # Get distance thresholds for both locations
            threshold_i = get_distance_threshold(row_i['name'])
            threshold_j = get_distance_threshold(row_j['name'])
            max_threshold = max(threshold_i, threshold_j)  # Use the larger threshold
            
            # Include pair if within threshold
            if haversine_dist <= max_threshold:
                proximity_pairs.append((
                    int(row_i['center_id']), 
                    int(row_j['center_id']),
                    haversine_dist,
                    threshold_i,
                    threshold_j
                ))
    
    print(f"[info] Found {len(proximity_pairs)} location pairs within proximity thresholds")
    
    # Print threshold breakdown
    hub_pairs = sum(1 for _, _, _, t1, t2 in proximity_pairs if max(t1, t2) == 100)
    regular_pairs = len(proximity_pairs) - hub_pairs
    print(f"[info] Hub pairs (100mi threshold): {hub_pairs}")
    print(f"[info] Regular pairs (50mi threshold): {regular_pairs}")
    
    return proximity_pairs

def load_rsl_baseline(rsl_path: Path, locations_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Set[Tuple[int, int]]]:
    """
    Load RSL baseline data to initialize matrices.
    """
    print("[info] Loading RSL baseline data...")
    
    # Load and process RSL data
    rsl = pd.read_csv(rsl_path)
    rsl = norm_cols(rsl)
    
    # Use existing function to build matrices
    D, T = build_matrices_from_rsl_original(rsl, locations_df, mph=45.0, fill_missing="zero")
    
    # Identify which pairs have RSL data (non-zero entries)
    rsl_pairs = set()
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i, j] > 0:
                rsl_pairs.add((i, j))
    
    print(f"[info] RSL baseline loaded: {len(rsl_pairs)} known pairs")
    return D, T, rsl_pairs

def compute_osrm_for_pairs(proximity_pairs: List[Tuple[int, int, float, int, int]], 
                          locations_df: pd.DataFrame, 
                          osrm_url: str,
                          rsl_pairs: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Compute OSRM routing for proximity pairs that aren't already in RSL data.
    """
    print("[info] Computing OSRM routes for proximity pairs...")
    
    # Filter out pairs already in RSL data
    osrm_needed = []
    for from_id, to_id, haversine_dist, t1, t2 in proximity_pairs:
        if (from_id, to_id) not in rsl_pairs and (to_id, from_id) not in rsl_pairs:
            osrm_needed.append((from_id, to_id, haversine_dist))
    
    print(f"[info] Need OSRM routing for {len(osrm_needed)} pairs (after excluding RSL pairs)")
    
    # Create coordinate lookup
    coord_lookup = {}
    for _, row in locations_df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            coord_lookup[int(row['center_id'])] = (float(row['lon']), float(row['lat']))
    
    osrm_results = {}
    failed_pairs = []
    
    print(f"[info] Starting OSRM route computation...")
    start_time = time.time()
    
    for idx, (from_id, to_id, haversine_dist) in enumerate(osrm_needed):
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = len(osrm_needed) - idx
            eta = remaining / rate if rate > 0 else 0
            print(f"[info] Progress: {idx}/{len(osrm_needed)} ({idx/len(osrm_needed)*100:.1f}%) - ETA: {eta/60:.1f}min")
        
        if from_id not in coord_lookup or to_id not in coord_lookup:
            failed_pairs.append((from_id, to_id, "Missing coordinates"))
            continue
        
        from_coords = coord_lookup[from_id]
        to_coords = coord_lookup[to_id]
        
        try:
            # Get both directions
            dist_fw, time_fw = get_osrm_route(from_coords, to_coords, osrm_url)
            dist_bw, time_bw = get_osrm_route(to_coords, from_coords, osrm_url)
            
            osrm_results[(from_id, to_id)] = (dist_fw, time_fw)
            osrm_results[(to_id, from_id)] = (dist_bw, time_bw)
            
            # Small delay to be respectful to OSRM
            time.sleep(0.05)
            
        except Exception as e:
            failed_pairs.append((from_id, to_id, str(e)))
            continue
    
    elapsed_total = time.time() - start_time
    success_rate = len(osrm_results) / (2 * len(osrm_needed)) * 100 if osrm_needed else 100
    
    print(f"[info] OSRM computation complete!")
    print(f"[info] Total time: {elapsed_total/60:.1f} minutes")
    print(f"[info] Success rate: {success_rate:.1f}%")
    print(f"[info] Successfully computed: {len(osrm_results)} route pairs")
    print(f"[info] Failed: {len(failed_pairs)} pairs")
    
    if failed_pairs:
        print(f"[info] Sample failures: {failed_pairs[:3]}")
    
    return osrm_results

def build_final_matrices(locations_df: pd.DataFrame, 
                        rsl_baseline_dist: np.ndarray,
                        rsl_baseline_time: np.ndarray,
                        osrm_results: Dict[Tuple[int, int], Tuple[float, float]],
                        fill_missing: str = "haversine",
                        mph: float = 45.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build final matrices combining RSL baseline + OSRM proximity data + haversine fallback.
    """
    print("[info] Building final matrices...")
    
    max_id = int(locations_df['center_id'].max())
    K = max_id + 1
    
    # Start with RSL baseline
    D = rsl_baseline_dist.copy()
    T = rsl_baseline_time.copy()
    
    # Add OSRM results
    osrm_added = 0
    for (from_id, to_id), (distance, duration) in osrm_results.items():
        D[from_id, to_id] = distance
        T[from_id, to_id] = duration
        osrm_added += 1
    
    print(f"[info] Added {osrm_added} OSRM route entries to matrices")
    
    # Fill remaining gaps with haversine if requested
    if fill_missing.lower() == "haversine":
        print("[info] Filling remaining gaps with haversine estimates...")
        
        coord_lookup = {}
        for _, row in locations_df.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                coord_lookup[int(row['center_id'])] = (float(row['lat']), float(row['lon']))
        
        haversine_added = 0
        for i in range(K):
            for j in range(K):
                if i != j and D[i, j] == 0 and i in coord_lookup and j in coord_lookup:
                    lat1, lon1 = coord_lookup[i]
                    lat2, lon2 = coord_lookup[j]
                    hav_dist = haversine_miles(lat1, lon1, lat2, lon2)
                    hav_time = hav_dist / mph * 60
                    
                    D[i, j] = hav_dist
                    T[i, j] = hav_time
                    haversine_added += 1
        
        print(f"[info] Added {haversine_added} haversine estimates")
    
    # Calculate final coverage
    total_pairs = K * K - K  # Exclude diagonal
    valid_pairs = np.sum(D > 0) - K  # Exclude diagonal
    coverage_pct = (valid_pairs / total_pairs * 100) if total_pairs > 0 else 100
    
    print(f"[info] Final matrix coverage: {coverage_pct:.1f}% ({valid_pairs}/{total_pairs} pairs)")
    
    return D, T

def main():
    parser = argparse.ArgumentParser(description="Build proximity-based OSRM matrices")
    parser.add_argument("--rsl", required=True, help="Path to df_rsl_clean CSV")
    parser.add_argument("--loc-index", required=True, help="Path to location_index.csv")
    parser.add_argument("--outdir", default="data/private/active", help="Output directory")
    parser.add_argument("--osrm-url", default="http://localhost:5001", help="OSRM service URL")
    parser.add_argument("--fill-missing", choices=["haversine", "zero"], default="haversine",
                       help="How to fill remaining gaps")
    parser.add_argument("--mph", type=float, default=45.0, help="Fallback mph for haversine time")
    parser.add_argument("--dry-run", action="store_true", help="Show proximity pairs without calling OSRM")
    
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PROXIMITY-BASED OSRM MATRIX BUILDING")
    print("=" * 60)
    
    # Load location index
    print(f"[info] Loading location index from {args.loc_index}")
    locations_df = pd.read_csv(args.loc_index)
    
    # Normalize column names
    locations_df.columns = [c.strip().lower().replace(" ", "_") for c in locations_df.columns]
    
    print(f"[info] Loaded {len(locations_df)} locations")
    
    # Validate coordinates
    valid_coords = locations_df[locations_df['lat'].notna() & locations_df['lon'].notna()]
    print(f"[info] {len(valid_coords)} locations have valid coordinates")
    
    # Identify proximity pairs
    proximity_pairs = identify_proximity_pairs(locations_df)
    
    if args.dry_run:
        print("\n[DRY RUN] Would compute OSRM routes for these proximity pairs")
        for i, (from_id, to_id, dist, t1, t2) in enumerate(proximity_pairs[:10]):
            from_name = locations_df[locations_df['center_id'] == from_id]['name'].iloc[0]
            to_name = locations_df[locations_df['center_id'] == to_id]['name'].iloc[0]
            print(f"  {i+1}: {from_name} <-> {to_name} ({dist:.1f}mi, thresholds: {t1}mi/{t2}mi)")
        if len(proximity_pairs) > 10:
            print(f"  ... and {len(proximity_pairs) - 10} more pairs")
        return
    
    # Check OSRM availability
    if not osrm_health_check(args.osrm_url):
        print(f"[error] OSRM service not available at {args.osrm_url}")
        return 1
    
    print(f"[info] OSRM service available at {args.osrm_url}")
    
    # Load RSL baseline
    rsl_baseline_dist, rsl_baseline_time, rsl_pairs = load_rsl_baseline(Path(args.rsl), locations_df)
    
    # Compute OSRM for proximity pairs
    osrm_results = compute_osrm_for_pairs(proximity_pairs, locations_df, args.osrm_url, rsl_pairs)
    
    # Build final matrices
    final_dist, final_time = build_final_matrices(
        locations_df, rsl_baseline_dist, rsl_baseline_time, osrm_results, 
        args.fill_missing, args.mph
    )
    
    # Save matrices
    dist_file = outdir / "distance_miles_matrix.npz"
    time_file = outdir / "time_minutes_matrix.npz"
    
    np.savez_compressed(dist_file, matrix=final_dist)
    np.savez_compressed(time_file, matrix=final_time)
    
    print(f"[ok] Saved matrices:")
    print(f"     {dist_file}")
    print(f"     {time_file}")
    
    # Save build statistics
    stats = {
        "build_method": "proximity_osrm",
        "osrm_url": args.osrm_url,
        "total_locations": len(locations_df),
        "proximity_pairs_identified": len(proximity_pairs),
        "osrm_routes_computed": len(osrm_results),
        "matrix_shape": list(final_dist.shape),
        "final_coverage_percent": float(np.sum(final_dist > 0) / (final_dist.size - final_dist.shape[0]) * 100),
    }
    
    stats_file = outdir / "matrix_build_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[ok] Saved build statistics: {stats_file}")
    print("\n" + "=" * 60)
    print("MATRIX BUILD COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    sys.exit(main())