from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from math import radians, sin, cos, asin, sqrt

import math
import sys
from typing import Optional, Tuple, Dict, Any

def enhanced_distance_time_lookup(
    from_name: str, 
    to_name: str, 
    M: Dict[str, Any], 
    loc_meta: Dict[str, Any]
) -> Tuple[float, float, str]:
    """
    Enhanced distance/time lookup with haversine fallback.
    Returns: (miles, minutes, warning_message)
    """
    
    def debug_log(msg):
        print(f"DISTANCE LOOKUP: {msg}", file=sys.stderr, flush=True)
    
    from_name_clean = from_name.upper().strip()
    to_name_clean = to_name.upper().strip()
    
    debug_log(f"Looking up {from_name_clean} -> {to_name_clean}")
    
    # 1. CHECK MATRIX IN BOTH DIRECTIONS
    matrix_miles, matrix_minutes = check_matrix_bidirectional(
        from_name_clean, to_name_clean, M
    )
    
    if matrix_miles > 0 and matrix_minutes > 0:
        debug_log(f"Found in matrix: {matrix_miles:.1f} miles, {matrix_minutes:.1f} mins")
        return matrix_miles, matrix_minutes, ""
    
    # 2. CHECK FOR COORDINATES
    from_coords = get_location_coordinates(from_name_clean, loc_meta)
    to_coords = get_location_coordinates(to_name_clean, loc_meta)
    
    debug_log(f"Coordinates - From: {from_coords}, To: {to_coords}")
    
    # 3. HAVERSINE FALLBACK (if both have coordinates)
    if from_coords and to_coords:
        haversine_miles = calculate_haversine_miles(*from_coords, *to_coords)
        
        # Apply route factor and speed model
        route_miles = haversine_miles * 1.25  # 25% route factor for UK roads
        
        # Speed model: average 35mph with 10min overhead per journey
        route_minutes = (route_miles / 35.0) * 60 + 10
        
        warning = f"⚠️ ESTIMATED via haversine ({haversine_miles:.1f}mi direct)"
        debug_log(f"Haversine fallback: {route_miles:.1f} miles, {route_minutes:.1f} mins")
        
        return route_miles, route_minutes, warning
    
    # 4. MISSING POSTCODE FALLBACK
    missing_locations = []
    if not from_coords:
        missing_locations.append(from_name_clean)
    if not to_coords:
        missing_locations.append(to_name_clean)
    
    warning = f"🔴 POSTCODE MISSING ({', '.join(missing_locations)}), TIMES ESTIMATED"
    debug_log(f"Missing coordinates fallback: 50 miles, 60 mins")
    
    return 50.0, 60.0, warning

def check_matrix_bidirectional(
    from_name: str, 
    to_name: str, 
    M: Dict[str, Any]
) -> Tuple[float, float]:
    """Check distance/time matrix in both directions."""
    
    loc2idx = M.get("loc2idx", {})
    dist_matrix = M.get("dist")
    time_matrix = M.get("time")
    
    if not all([loc2idx, dist_matrix is not None, time_matrix is not None]):
        return 0.0, 0.0
    
    from_idx = loc2idx.get(from_name)
    to_idx = loc2idx.get(to_name)
    
    if from_idx is None or to_idx is None:
        return 0.0, 0.0
    
    try:
        # Try A -> B direction
        dist_ab = float(dist_matrix[from_idx, to_idx])
        time_ab = float(time_matrix[from_idx, to_idx])
        
        if dist_ab > 0 and time_ab > 0:
            return dist_ab, time_ab
        
        # Try B -> A direction (symmetric lookup)
        dist_ba = float(dist_matrix[to_idx, from_idx])
        time_ba = float(time_matrix[to_idx, from_idx])
        
        if dist_ba > 0 and time_ba > 0:
            return dist_ba, time_ba
            
    except (IndexError, ValueError, TypeError) as e:
        print(f"Matrix lookup error: {e}", file=sys.stderr)
    
    return 0.0, 0.0

def get_location_coordinates(
    location_name: str, 
    loc_meta: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    """Get lat/lon coordinates for a location."""
    
    # Try direct lookup in loc_meta
    if location_name in loc_meta:
        meta = loc_meta[location_name]
        lat = meta.get("lat")
        lon = meta.get("lon") 
        
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except (ValueError, TypeError):
                pass
    
    # Fallback: try to find in any key that contains the location name
    for key, meta in loc_meta.items():
        if location_name in key.upper():
            lat = meta.get("lat")
            lon = meta.get("lon")
            
            if lat is not None and lon is not None:
                try:
                    return float(lat), float(lon)
                except (ValueError, TypeError):
                    continue
    
    return None

def calculate_haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in miles."""
    
    # Earth radius in miles
    R = 3958.7613
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.7613  # miles
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    phi1 = radians(lat1); phi2 = radians(lat2)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2 * R * asin(sqrt(a))

def build_loc_meta_from_locations_csv(df) -> Dict[str, Dict[str, Any]]:
    """
    Build LOC_META keyed by uppercased canonical name (Mapped Name A).
    Adds alias by From Site if present.
    Expects columns: "Mapped Name A", "From Site", "Lat_A", "Long_A", "Mapped Postcode A"
    """
    meta: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return meta
    for _, row in df.iterrows():
        name = str(row.get("Mapped Name A", "")).strip()
        lat_s = str(row.get("Lat_A", "")).strip()
        lon_s = str(row.get("Long_A", "")).strip()
        pc    = str(row.get("Mapped Postcode A", "")).strip()
        if not name or not lat_s or not lon_s:
            continue
        try:
            lat = float(lat_s); lon = float(lon_s)
        except Exception:
            continue
        key = name.upper()
        rec = {"lat": lat, "lon": lon}
        if pc:
            rec["postcode"] = pc
        meta[key] = rec

        alias = str(row.get("From Site", "")).strip()
        if alias:
            meta[alias.upper()] = rec
    return meta

def postcode_is_NI(pc: Optional[str]) -> bool:
    return bool(pc) and pc.upper().strip().startswith("BT")

def same_island_by_meta(mi: Dict[str,Any], mj: Dict[str,Any]) -> Optional[bool]:
    if not (mi and mj):
        return None
    if ("postcode" in mi) or ("postcode" in mj):
        return postcode_is_NI(mi.get("postcode")) == postcode_is_NI(mj.get("postcode"))
    # fallback coarse bbox heuristic
    def _is_NI(m): return (m.get("lon", 0) < -5.4 and m.get("lat", 0) > 54.0)
    return _is_NI(mi) == _is_NI(mj)

def haversine_between_idx(i: int, j: int, loc2idx: Dict[str,int], loc_meta: Dict[str,Any]) -> Optional[float]:
    nm_i = next((k for k,v in loc2idx.items() if int(v)==int(i)), None)
    nm_j = next((k for k,v in loc2idx.items() if int(v)==int(j)), None)
    mi = loc_meta.get(nm_i.upper()) if nm_i else None
    mj = loc_meta.get(nm_j.upper()) if nm_j else None
    if not (mi and mj and "lat" in mi and "lon" in mi and "lat" in mj and "lon" in mj):
        return None
    return haversine_miles(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
