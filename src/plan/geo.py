from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from math import radians, sin, cos, asin, sqrt

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
