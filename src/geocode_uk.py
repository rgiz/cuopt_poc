# src/geocode_uk.py
import re
from typing import Optional, Tuple
import pandas as pd
import pgeocode

_POSTCODE_RE = re.compile(r"[A-Za-z]{1,2}\d(?:\d|[A-Za-z])?\s*\d[A-Za-z]{2}")

def normalize_postcode(pc: str) -> Optional[str]:
    if not isinstance(pc, str):
        return None
    s = pc.upper().replace(" ", "")
    m = _POSTCODE_RE.search(s)
    if not m:
        return None
    s = m.group(0)
    return s[:-3].strip() + " " + s[-3:]

_nom = pgeocode.Nominatim("gb")

def lookup_latlon(postcode: str) -> Tuple[Optional[float], Optional[float]]:
    if not postcode:
        return (None, None)
    rec = _nom.query_postal_code(postcode)
    lat = None if pd.isna(rec.latitude) else float(rec.latitude)
    lon = None if pd.isna(rec.longitude) else float(rec.longitude)
    return (lat, lon)

def enrich_locations(
    locs: pd.DataFrame,
    id_col="location_id",
    lat_col="lat",
    lon_col="lon",
    pc_col="postcode",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (enriched_locs, issues). Fill missing lat/lon from postcode."""
    df = locs.copy()
    if pc_col in df.columns:
        df["postcode_norm"] = df[pc_col].apply(normalize_postcode)
    else:
        df["postcode_norm"] = None

    need = df[lat_col].isna() | df[lon_col].isna()
    fill_idx = df.index[need & df["postcode_norm"].notna()]
    for i in fill_idx:
        lat, lon = lookup_latlon(df.at[i, "postcode_norm"])
        if lat is not None and lon is not None:
            df.at[i, lat_col] = lat
            df.at[i, lon_col] = lon

    unresolved = df[df[lat_col].isna() | df[lon_col].isna()].copy()
    issues = pd.DataFrame(
        {
            "location_id": unresolved[id_col].tolist(),
            "reason": ["missing_latlon_after_enrichment"] * len(unresolved),
            "postcode_norm": unresolved.get("postcode_norm", pd.Series([None]*len(unresolved))),
        }
    )
    return df, issues
