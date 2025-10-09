set -euo pipefail

DATA_DIR="${1:-/data}"

echo "[1/4] Check input CSV exists"
test -f "$DATA_DIR/df_rsl_clean.csv" || {
  echo "ERR: $DATA_DIR/df_rsl_clean.csv not found"; exit 1; }

echo "[2/4] Prep centers + location_index + locations"
PYTHONPATH=. python3 scripts/data_prep.py --csv "$DATA_DIR/df_rsl_clean.csv" --out "$DATA_DIR"

echo "[3/4] Build matrices"
PYTHONPATH=. python3 scripts/quick_build_matrices.py \
  --locations_index "$DATA_DIR/location_index.csv" \
  --outdir "$DATA_DIR" --mph 28

# Optional: generate driver_states.json from df (weekday-aware)
if [ -f "$DATA_DIR/df_rsl_clean.csv" ]; then
PYTHONPATH=. python3 - <<PY
import pandas as pd, json, os

# ✅ Use DATA_DIR from environment
DATA_DIR = os.environ.get("DATA_DIR", "data")
CSV = os.path.join(DATA_DIR, "df_rsl_clean.csv")
LI_FILE = os.path.join(DATA_DIR, "location_index.csv")

LI = pd.read_csv(LI_FILE)
L2I = {str(n).upper(): int(i) for n, i in zip(LI["name"], LI["center_id"])}

# ✅ LOAD PRIORITY MAP
PRIORITY_MAP_FILE = os.path.join(DATA_DIR, "priority_map.json")
if os.path.exists(PRIORITY_MAP_FILE):
    with open(PRIORITY_MAP_FILE, 'r') as f:
        PRIORITY_MAP = json.load(f)
    print(f"[info] Loaded priority map with {len(PRIORITY_MAP)} entries")
else:
    PRIORITY_MAP = {"DEFAULT": 3}
    print("[warn] No priority_map.json found, using default priority 3")

def get_priority(due_to_convey, planz_code="", load_type=""):
    """Get priority from Due to Convey, then PLANZ Code, then Load Type"""
    # Try Due to Convey first
    dtc = str(due_to_convey).strip().upper()
    if dtc and dtc != "NAN" and dtc in PRIORITY_MAP:
        return PRIORITY_MAP[dtc]
    
    # Try PLANZ Code second
    pc = str(planz_code).strip().upper()
    if pc and pc != "NAN" and pc in PRIORITY_MAP:
        return PRIORITY_MAP[pc]
    
    # Try Load Type third
    lt = str(load_type).strip().upper()
    if lt and lt != "NAN" and lt in PRIORITY_MAP:
        return PRIORITY_MAP[lt]
    
    # Default priority
    return PRIORITY_MAP.get("DEFAULT", 3)

def to_min(s):
    s = str(s)
    hh, mm, ss = (s.split(':') + ['0', '0'])[:3]
    return int(hh) * 60 + int(mm)

df = pd.read_csv(CSV, dtype=str).fillna("")
df["Mon"] = df["Mon"].astype(int)
df["Tue"] = df["Tue"].astype(int)
df["Wed"] = df["Wed"].astype(int)
df["Thu"] = df["Thu"].astype(int)
df["Fri"] = df["Fri"].astype(int)
df["Sat"] = df["Sat"].astype(int)
df["Sun"] = df["Sun"].astype(int)

duties = {}
for duty, g in df.groupby("Duty ID"):
    elements = []
    for _, r in g.iterrows():
        e_type = r["Element Type"].upper()
        
        # Extract fields for priority lookup
        due_to_convey = r.get("Due To Convey", r.get("Due to Convey", ""))
        planz_code = r.get("PLANZ Code", r.get("Planz Code", ""))
        load_type = r.get("Load Type", "")
        
        is_travel = ("TRAVEL" in e_type) or (str(planz_code).strip().upper() in ("EMPTY", "TRAVEL_NO_DATA", "TRAVEL NO DATA"))
        
        # ✅ GET PRIORITY
        priority = get_priority(due_to_convey, planz_code, load_type)
        
        e = {
            "element_type": r["Element Type"],
            "is_travel": bool(is_travel),
            "due_to_convey": str(due_to_convey),
            "planz_code": str(planz_code),
            "load_type": str(load_type),
            "from": str(r["Mapped Name A"]).upper(),
            "to": str(r["Mapped Name B"]).upper(),
            "from_id": L2I.get(str(r["Mapped Name A"]).upper()),
            "to_id": L2I.get(str(r["Mapped Name B"]).upper()),
            "start_min": to_min(r["Commencement Time"]),
            "end_min": to_min(r["Ending Time"]),
            "duration_min": max(0, to_min(r["Ending Time"]) - to_min(r["Commencement Time"])),
            "priority": priority,
            "Mon": int(r["Mon"]),
            "Tue": int(r["Tue"]),
            "Wed": int(r["Wed"]),
            "Thu": int(r["Thu"]),
            "Fri": int(r["Fri"]),
            "Sat": int(r["Sat"]),
            "Sun": int(r["Sun"]),
        }
        elements.append(e)
    duties[str(duty)] = {"elements": elements}

out = os.path.join(DATA_DIR, "driver_states.json")
with open(out, "w") as f:
    json.dump(duties, f)
print(f"[info] wrote {out} with {len(duties)} duties")

# ✅ PRINT PRIORITY DISTRIBUTION
priority_counts = {}
priority_samples = {}
for duty_data in duties.values():
    for elem in duty_data["elements"]:
        p = elem["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1
        
        dtc = elem.get("due_to_convey", "")
        if dtc and dtc.strip() and dtc != "NAN":
            if p not in priority_samples:
                priority_samples[p] = set()
            priority_samples[p].add(dtc.strip())

print(f"[info] Priority distribution: {priority_counts}")
print(f"[info] Sample mappings:")
for p in sorted(priority_samples.keys()):
    samples = list(priority_samples[p])[:3]
    print(f"  Priority {p}: {', '.join(samples)}")
PY
fi

echo "[4/4] Reload backend"
curl -fsS -X POST http://backend:8000/admin/reload > /dev/null || true
echo "Done."