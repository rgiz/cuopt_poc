set -euo pipefail

DATA_DIR="${1:-/data}"

echo "[1/4] Check input CSV exists"
test -f "$DATA_DIR/df_rsl_clean.csv" || {
  echo "ERR: $DATA_DIR/df_rsl_clean.csv not found"; exit 1; }

echo "[2/4] Prep centers + location_index + locations"
python scripts/data_prep.py --csv "$DATA_DIR/df_rsl_clean.csv" --out "$DATA_DIR"

echo "[3/4] Build matrices"
python scripts/quick_build_matrices.py \
  --locations_index "$DATA_DIR/location_index.csv" \
  --outdir "$DATA_DIR" --mph 28

# Optional: generate driver_states.json from df (weekday-aware)
if [ -f "$DATA_DIR/df_rsl_clean.csv" ]; then
python - <<'PY'
import pandas as pd, json, os
CSV=os.environ.get("CSV","/data/df_rsl_clean.csv")
LI=pd.read_csv(os.environ.get("LI","/data/location_index.csv"))
L2I={str(n).upper():int(i) for n,i in zip(LI["name"], LI["center_id"])}

def to_min(s):
    s=str(s)
    hh,mm,ss = (s.split(':')+['0','0'])[:3]
    return int(hh)*60+int(mm)

df=pd.read_csv(CSV, dtype=str).fillna("")
df["Mon"]=df["Mon"].astype(int); df["Tue"]=df["Tue"].astype(int); df["Wed"]=df["Wed"].astype(int)
df["Thu"]=df["Thu"].astype(int); df["Fri"]=df["Fri"].astype(int); df["Sat"]=df["Sat"].astype(int); df["Sun"]=df["Sun"].astype(int)

duties={}
for duty, g in df.groupby("Duty ID"):
    elements=[]
    for _,r in g.iterrows():
        e_type = r["Element Type"].upper()
        is_travel = ("TRAVEL" in e_type) or (r["PLANZ Code"].strip().upper() in ("EMPTY","TRAVEL_NO_DATA","TRAVEL NO DATA"))
        e={
            "element_type": r["Element Type"],
            "is_travel": bool(is_travel),
            "planz_code": r["PLANZ Code"],
            "from": str(r["Mapped Name A"]).upper(),
            "to":   str(r["Mapped Name B"]).upper(),
            "from_id": L2I.get(str(r["Mapped Name A"]).upper()),
            "to_id":   L2I.get(str(r["Mapped Name B"]).upper()),
            "start_min": to_min(r["Commencement Time"]),
            "end_min":   to_min(r["Ending Time"]),
            "duration_min": max(0, to_min(r["Ending Time"])-to_min(r["Commencement Time"])),
            "priority": 3,
            "Mon": int(r["Mon"]), "Tue": int(r["Tue"]), "Wed": int(r["Wed"]),
            "Thu": int(r["Thu"]), "Fri": int(r["Fri"]), "Sat": int(r["Sat"]), "Sun": int(r["Sun"]),
        }
        elements.append(e)
    duties[str(duty)] = {"elements": elements}

out=os.environ.get("OUT","/data/driver_states.json")
with open(out,"w") as f: json.dump(duties,f)
print(f"[info] wrote {out} with {len(duties)} duties")
PY
fi

echo "[4/4] Reload backend"
curl -fsS -X POST http://backend:8000/admin/reload > /dev/null || true
echo "Done."