from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

ENFORCE_SAME_ISLAND = _env_bool("ENFORCE_SAME_ISLAND", True)
USE_HAVERSINE_DEADHEAD = _env_bool("USE_HAVERSINE_DEADHEAD", True)
HAV_MAX_DEADHEAD_ONE_WAY_MI = _env_float("HAV_MAX_DEADHEAD_ONE_WAY_MI", 120.0)

def dataset_dir() -> Path:
    base = Path(os.getenv("PRIVATE_DATA_DIR", "./data/private")).resolve()
    d = base / "active"
    return d if d.exists() else base

def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def load_priority_map() -> Dict[str,int]:
    mp = _load_json(dataset_dir() / "priority_map.json", {})
    return {str(k).upper(): int(v) for k, v in mp.items()}

def load_sla_windows() -> Dict[int, Dict[str,int]]:
    raw = _load_json(dataset_dir() / "sla_windows.json", {})
    out: Dict[int, Dict[str,int]] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = {
                "early_min": int(v.get("early_min", 60)),
                "late_min":  int(v.get("late_min", 60)),
            }
        except Exception:
            continue
    if not out:
        out = {
            1: {"early_min":15, "late_min":30},
            2: {"early_min":30, "late_min":45},
            3: {"early_min":60, "late_min":60},
            4: {"early_min":90, "late_min":90},
            5: {"early_min":120,"late_min":120},
        }
    return out
