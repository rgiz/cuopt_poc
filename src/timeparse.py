# src/timeparse.py
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Tuple, Optional
from dateutil import parser as du
from zoneinfo import ZoneInfo

UK = ZoneInfo("Europe/London")
_HHMM = re.compile(r"^(\d{1,2}):?(\d{2})$")

def parse_date(d) -> datetime.date:
    return du.parse(str(d), dayfirst=True).date()

def parse_time(t) -> Tuple[int, int]:
    if t is None:
        return (0, 0)
    s = str(t).strip()
    if s.isdigit():
        if len(s) <= 2:
            return (int(s), 0)
        return (int(s[:-2]), int(s[-2:]))
    m = _HHMM.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    dt = du.parse(s)
    return (dt.hour, dt.minute)

def combine_dt(date_str, time_val, tz=UK, allow_24=True) -> datetime:
    d = parse_date(date_str)
    hh, mm = parse_time(time_val)
    if allow_24 and hh == 24 and mm == 0:
        return datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz) + timedelta(days=1)
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)

def minutes_since_midnight(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute

def window_from_priority(priority: int, anchor_start: datetime, policy: Optional[dict[int,int]] = None) -> tuple[int,int]:
    if policy is None:
        policy = {1: 60, 2: 120, 3: 240, 4: 720, 5: 1440}
    s = minutes_since_midnight(anchor_start)
    e = min(1440, s + policy.get(int(priority or 3), 240))
    return s, e
