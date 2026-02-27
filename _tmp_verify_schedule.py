import requests

payload = {
    'start_location': 'BIRMINGHAM MAIL CENTRE',
    'end_location': 'MIDLANDS SUPER HUB',
    'when_local': '2025-09-02T10:30',
    'priority': 3,
    'mode': 'depart_after',
    'max_cascades': 2,
    'max_drivers_affected': 3,
}

r = requests.post('http://localhost:8000/plan/solve_cascades', json=payload, timeout=180)
print('status', r.status_code)
r.raise_for_status()
body = r.json()
print('assignments', [(a.get('driver_id'), a.get('candidate_id')) for a in body.get('assignments', [])])
print('cascade_diag', body.get('details', {}).get('cascade_diagnostics', {}))

schedules = body.get('schedules', []) or []
print('schedule_count', len(schedules))

for s in schedules:
    driver = s.get('driver_id')
    after = s.get('after', []) or []
    end_idxs = [i for i,e in enumerate(after) if 'END FACILITY' in str(e.get('element_type','')).upper()]
    rows_after = None if not end_idxs else len(after) - end_idxs[-1] - 1

    continuity_ok = True
    continuity_issue = None
    prev_end = None
    prev_to = None
    for e in sorted(after, key=lambda x: int(x.get('start_min', 0) or 0)):
        st = int(e.get('start_min', 0) or 0)
        en = int(e.get('end_min', st) or st)
        frm = str(e.get('from', '')).upper().strip()
        to = str(e.get('to', '')).upper().strip() or frm
        is_travel = bool(e.get('is_travel', False))
        if en < st:
            continuity_ok = False
            continuity_issue = 'end_before_start'
            break
        if prev_end is not None and st < prev_end:
            continuity_ok = False
            continuity_issue = 'time_overlap'
            break
        if prev_to is not None and frm and prev_to and frm != prev_to and not is_travel:
            continuity_ok = False
            continuity_issue = 'location_discontinuity'
            break
        prev_end = en
        prev_to = to

    print('driver', driver, {
        'after_len': len(after),
        'end_facility_indices': end_idxs,
        'rows_after_last_end_facility': rows_after,
        'continuity_ok': continuity_ok,
        'continuity_issue': continuity_issue,
    })

    tail = after[-8:] if len(after) > 8 else after
    print('tail', driver)
    for i,e in enumerate(tail, start=max(0, len(after)-len(tail))):
        print(i, e.get('element_type'), e.get('from'), '->', e.get('to'), e.get('start_min'), e.get('end_min'), e.get('changes'), e.get('load_type'), e.get('planz_code'))
