"""Parse a user time range and check it against booked slots."""


def parse_time_range(text):
    """Parse 'HH:MM-HH:MM' into two (hour, minute) tuples."""
    parts = text.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM-HH:MM, got: {text!r}")
    return _parse_hhmm(parts[0]), _parse_hhmm(parts[1])


def _parse_hhmm(s):
    """Parse 'HH:MM' or 'H:MM' into (hour, minute)."""
    pieces = s.strip().split(":")
    if len(pieces) != 2:
        raise ValueError(f"Invalid time: {s!r}")
    h, m = int(pieces[0]), int(pieces[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Time out of range: {s!r}")
    return h, m


def _to_minutes(hm):
    """Convert (hour, minute) to total minutes since midnight."""
    return hm[0] * 60 + hm[1]


def check_availability(booked_slots, req_start, req_end):
    """Check whether the requested range overlaps any booked slot.

    booked_slots: list of ((start_h, start_m), (end_h, end_m))
    req_start, req_end: (hour, minute) tuples

    Returns 'available' or 'not available'.
    """
    req_s = _to_minutes(req_start)
    req_e = _to_minutes(req_end)

    for slot_start, slot_end in booked_slots:
        s = _to_minutes(slot_start)
        e = _to_minutes(slot_end)
        if req_s < e and req_e > s:
            return "not available"

    return "available"
