"""Compute available time slots from a list of booked ranges."""


def _to_minutes(hm):
    """Convert (hour, minute) to total minutes since midnight."""
    return hm[0] * 60 + hm[1]


def _from_minutes(m):
    """Convert total minutes since midnight to (hour, minute)."""
    return (m // 60, m % 60)


def _fmt(hm):
    """Format (hour, minute) as HH:MM."""
    return f"{hm[0]:02d}:{hm[1]:02d}"


def find_free_slots(booked_slots, day_start=(8, 0), day_end=(20, 0),
                    min_gap_minutes=15):
    """Return a list of free (start, end) tuples between booked slots.

    booked_slots: list of ((h,m), (h,m)) already sorted by start time.
    Only gaps >= min_gap_minutes are included.
    """
    ds = _to_minutes(day_start)
    de = _to_minutes(day_end)

    merged = _merge_overlapping(booked_slots)

    free = []
    cursor = ds
    for slot_start, slot_end in merged:
        s = max(_to_minutes(slot_start), ds)
        e = min(_to_minutes(slot_end), de)
        if s > cursor and s - cursor >= min_gap_minutes:
            free.append((_from_minutes(cursor), _from_minutes(s)))
        cursor = max(cursor, e)

    if de > cursor and de - cursor >= min_gap_minutes:
        free.append((_from_minutes(cursor), _from_minutes(de)))

    return free


def _merge_overlapping(slots):
    """Merge overlapping / adjacent booked slots into non-overlapping ranges."""
    if not slots:
        return []
    by_start = sorted(slots, key=lambda s: _to_minutes(s[0]))
    merged = [by_start[0]]
    for start, end in by_start[1:]:
        prev_end_m = _to_minutes(merged[-1][1])
        cur_start_m = _to_minutes(start)
        cur_end_m = _to_minutes(end)
        if cur_start_m <= prev_end_m + 1:
            new_end = _from_minutes(max(prev_end_m, cur_end_m))
            merged[-1] = (merged[-1][0], new_end)
        else:
            merged.append((start, end))
    return merged


def format_slots(slots):
    """Return a human-readable multi-line string of time slots."""
    if not slots:
        return "  (none)"
    lines = []
    for start, end in slots:
        lines.append(f"  {_fmt(start)} - {_fmt(end)}")
    return "\n".join(lines)
