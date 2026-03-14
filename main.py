#!/usr/bin/env python3
"""Check time-slot availability for a practitioner in a scheduling grid.

Usage:
    python main.py <image> <practitioner> <HH:MM-HH:MM>

Example:
    python main.py schedule.png "Dre Yasmin Bouzaza" 11:00-11:30
"""

import sys

from image_processor import load_image, find_grid_area, find_columns, detect_booked_blocks
from ocr_reader import (extract_time_labels, build_pixel_to_time,
                        extract_column_headers, match_column)
from availability_checker import parse_time_range, check_availability


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <image> <practitioner> <HH:MM-HH:MM>")
        sys.exit(1)

    image_path, practitioner, time_range = sys.argv[1], sys.argv[2], sys.argv[3]
    image = load_image(image_path)

    # Detect grid borders and column layout
    x_left, x_right, y_top, y_bottom = find_grid_area(image)
    all_cols = find_columns(image, x_left, x_right, y_top, y_bottom)

    if len(all_cols) < 2:
        print("Error: could not detect columns.", file=sys.stderr)
        sys.exit(1)

    # Separate time-label column (narrowest, usually first) from data columns
    widths = [b - a for a, b in all_cols]
    median_w = sorted(widths)[len(widths) // 2]
    time_col = None
    data_cols = []
    for bounds in all_cols:
        if bounds[1] - bounds[0] < median_w * 0.6 and time_col is None:
            time_col = bounds
        else:
            data_cols.append(bounds)

    if time_col is None:
        time_col = (x_left, all_cols[0][0])

    # Column headers sit in the first ~25px below y_top (just under the
    # internal green separator). Scheduling content starts below that.
    header_h = max(18, int((y_bottom - y_top) * 0.03))
    header_y_top = y_top
    header_y_bot = y_top + header_h
    content_y_top = header_y_bot + 2

    col_names = extract_column_headers(image, data_cols, header_y_top, header_y_bot)

    # Match practitioner name to a column
    result = match_column(practitioner, col_names, data_cols)
    if result is None:
        print(f"Error: '{practitioner}' not found.", file=sys.stderr)
        for i, name in enumerate(col_names):
            print(f"  {i + 1}. {name or '(unrecognized)'}", file=sys.stderr)
        sys.exit(1)

    col_idx, (col_x0, col_x1) = result

    # OCR time labels and build pixel-to-time mapping
    labels = extract_time_labels(image, time_col[0], time_col[1],
                                 content_y_top, y_bottom)
    pixel_to_time, (grid_y_top, grid_y_bot) = build_pixel_to_time(labels)

    # Detect booked blocks in the matched column
    rects = detect_booked_blocks(image, grid_y_top, grid_y_bot, col_x0, col_x1)
    booked_slots = [(pixel_to_time(y), pixel_to_time(y + h)) for _, y, _, h in rects]

    # Check overlap and print result
    req_start, req_end = parse_time_range(time_range)
    print(check_availability(booked_slots, req_start, req_end))


if __name__ == "__main__":
    main()
