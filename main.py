#!/usr/bin/env python3
"""List available time slots for a practitioner in a scheduling grid.

Usage:
    python main.py <image> <practitioner>

Example:
    python main.py schedule.png "Karine Chartier"
"""

import os
import sys

from image_processor import (
    load_image, preprocess, apply_deskew,
    find_grid_area, find_columns,
)
from ocr_reader import (
    extract_time_labels, build_pixel_to_time,
    extract_column_headers, match_column,
)
from availability_checker import find_free_slots, format_slots
from cnn_predictor import load_cnn_model, classify_column_slots, estimate_row_height


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <image> <practitioner>")
        sys.exit(1)

    image_path = sys.argv[1]
    practitioner = sys.argv[2]

    # Step 1: Load
    image = load_image(image_path)

    # Step 2: Deskew
    _, skew_angle = preprocess(image)
    image = apply_deskew(image, skew_angle)

    # Step 3: Detect grid borders and column layout
    x_left, x_right, y_top, y_bottom = find_grid_area(image)
    all_cols = find_columns(image, x_left, x_right, y_top, y_bottom)

    if len(all_cols) < 2:
        print("Error: could not detect columns.", file=sys.stderr)
        sys.exit(1)

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

    header_h = max(18, int((y_bottom - y_top) * 0.03))
    header_y_top = y_top
    header_y_bot = y_top + header_h
    content_y_top = header_y_bot + 2

    # Step 6: OCR column headers and match practitioner
    col_names = extract_column_headers(image, data_cols, header_y_top, header_y_bot)

    result = match_column(practitioner, col_names, data_cols)
    if result is None:
        print(f"Error: '{practitioner}' not found.", file=sys.stderr)
        print("Detected columns:", file=sys.stderr)
        for i, name in enumerate(col_names):
            print(f"  {i + 1}. {name or '(unrecognized)'}", file=sys.stderr)
        sys.exit(1)

    col_idx, (col_x0, col_x1) = result

    # Step 5: OCR time labels and build pixel-to-time mapping
    labels = extract_time_labels(image, time_col[0], time_col[1],
                                 content_y_top, y_bottom)
    pixel_to_time, _ = build_pixel_to_time(labels)

    # Step 4: CNN-based slot classification
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "slot_classifier_cnn.keras")
    model = load_cnn_model(model_path)
    row_h = estimate_row_height(content_y_top, y_bottom)
    slot_results = classify_column_slots(image, model,
                                         (col_x0, col_x1),
                                         content_y_top, y_bottom, row_h)

    booked_slots = [
        (pixel_to_time(s["y_top"]), pixel_to_time(s["y_bot"]))
        for s in slot_results if s["label"] == "BOOKED"
    ]

    # Step 7: Compute and display free slots
    free = find_free_slots(booked_slots)
    matched_name = col_names[col_idx] or practitioner
    print(f"Available slots for {matched_name}:")
    print(format_slots(free))


if __name__ == "__main__":
    main()
