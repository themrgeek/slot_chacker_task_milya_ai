"""Load a scheduling grid image, detect grid structure and booked blocks."""

import cv2
import numpy as np


def load_image(path):
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def _group_consecutive(arr, gap=3):
    """Group sorted integers that are within *gap* of each other."""
    if len(arr) == 0:
        return []
    groups, current = [], [arr[0]]
    for v in arr[1:]:
        if v - current[-1] <= gap:
            current.append(v)
        else:
            groups.append(current)
            current = [v]
    groups.append(current)
    return groups


def find_grid_area(image):
    """Detect the green-bordered scheduling grid.

    Returns (x_left, x_right, y_top, y_bottom) where y_top is the first row
    below any internal green separator (i.e. the start of the column-header +
    content area, not the toolbar).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h >= 35) & (h <= 85) & (s > 100) & (v > 100)

    height, width = image.shape[:2]

    # Left / right borders — vertical green bands
    green_per_col = np.sum(green, axis=0)
    high_cols = np.where(green_per_col > height * 0.5)[0]
    col_groups = _group_consecutive(high_cols)
    if len(col_groups) >= 2:
        x_left = col_groups[0][-1] + 1
        x_right = col_groups[-1][0] - 1
    else:
        x_left, x_right = 30, width - 30

    # Horizontal green bands within the grid x-range
    green_in_grid = np.sum(green[:, x_left:x_right], axis=1)
    grid_width = max(x_right - x_left, 1)
    high_rows = np.where(green_in_grid > grid_width * 0.8)[0]
    row_groups = _group_consecutive(high_rows)

    if len(row_groups) >= 3:
        # Multiple horizontal green lines: skip toolbar by using the
        # second-to-last internal separator as the top boundary.
        y_top = row_groups[-2][-1] + 1
        y_bottom = row_groups[-1][0] - 1
    elif len(row_groups) >= 2:
        y_top = row_groups[0][-1] + 1
        y_bottom = row_groups[-1][0] - 1
    else:
        y_top, y_bottom = 70, height - 40

    return x_left, x_right, y_top, y_bottom


def find_columns(image, x_left, x_right, y_top, y_bottom):
    """Find column boundaries by detecting vertical dark separator lines.

    Returns list of (x_start, x_end) for every column (including time column).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    region = gray[y_top:y_bottom, x_left:x_right]

    dark_count = np.sum(region < 80, axis=0)
    threshold = (y_bottom - y_top) * 0.15
    dark_cols = np.where(dark_count > threshold)[0]

    groups = _group_consecutive(dark_cols)
    separators = [int(np.mean(g)) + x_left for g in groups]

    edges = [x_left] + separators + [x_right]
    bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return [(a, b) for a, b in bounds if b - a >= 20]


def detect_booked_blocks(image, y_top, y_bottom, x_start, x_end):
    """Find booked appointment rectangles within a specific column region.

    Free: yellow, desaturated (white/grey/peach), green borders.
    Booked: saturated non-yellow/non-green (pink, orange, teal, etc.).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    yellow = (h >= 15) & (h <= 35) & (s > 80) & (v > 150)
    neutral = (s < 30) & (v > 80)
    green = (h >= 35) & (h <= 85) & (s > 80) & (v > 80)

    free_mask = yellow | neutral | green
    booked_mask = ~free_mask

    margin = 3
    roi = np.zeros_like(booked_mask)
    roi[y_top:y_bottom, x_start + margin:x_end - margin] = True
    booked_mask = booked_mask & roi

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(booked_mask.astype(np.uint8) * 255,
                              cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_height = 10
    min_booked_fraction = 0.35
    blocks = []
    for cnt in contours:
        x, y, w, bh = cv2.boundingRect(cnt)
        if bh < min_height or w * bh < 300:
            continue
        # Verify the region actually has enough booked pixels (not just text)
        region_mask = booked_mask[y:y + bh, x:x + w]
        if region_mask.size > 0 and np.mean(region_mask) < min_booked_fraction:
            continue
        blocks.append((x, y, w, bh))
    return blocks
