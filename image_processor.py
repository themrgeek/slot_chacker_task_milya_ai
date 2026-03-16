"""Load a scheduling grid image, detect grid structure and booked rows."""

import cv2
import numpy as np


def load_image(path):
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def preprocess(image):
    """Grayscale -> Gaussian blur -> Otsu threshold -> noise removal -> deskew.

    Returns (binary, skew_angle).  *binary* uses BINARY_INV convention:
    foreground (text / dark content) = 255, background = 0.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    angle = _detect_skew(binary)
    if abs(angle) > 0.5:
        binary = _rotate_img(binary, -angle)

    return binary, angle


def apply_deskew(image, angle):
    """Rotate a colour image by the same skew angle found in *preprocess*."""
    if abs(angle) > 0.5:
        return _rotate_img(image, -angle)
    return image


def _detect_skew(binary):
    """Estimate skew angle (degrees) from the binary image."""
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    return angle


def _rotate_img(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Grid detection
# ------------------------------------------------------------------

def find_grid_area(image):
    """Detect the green-bordered scheduling grid.

    Returns (x_left, x_right, y_top, y_bottom).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h >= 35) & (h <= 85) & (s > 100) & (v > 100)

    height, width = image.shape[:2]

    green_per_col = np.sum(green, axis=0)
    high_cols = np.where(green_per_col > height * 0.5)[0]
    col_groups = _group_consecutive(high_cols)
    if len(col_groups) >= 2:
        x_left = col_groups[0][-1] + 1
        x_right = col_groups[-1][0] - 1
    else:
        x_left, x_right = 30, width - 30

    green_in_grid = np.sum(green[:, x_left:x_right], axis=1)
    grid_width = max(x_right - x_left, 1)
    high_rows = np.where(green_in_grid > grid_width * 0.8)[0]
    row_groups = _group_consecutive(high_rows)

    if len(row_groups) >= 3:
        y_top = row_groups[-2][-1] + 1
        y_bottom = row_groups[-1][0] - 1
    elif len(row_groups) >= 2:
        y_top = row_groups[0][-1] + 1
        y_bottom = row_groups[-1][0] - 1
    else:
        y_top, y_bottom = 70, height - 40

    return x_left, x_right, y_top, y_bottom


def find_columns(image, x_left, x_right, y_top, y_bottom):
    """Find column boundaries by detecting vertical separator lines.

    Returns list of (x_start, x_end).
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


# ------------------------------------------------------------------
# Grid-line removal
# ------------------------------------------------------------------

def remove_grid_lines(binary, image, x_left, x_right, y_top, y_bottom):
    """Remove grid-separator lines from the binary image.

    Uses morphological long-line detection to find horizontal and vertical
    structures that are grid borders, then erases them.  Green grid lines
    are handled by applying the same morphological filter to a green-pixel
    mask (so only green *lines* are removed, not green content blocks).
    """
    cleaned = binary.copy()
    region = cleaned[y_top:y_bottom, x_left:x_right].copy()
    rh, rw = region.shape

    # Horizontal kernel must be longer than any single column so that
    # column content (~170 px) is NOT mistaken for a grid line.
    # Real grid lines span the full grid width (~2000 px).
    h_len = max(rw // 4, 200)
    h_kernel = np.ones((1, h_len), np.uint8)
    h_lines = cv2.morphologyEx(region, cv2.MORPH_OPEN, h_kernel)

    v_len = max(rh // 4, 100)
    v_kernel = np.ones((v_len, 1), np.uint8)
    v_lines = cv2.morphologyEx(region, cv2.MORPH_OPEN, v_kernel)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gr = hsv[y_top:y_bottom, x_left:x_right]
    green_all = (
        (gr[:, :, 0] >= 35) & (gr[:, :, 0] <= 85)
        & (gr[:, :, 1] > 80) & (gr[:, :, 2] > 80)
    ).astype(np.uint8) * 255

    green_h = cv2.morphologyEx(green_all, cv2.MORPH_OPEN, h_kernel)
    green_v = cv2.morphologyEx(green_all, cv2.MORPH_OPEN, v_kernel)
    green_lines = green_h | green_v

    h_lines = cv2.dilate(h_lines, np.ones((3, 1), np.uint8))
    v_lines = cv2.dilate(v_lines, np.ones((1, 3), np.uint8))
    green_lines = cv2.dilate(green_lines, np.ones((3, 3), np.uint8))

    grid_mask = h_lines | v_lines | green_lines
    region[grid_mask > 0] = 0
    cleaned[y_top:y_bottom, x_left:x_right] = region
    return cleaned


# ------------------------------------------------------------------
# Booked / free classification
# ------------------------------------------------------------------

def detect_booked_ranges(image, binary_clean, y_top, y_bottom, x_start, x_end):
    """Return booked (y_start, y_end) pixel ranges for one column.

    Three complementary signals are combined:
      1. **Ink fraction** from the Otsu-binarised image (catches dark text on
         any background -- "Hors du bureau", patient names, etc.).
      2. **Non-yellow colour fraction** from the original HSV image (catches
         green / pink / purple appointment blocks).
      3. **Dark fraction** (V < 140) for gray content blocks such as
         "Pause-sante" backgrounds.

    Plain yellow rows (the FREE state) score ~0 on all three signals.
    """
    margin = 4
    xs = x_start + margin
    xe = x_end - margin
    if xe <= xs:
        return []
    col_width = xe - xs

    ink_region = binary_clean[y_top:y_bottom, xs:xe]
    ink_frac = np.sum(ink_region > 0, axis=1) / col_width

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_col = hsv[y_top:y_bottom, xs:xe, 0].astype(int)
    s_col = hsv[y_top:y_bottom, xs:xe, 1].astype(int)
    v_col = hsv[y_top:y_bottom, xs:xe, 2].astype(int)

    is_saturated = s_col > 40
    is_yellow = (h_col >= 18) & (h_col <= 38) & (v_col > 180)
    non_yellow_colored = is_saturated & ~is_yellow
    color_frac = np.sum(non_yellow_colored, axis=1) / col_width

    is_dark = v_col < 140
    dark_frac = np.sum(is_dark, axis=1) / col_width

    booked_flags = (ink_frac > 0.04) | (color_frac > 0.15) | (dark_frac > 0.25)

    booked_indices = np.where(booked_flags)[0]
    if len(booked_indices) == 0:
        return []

    groups = _group_consecutive(booked_indices.tolist(), gap=8)

    min_height = 4
    ranges = []
    for g in groups:
        if len(g) < min_height:
            continue
        r_start = g[0] + y_top
        r_end = g[-1] + y_top + 1
        if r_end > r_start:
            ranges.append((r_start, r_end))
    return ranges
