"""OCR time labels and column headers, build pixel-to-time mapping."""

import cv2
import numpy as np
import pytesseract


def extract_time_labels(image, x_start, x_end, y_top, y_bottom):
    """OCR a vertical region to find (y_pixel, hour) pairs.

    Returns a sorted, monotonically increasing list of (y, hour) tuples.
    """
    region = image[y_top:y_bottom, x_start:x_end]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    scale = 5
    scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    thresh = int(np.mean(scaled) * 0.7)
    _, binary = cv2.threshold(scaled, thresh, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(
        binary,
        config="--psm 4 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DICT,
    )

    raw = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            hour = int(text)
        except ValueError:
            continue
        if 6 <= hour <= 22:
            raw.append((data["top"][i] // scale + y_top, hour))

    raw.sort(key=lambda p: p[0])
    # Keep only strictly increasing y and hour values
    if not raw:
        return []
    result = [raw[0]]
    for y, hour in raw[1:]:
        if y > result[-1][0] and hour > result[-1][1]:
            result.append((y, hour))
    return result


def build_pixel_to_time(labels, start_hour=8, end_hour=20):
    """Return (mapping_fn, (y_top, y_bot)) where mapping_fn(y) -> (hour, min)."""
    if len(labels) >= 2:
        y_top, h_top = labels[0]
        y_bot, h_bot = labels[-1]
    else:
        y_top, h_top = 0, start_hour
        y_bot, h_bot = 1, end_hour

    span = max(h_bot - h_top, 1)

    def pixel_to_time(y):
        relative = (y - y_top) / max(y_bot - y_top, 1)
        total_min = int(h_top * 60 + relative * span * 60)
        total_min = max(start_hour * 60, min(total_min, end_hour * 60))
        return total_min // 60, total_min % 60

    return pixel_to_time, (y_top, y_bot)


def extract_column_headers(image, col_bounds, y_start, y_end):
    """OCR the header cell above each column. Returns list of name strings."""
    if y_end <= y_start or y_end - y_start < 5:
        return [""] * len(col_bounds)

    names = []
    for x0, x1 in col_bounds:
        margin = max(2, int((x1 - x0) * 0.04))
        cell = image[y_start:y_end, x0 + margin:x1 - margin]
        if cell.size == 0:
            names.append("")
            continue
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config="--psm 7").strip()
        text = text.strip("|-_.,;:!?")
        names.append(text if len(text) > 1 else "")
    return names


def match_column(query, col_names, col_bounds):
    """Match a query (name or 1-based index) to a column.

    Returns (index, (x_start, x_end)) or None.
    """
    # Try numeric index first
    try:
        idx = int(query) - 1
        if 0 <= idx < len(col_bounds):
            return idx, col_bounds[idx]
    except ValueError:
        pass

    q = query.lower()
    # Substring match
    for i, name in enumerate(col_names):
        if not name:
            continue
        if q in name.lower() or name.lower() in q:
            return i, col_bounds[i]

    # Fuzzy character-overlap fallback
    best_idx, best_score = None, 0.0
    for i, name in enumerate(col_names):
        if not name:
            continue
        n = name.lower()
        common = sum(1 for c in q if c in n)
        score = common / max(len(q), len(n))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is not None and best_score > 0.4:
        return best_idx, col_bounds[best_idx]
    return None
