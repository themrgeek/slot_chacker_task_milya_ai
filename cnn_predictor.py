"""CNN-based slot classification for dental scheduling grids."""

import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = 64


def load_cnn_model(path="slot_classifier_cnn.keras"):
    """Load a trained Keras CNN model from disk."""
    return keras.models.load_model(path)


def predict_slot_cell(cell_bgr, model, img_size=IMG_SIZE):
    """Classify a single slot cell (BGR numpy array) as FREE or BOOKED."""
    cell_resized = cv2.resize(cell_bgr, (img_size, img_size))
    cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)
    cell_norm = cell_rgb.astype(np.float32) / 255.0
    cell_batch = np.expand_dims(cell_norm, axis=0)

    preds = model.predict(cell_batch, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    label = "FREE" if class_idx == 0 else "BOOKED"
    return label, confidence


def estimate_row_height(y_top, y_bottom, expected_hours=13):
    """Pixel height per hour-row based on grid span (typically 7h-20h = 13 rows)."""
    return (y_bottom - y_top) // expected_hours


def classify_column_slots(image, model, col_bounds, content_y_top, y_bottom, row_h):
    """Classify every row-cell in a single practitioner column.

    Returns a list of dicts with keys:
        row, y_top, y_bot, label, confidence
    """
    cx0, cx1 = col_bounds
    margin = 3
    y = content_y_top
    row_idx = 0
    results = []

    while y + row_h <= y_bottom:
        cell = image[y:y + row_h, cx0 + margin:cx1 - margin]
        if cell.size == 0:
            y += row_h
            row_idx += 1
            continue

        label, conf = predict_slot_cell(cell, model)
        results.append({
            "row": row_idx,
            "y_top": y,
            "y_bot": y + row_h,
            "label": label,
            "confidence": conf,
        })

        y += row_h
        row_idx += 1

    return results
