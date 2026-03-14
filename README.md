# Slot Checker

A Python tool to check time-slot availability for practitioners in scheduling grid images using OCR and image processing.

## Overview

Slot Checker analyzes scheduling grid images to determine whether a requested time slot is available for a specific practitioner. It uses:

- **Image processing** (OpenCV) to detect grid structure and booked appointment blocks
- **OCR** (Tesseract) to extract time labels and practitioner names
- **Availability logic** to check for conflicts with existing bookings

## Installation

### Prerequisites

- Python 3.7+
- Tesseract OCR engine (system-level installation required)

### Tesseract Installation

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python pytesseract numpy Pillow
```

## Usage

```bash
python main.py <image> <practitioner> <HH:MM-HH:MM>
```

### Arguments

- `<image>` — Path to the scheduling grid image file
- `<practitioner>` — Name of the practitioner (or column number if name matching fails)
- `<HH:MM-HH:MM>` — Requested time range in 24-hour format (e.g., `11:00-11:30`)

### Example

```bash
python main.py schedule.png "Dre Yasmin Bouzaza" 11:00-11:30
```

### Output

- `available` — If the requested time slot is free
- `not available` — If the requested time slot overlaps with a booked appointment

If the practitioner is not found, the tool lists available columns:

```
Error: 'Name' not found.
  1. Dre Yasmin Bouzaza
  2. Dr John Smith
  ...
```

## Module Reference

### `main.py`

Entry point that orchestrates the workflow:

1. Parses command-line arguments
2. Detects grid boundaries and column layout
3. Separates time-label column from practitioner columns
4. Extracts OCR data and checks availability
5. Prints result

### `image_processor.py`

Handles image loading and grid detection:

**Key Functions:**

- `load_image(path)` — Load image from file as BGR numpy array
- `find_grid_area(image)` — Detect green-bordered scheduling grid and return boundaries `(x_left, x_right, y_top, y_bottom)`
- `find_columns(image, x_left, x_right, y_top, y_bottom)` — Find column boundaries by detecting vertical separator lines
- `detect_booked_blocks(image, y_top, y_bottom, x_start, x_end)` — Detect booked appointment rectangles
  - **Free slots:** Yellow, desaturated colors (white/grey/peach), green borders
  - **Booked slots:** Saturated non-yellow/non-green colors (pink, orange, teal, etc.)

### `ocr_reader.py`

Performs OCR on grid elements:

**Key Functions:**

- `extract_time_labels(image, x_start, x_end, y_top, y_bottom)` — OCR vertical time column to extract `(y_pixel, hour)` pairs
  - Uses Tesseract with digit-only whitelist
  - Returns sorted, monotonically increasing list of labels
  - Recognizes hours 6-22
- `build_pixel_to_time(labels, start_hour=8, end_hour=20)` — Create pixel-to-time mapping function
  - Linear interpolation between labeled time points
  - Returns `(mapping_fn, (y_top, y_bot))`
- `extract_column_headers(image, col_bounds, y_start, y_end)` — OCR header cells to extract practitioner names
  - 4x upscaling for better recognition
  - Strips boundary characters (`|-_.,;:!?`)
- `match_column(query, col_names, col_bounds)` — Match practitioner name (or 1-based column index) to a column
  - Returns `(index, (x_start, x_end))` or `None`

### `availability_checker.py`

Checks time-slot availability:

**Key Functions:**

- `parse_time_range(text)` — Parse `'HH:MM-HH:MM'` string into two `(hour, minute)` tuples
  - Validates 24-hour time format
  - Supports single-digit hours (e.g., `9:00`)
- `check_availability(booked_slots, req_start, req_end)` — Check if requested range overlaps any booked slot
  - `booked_slots` — List of `((start_h, start_m), (end_h, end_m))` tuples
  - `req_start`, `req_end` — `(hour, minute)` tuples from `parse_time_range()`
  - Returns `'available'` or `'not available'`

## Grid Image Format

The tool expects scheduling grid images with:

- **Green borders** defining the grid boundaries
- **Vertical separators** dividing columns (leftmost is time labels)
- **Horizontal separators** between time rows
- **Column headers** (first ~25px below the grid top) with practitioner names
- **Colored blocks** indicating booked appointments:
  - Yellow/desaturated = free slots
  - Saturated colors (pink, orange, teal, etc.) = booked slots
- **Time labels** in the leftmost column (hours as digits, e.g., "8", "9", ..., "22")

## Requirements

See `requirements.txt`:

```
opencv-python
pytesseract
numpy
Pillow
```

## Troubleshooting

### "Error: could not detect columns"

- Verify the image has clear vertical separators
- Ensure grid boundaries are properly detected (green borders)

### "Error: '<name>' not found"

- Check practitioner name spelling
- Use the listed column numbers (1-indexed) instead
- Name matching uses fuzzy matching; very unusual spellings may fail

### Tesseract not found

- Ensure Tesseract is installed system-wide
- On Linux, may need: `sudo apt-get install tesseract-ocr`
- Update `pytesseract.pytesseract.pytesseract_cmd` if Tesseract is in a non-standard location

### Poor OCR results

- Ensure image quality is high
- Time column text should be clearly visible
- Column headers (practitioner names) should be legible at the top of columns

## License

This project is provided as-is for educational and instructional purposes.
