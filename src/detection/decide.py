"""
decide.py — Stage 5: Decision Logic
Person 3 — Lead CV Engineer

Responsibilities:
  - Interpret detected bounding boxes to produce a final verdict
  - Classify each detected region (word / line / block)
  - Return a structured decision dict
  - Save per-image JSON result files

Output contract (to visualisation and main.py):
  decision: dict — see make_decision() docstring for schema
"""

import json
from pathlib import Path


# ──────────────────────────────────────────────
# Region classification
# ──────────────────────────────────────────────

def classify_region(box, word_max_width=120):
    _, _, w, h = box
    aspect = w / h if h > 0 else 1.0

    if w <= word_max_width:
        return "word"
    if aspect >= 3.0:
        return "line"
    return "block"


# ──────────────────────────────────────────────
# Decision
# ──────────────────────────────────────────────

def make_decision(
    boxes,
    image_shape,
    min_regions=1,
    coverage_threshold=0.01,
):
    """Produce the final automatic decision for one image.

    text_found = True  if at least min_regions boxes exist AND their combined
                       area covers >= coverage_threshold of the image area.

    Args:
        boxes:               List of (x, y, w, h) bounding boxes from detect.py.
        image_shape:         (H, W) or (H, W, C) shape of the original image.
        min_regions:         Minimum box count to declare text found.
        coverage_threshold:  Minimum coverage fraction.

    Returns:
        dict with keys:
          text_found     (bool)
          region_count   (int)
          total_area     (int)   — sum of box areas in pixels
          image_area     (int)   — H * W
          coverage       (float) — total_area / image_area
          regions        (list)  — per-box dicts: {x, y, w, h, area, label}
          verdict        (str)   — human-readable summary string
    """
    H, W = image_shape[:2]
    image_area = H * W

    regions = []
    total_area = 0
    for box in boxes:
        x, y, w, h = box
        area = w * h
        total_area += area
        regions.append({
            "x":     int(x),
            "y":     int(y),
            "w":     int(w),
            "h":     int(h),
            "area":  int(area),
            "label": classify_region(box),
        })

    coverage = total_area / image_area if image_area > 0 else 0.0
    text_found = len(boxes) >= min_regions and coverage >= coverage_threshold

    if text_found:
        labels = [r["label"] for r in regions]
        type_summary = ", ".join(
            f"{labels.count(t)} {t}(s)"
            for t in ("word", "line", "block")
            if labels.count(t) > 0
        )
        verdict = f"TEXT FOUND — {len(regions)} region(s) [{type_summary}]"
    else:
        verdict = "NO TEXT DETECTED"

    return {
        "text_found":   text_found,
        "region_count": len(regions),
        "total_area":   int(total_area),
        "image_area":   int(image_area),
        "coverage":     round(coverage, 4),
        "regions":      regions,
        "verdict":      verdict,
    }


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def save_decision(decision, output_path):
    """Serialise a decision dict to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)


def load_decision(path):
    """Load a previously saved decision JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
