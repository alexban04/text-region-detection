"""
plot_stages.py — Pipeline Visualisation
Person 2 — Morphology & Report Lead

Responsibilities:
  - Assemble a 2×3 grid showing every pipeline stage for one image
  - Save per-image stage grids to output/
  - Build a summary contact sheet for the demo / report
  - All functions are pure (take arrays, return arrays or write files)

Stage order visualised:
  [0] Original         [1] Enhanced
  [2] Segmentation     [3] Cleaned mask
  [4] Detection result [5] Final decision overlay
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Sequence


# ──────────────────────────────────────────────
# Colour palette (BGR)
# ──────────────────────────────────────────────

LABEL_COLOUR   = (0,   200, 255)   # amber — label text
BORDER_COLOUR  = (50,  50,  50)    # dark grey — grid border
BOX_COLOUR     = (0,   255,  0)    # green — detection bounding boxes
DECISION_FG    = (255, 255, 255)   # white — decision text
DECISION_BG    = (0,   0,    0)    # black — decision background strip
FONT           = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale or binary image to BGR for uniform compositing."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _resize_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize img to exactly (target_h, target_w) using INTER_AREA."""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _add_label(
    panel: np.ndarray,
    text: str,
    font_scale: float = 0.6,
    thickness: int = 1,
    padding: int = 6,
) -> np.ndarray:
    """Overlay a text label on the top-left corner of a BGR panel."""
    panel = panel.copy()
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    # Dark background strip for readability
    cv2.rectangle(panel, (0, 0), (tw + padding * 2, th + baseline + padding * 2), DECISION_BG, -1)
    cv2.putText(
        panel,
        text,
        (padding, th + padding),
        FONT,
        font_scale,
        LABEL_COLOUR,
        thickness,
        cv2.LINE_AA,
    )
    return panel


def _add_decision_banner(panel: np.ndarray, decision_text: str) -> np.ndarray:
    """Add a decision label at the bottom of a panel."""
    panel = panel.copy()
    h, w = panel.shape[:2]
    banner_h = 36
    cv2.rectangle(panel, (0, h - banner_h), (w, h), DECISION_BG, -1)
    cv2.putText(
        panel,
        decision_text,
        (8, h - 10),
        FONT,
        0.65,
        DECISION_FG,
        1,
        cv2.LINE_AA,
    )
    return panel


# ──────────────────────────────────────────────
# Detection overlay helper (used by Person 3 output)
# ──────────────────────────────────────────────

def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    colour: tuple[int, int, int] = BOX_COLOUR,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a copy of the image.

    Args:
        image:     BGR image to annotate.
        boxes:     List of (x, y, w, h) tuples from Person 3's detect stage.
        colour:    BGR colour for box outlines.
        thickness: Line thickness in pixels.

    Returns:
        Annotated BGR image copy.
    """
    out = _to_bgr(image).copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
    return out


# ──────────────────────────────────────────────
# Stage grid builder
# ──────────────────────────────────────────────

STAGE_LABELS = [
    "1. Original",
    "2. Enhanced",
    "3. Segmentation Mask",
    "4. Cleaned Mask",
    "5. Detection Result",
    "6. Final Decision",
]


def build_stage_grid(
    original: np.ndarray,
    enhanced: np.ndarray,
    seg_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    detection_result: np.ndarray,
    decision: dict,
    panel_w: int = 400,
    panel_h: int = 300,
    border: int = 4,
) -> np.ndarray:
    """Build a 2×3 grid image showing all six pipeline stages.

    Panels (left-to-right, top-to-bottom):
        [0] Original image
        [1] Enhanced (grayscale)
        [2] Segmentation mask (binary)
        [3] Cleaned mask (binary)
        [4] Detection result (boxes drawn on original)
        [5] Final decision overlay (boxes + decision text)

    Args:
        original:         BGR original image from Person 1.
        enhanced:         Grayscale enhanced image from Person 1.
        seg_mask:         Binary segmentation mask from Person 1.
        cleaned_mask:     Binary cleaned mask from Person 2 (clean.py).
        detection_result: BGR image with bounding boxes from Person 3.
        decision:         Dict from Person 3: {"text_found": bool, "region_count": int}.
        panel_w:          Width of each panel in pixels.
        panel_h:          Height of each panel in pixels.
        border:           Border thickness between panels in pixels.

    Returns:
        BGR grid image, shape (panel_h*2 + border*3, panel_w*3 + border*4, 3).
    """
    decision_text = (
        f"TEXT FOUND — {decision.get('region_count', 0)} region(s)"
        if decision.get("text_found")
        else "NO TEXT DETECTED"
    )

    # Build decision panel: detection result + banner
    decision_panel = _add_decision_banner(
        _to_bgr(detection_result), decision_text
    )

    raw_panels = [
        _to_bgr(original),
        _to_bgr(enhanced),
        _to_bgr(seg_mask),
        _to_bgr(cleaned_mask),
        _to_bgr(detection_result),
        decision_panel,
    ]

    panels = []
    for raw, label in zip(raw_panels, STAGE_LABELS):
        p = _resize_to(raw, panel_h, panel_w)
        p = _add_label(p, label)
        panels.append(p)

    # Build border fill colour
    fill = np.full((panel_h, border, 3), BORDER_COLOUR[0], dtype=np.uint8)
    h_border = np.full((border, panel_w * 3 + border * 4, 3), BORDER_COLOUR[0], dtype=np.uint8)

    def hstack_row(row_panels):
        parts = [fill]
        for p in row_panels:
            parts.append(p)
            parts.append(fill)
        return np.concatenate(parts, axis=1)

    row1 = hstack_row(panels[:3])
    row2 = hstack_row(panels[3:])

    grid = np.concatenate([h_border, row1, h_border, row2, h_border], axis=0)
    return grid


def save_stage_grid(grid: np.ndarray, output_path: str | Path) -> None:
    """Save the stage grid image to disk.

    Args:
        grid:        BGR grid image produced by build_stage_grid().
        output_path: Destination file path (.png recommended).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


# ──────────────────────────────────────────────
# Contact sheet (summary of all processed images)
# ──────────────────────────────────────────────

def build_contact_sheet(
    result_images: Sequence[np.ndarray],
    labels: Sequence[str],
    thumb_w: int = 320,
    thumb_h: int = 240,
    cols: int = 3,
    border: int = 6,
) -> np.ndarray:
    """Build a contact-sheet grid from a list of detection result images.

    Useful for the demo and report to show the system working across the
    full test dataset at a glance.

    Args:
        result_images: List of BGR images (detection result panels).
        labels:        Filename / image name label for each image.
        thumb_w:       Thumbnail width in pixels.
        thumb_h:       Thumbnail height in pixels.
        cols:          Number of columns in the contact sheet.
        border:        Border thickness between thumbnails.

    Returns:
        BGR contact-sheet image.
    """
    if not result_images:
        raise ValueError("result_images must not be empty")

    rows = (len(result_images) + cols - 1) // cols
    fill_colour = 30  # dark background

    sheet_h = rows * (thumb_h + border) + border
    sheet_w = cols * (thumb_w + border) + border
    sheet = np.full((sheet_h, sheet_w, 3), fill_colour, dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(result_images, labels)):
        row = idx // cols
        col = idx % cols
        y = border + row * (thumb_h + border)
        x = border + col * (thumb_w + border)

        thumb = _resize_to(_to_bgr(img), thumb_h, thumb_w)
        thumb = _add_label(thumb, label, font_scale=0.45)
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb

    return sheet


def save_contact_sheet(sheet: np.ndarray, output_path: str | Path) -> None:
    """Save the contact sheet to disk.

    Args:
        sheet:       BGR contact sheet image.
        output_path: Destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
