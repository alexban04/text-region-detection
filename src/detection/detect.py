"""
detect.py — Stage 4: Detection
Person 3 — Lead CV Engineer

Responsibilities:
  - Find contours on the cleaned binary mask (Person 2 output)
  - Convert contours to (x, y, w, h) bounding boxes
  - Filter boxes by geometry (min area, aspect ratio, overlap)
  - Return a clean list of text-region bounding boxes

Input contract  (from Person 2):
  cleaned_mask: np.ndarray  shape (H, W), dtype uint8, values in {0, 255}

Output contract (to decide.py and visualization):
  boxes: list[tuple[int, int, int, int]]  — list of (x, y, w, h)
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Contour → box helpers
# ──────────────────────────────────────────────

def _contours_to_boxes(contours) -> list:
    """Convert a sequence of OpenCV contours to (x, y, w, h) tuples."""
    return [cv2.boundingRect(c) for c in contours]


def _box_area(box):
    return box[2] * box[3]


def _box_aspect(box):
    _, _, w, h = box
    return w / h if h > 0 else float("inf")


def _iou(a, b):
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────

def _box_centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _distance(a, b):
    ax, ay = _box_centre(a)
    bx, by = _box_centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def filter_isolated(boxes, image_area, neighbour_radius_factor=10.0, small_area_ratio=0.03):
    """Remove small boxes that have no neighbours nearby.

    Noise blobs that survive geometry filtering tend to be spatially isolated —
    scattered across the image with no other boxes nearby.  Real text regions
    appear in clusters (word next to word, line above line) — but "cluster"
    can mean spread across the full image (e.g. two overlay captions placed
    in opposite corners).

    A box is considered *isolated* if:
      - Its area is below small_area_ratio * image_area  (it's small), AND
      - The total box count is > 2, AND
      - No other box has its centre within neighbour_radius_factor * box_height
        of this box's centre.

    When only 2 boxes survive geometry filtering they are almost certainly both
    real text (two noise blobs both surviving geometry filters is very unlikely),
    so the filter is skipped entirely in that case.

    neighbour_radius_factor raised from 3 to 10: overlay text captions placed
    in different parts of the image (e.g. top-right and left-middle) can be
    far apart in absolute pixels yet both be real text.  A factor of 3x the
    box height was too short to connect them; 10x covers the typical distances
    seen in social-media overlay text without re-introducing noise.

    Large boxes (>= small_area_ratio of image) are never removed regardless
    of isolation.

    Args:
        boxes:                  List of (x, y, w, h) boxes.
        image_area:             H * W of the original image.
        neighbour_radius_factor: Distance multiplier relative to box height.
        small_area_ratio:       Boxes smaller than this fraction of image_area
                                are candidates for isolation removal.

    Returns:
        Filtered list of boxes.
    """
    if len(boxes) <= 2:
        return boxes  # 0-2 boxes: skip filter — not enough boxes for isolation logic

    small_threshold = image_area * small_area_ratio
    result = []
    for i, box in enumerate(boxes):
        area = _box_area(box)
        if area >= small_threshold:
            result.append(box)  # large boxes always kept
            continue

        _, _, bw, bh = box
        # Use height only for the proximity radius.
        # Using max(h,w) caused wide noise boxes to claim huge radii and
        # find distant real-text boxes as "neighbours", keeping them alive.
        radius = neighbour_radius_factor * bh
        has_neighbour = any(
            j != i and _distance(box, boxes[j]) < radius
            for j in range(len(boxes))
        )
        if has_neighbour:
            result.append(box)
        # else: isolated small blob — discard as noise

    return result


def filter_boxes(
    boxes,
    min_area=200,
    max_area=None,
    min_width=5,
    min_height=5,
    max_aspect_ratio=20.0,
    max_width_ratio=0.95,
    nms_iou_threshold=0.5,
    image_shape=None,
):
    """Remove bounding boxes that are unlikely to be text regions.

    Applies three filters in order:
      1. Geometry filter       — drop boxes too small, too large, too thin,
                                 or spanning nearly the full image width.
      2. Non-Maximum Suppression (NMS) — keep the larger of two heavily
                                 overlapping boxes.
      3. Isolated noise filter — remove small boxes with no nearby neighbours.

    Args:
        boxes:             List of (x, y, w, h) boxes.
        min_area:          Minimum box area in pixels.
        max_area:          Maximum box area. None = no upper limit.
        min_width:         Minimum box width in pixels.
        min_height:        Minimum box height in pixels.
        max_aspect_ratio:  Maximum w/h ratio.
        max_width_ratio:   Maximum box width as fraction of image width.
                           Boxes spanning >= this fraction are background
                           edges (desk surface, horizon), not text.
        nms_iou_threshold: IoU threshold above which a smaller overlapping
                           box is suppressed.
        image_shape:       (H, W) or (H, W, C) — needed for width-ratio and
                           isolated-noise filters. Inferred from boxes if None.

    Returns:
        Filtered list of (x, y, w, h) bounding boxes sorted in reading order
        (top-to-bottom, left-to-right).
    """
    if not boxes:
        return []

    # Determine image dimensions for ratio-based filters
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
    else:
        # Estimate from the union of all boxes (conservative fallback)
        img_w = max(x + w for x, y, w, h in boxes)
        img_h = max(y + h for x, y, w, h in boxes)
    image_area = img_h * img_w

    # 1. Geometry filter
    kept = []
    for box in boxes:
        x, y, w, h = box
        if _box_area(box) < min_area:
            continue
        if max_area is not None and _box_area(box) > max_area:
            continue
        if w < min_width or h < min_height:
            continue
        if _box_aspect(box) > max_aspect_ratio:
            continue
        # Reject blobs spanning nearly full image width (desk edges, horizon bars)
        if img_w > 0 and (w / img_w) > max_width_ratio:
            continue
        kept.append(box)

    # 2. NMS — sort by area descending, suppress smaller overlapping boxes
    kept.sort(key=_box_area, reverse=True)
    suppressed = [False] * len(kept)
    for i in range(len(kept)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(kept)):
            if not suppressed[j] and _iou(kept[i], kept[j]) > nms_iou_threshold:
                suppressed[j] = True

    kept = [b for b, s in zip(kept, suppressed) if not s]

    # 3. Isolated noise filter — remove small boxes with no nearby neighbours
    kept = filter_isolated(kept, image_area)

    kept.sort(key=lambda b: (b[1], b[0]))
    return kept


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def find_text_boxes(
    cleaned_mask,
    min_area=None,
    max_area=None,
    min_width=5,
    min_height=5,
    max_aspect_ratio=20.0,
    max_width_ratio=0.95,
    nms_iou_threshold=0.5,
):
    """Detect text regions in a cleaned binary mask.

    Pipeline:
        find contours → convert to boxes → geometry filter → NMS
        → isolated-noise filter

    Args:
        cleaned_mask:      Binary mask from clean.py, shape (H, W), uint8.
        min_area:          Minimum bounding-box area. None = 0.05 % of image
                           area (scales with image resolution).
        max_area:          Maximum bounding-box area. None = 40 % of image area.
        min_width:         Minimum box width in pixels.
        min_height:        Minimum box height in pixels.
        max_aspect_ratio:  Maximum width/height ratio.
        max_width_ratio:   Maximum box width as fraction of image width.
                           Boxes >= 88 % wide are background edges, not text.
        nms_iou_threshold: IoU threshold for NMS.

    Returns:
        List of (x, y, w, h) bounding boxes in reading order.
        Empty list if no text regions are found.
    """
    h, w = cleaned_mask.shape[:2]
    image_area = h * w

    if min_area is None:
        min_area = max(200, int(image_area * 0.0005))  # 0.05 % of image
    if max_area is None:
        max_area = int(image_area * 0.70)  # 70 % of image area — keeps large bold text

    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    raw_boxes = _contours_to_boxes(contours)
    return filter_boxes(
        raw_boxes,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_height=min_height,
        max_aspect_ratio=max_aspect_ratio,
        max_width_ratio=max_width_ratio,
        nms_iou_threshold=nms_iou_threshold,
        image_shape=(h, w),
    )