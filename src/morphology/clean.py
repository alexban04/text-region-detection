"""
clean.py — Stage 3: Morphological Cleaning
Person 2 — Morphology & Report Lead

Responsibilities:
  - Remove small noise blobs from the segmentation mask
  - Fill gaps inside text regions (dilation / closing)
  - Filter connected components by size to suppress false positives
  - Pass a clean binary mask to Person 3 (detection stage)

Input contract  (from Person 1):
  seg_mask: np.ndarray  shape (H, W), dtype uint8, values in {0, 255}

Output contract (to Person 3):
  cleaned_mask: np.ndarray  shape (H, W), dtype uint8, values in {0, 255}
"""

import cv2
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Kernel helpers
# ──────────────────────────────────────────────

def _rect_kernel(w: int, h: int) -> np.ndarray:
    """Return a rectangular structuring element of size (w, h)."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))


def _ellipse_kernel(w: int, h: int) -> np.ndarray:
    """Return an elliptical structuring element of size (w, h)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))


# ──────────────────────────────────────────────
# Individual morphological operations
# ──────────────────────────────────────────────

def erode(mask: np.ndarray, ksize: int = 2, iterations: int = 1) -> np.ndarray:
    """Shrink white regions — removes thin noise strands and isolated pixels.

    Erosion is applied first to eliminate salt-and-pepper noise and thin
    artefacts that are not text strokes.

    Args:
        mask:       Binary mask, uint8, {0, 255}.
        ksize:      Square kernel side length.
        iterations: Number of erosion passes.

    Returns:
        Eroded binary mask, uint8, {0, 255}.
    """
    kernel = _rect_kernel(ksize, ksize)
    return cv2.erode(mask, kernel, iterations=iterations)


def dilate(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Expand white regions — reconnects broken character strokes.

    After erosion, dilation restores the size of real text regions while
    keeping noise blobs (which were destroyed by erosion) suppressed.

    Args:
        mask:       Binary mask, uint8, {0, 255}.
        ksize:      Square kernel side length.
        iterations: Number of dilation passes.

    Returns:
        Dilated binary mask, uint8, {0, 255}.
    """
    kernel = _rect_kernel(ksize, ksize)
    return cv2.dilate(mask, kernel, iterations=iterations)


def opening(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Erosion followed by dilation — removes small isolated noise blobs.

    Opening is the standard first step for cleaning a noisy binary mask.
    It eliminates pixels smaller than the kernel without significantly
    shrinking larger text regions.

    Args:
        mask:  Binary mask, uint8, {0, 255}.
        ksize: Square kernel side length.

    Returns:
        Opened binary mask, uint8, {0, 255}.
    """
    kernel = _rect_kernel(ksize, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def closing(mask: np.ndarray, kw: int | None = None, kh: int | None = None) -> np.ndarray:
    """Dilation followed by erosion — fills small holes inside text blobs.

    A wide horizontal kernel (kw >> kh) merges individual characters
    within the same text line into a single connected region, making
    it easier for Person 3 to detect word/line bounding boxes.

    kw and kh default to None so the clean() function can scale them
    automatically to the image size. If explicit values are passed they
    are used as-is (legacy behaviour).

    Args:
        mask: Binary mask, uint8, {0, 255}.
        kw:   Kernel width  (horizontal extent). None = auto (5% of image width).
        kh:   Kernel height (vertical extent).  None = auto (1.25% of image height).

    Returns:
        Closed binary mask, uint8, {0, 255}.
    """
    h, w = mask.shape[:2]
    if kw is None:
        kw = max(15, w // 20)   # ~5 % of image width, minimum 15 px
    if kh is None:
        kh = max(3, h // 80)    # ~1.25 % of image height, minimum 3 px
    kernel = _rect_kernel(kw, kh)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ──────────────────────────────────────────────
# Connected-component filtering
# ──────────────────────────────────────────────

def filter_components(
    mask: np.ndarray,
    min_area: int = 50,
    max_area: int | None = None,
    min_width: int = 5,
    min_height: int = 5,
    max_aspect_ratio: float = 30.0,
    max_width_ratio: float = 0.95,
) -> np.ndarray:
    """Remove connected components that are unlikely to be text regions.

    Text blobs have characteristic size and shape constraints.  Components
    that are too small (single pixels, dust), too large (whole-page blobs),
    too elongated (horizontal lines, borders), or span nearly the full image
    width (desk edges, horizon lines, letterbox bars) are suppressed.

    Args:
        mask:             Binary mask, uint8, {0, 255}.
        min_area:         Minimum pixel area to keep (removes dust / noise).
        max_area:         Maximum pixel area to keep.  None = no upper limit.
        min_width:        Minimum bounding-box width in pixels.
        min_height:       Minimum bounding-box height in pixels.
        max_aspect_ratio: Maximum width/height ratio.  30.0 keeps wide sign
                          banners while still rejecting ruled lines.
        max_width_ratio:  Maximum box width as a fraction of image width.
                          Blobs spanning >= 88 % of the image width are almost
                          certainly background edges, not text.

    Returns:
        Filtered binary mask, uint8, {0, 255}.
    """
    img_h, img_w = mask.shape[:2]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)
    for label in range(1, n_labels):  # label 0 is background
        x, y, w, h, area = (
            stats[label, cv2.CC_STAT_LEFT],
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_WIDTH],
            stats[label, cv2.CC_STAT_HEIGHT],
            stats[label, cv2.CC_STAT_AREA],
        )

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if w < min_width or h < min_height:
            continue
        aspect = w / h if h > 0 else float("inf")
        if aspect > max_aspect_ratio:
            continue
        # Kill blobs that span nearly the full image width — these are
        # background edges (desk surface, horizon, letterbox bars), not text.
        if img_w > 0 and (w / img_w) > max_width_ratio:
            continue
        # Kill blobs that span most of the image height — these are
        # person silhouettes or large foreground objects, not text.
        if img_h > 0 and (h / img_h) > 0.60:
            continue

        out[labels == label] = 255

    return out


# ──────────────────────────────────────────────
# Full cleaning pipeline
# ──────────────────────────────────────────────

def clean(
    seg_mask: np.ndarray,
    open_ksize: int = 5,
    erode_ksize: int = 2,
    erode_iters: int = 1,
    dilate_ksize: int = 3,
    dilate_iters: int = 2,
    close_kw: int | None = None,
    close_kh: int | None = None,
    min_area: int = 50,
    max_area: int | None = None,
    min_width: int = 5,
    min_height: int = 5,
    max_aspect_ratio: float = 30.0,
) -> np.ndarray:
    """Full morphological cleaning pipeline for a segmentation mask.

    Pipeline:
        opening → erosion → dilation → closing → component filtering

    Steps:
        1. **Opening**  — removes isolated noise specks (kernel raised to 5
                          to kill fur/texture speckles that fool the gradient).
        2. **Erosion**  — thins remaining noise without destroying text strokes.
        3. **Dilation** — restores text stroke width after erosion.
        4. **Closing**  — merges nearby characters into word/line blobs.
                          Kernel is now scaled to image size so wide text
                          (e.g. sign banners, bold headlines) gets merged too.
        5. **Component filter** — discards blobs that are too small, too
                          large (> 15 % of image area), or too elongated.
                          max_aspect_ratio raised to 30 so wide sign banners
                          and subtitle bars survive.

    Args:
        seg_mask:         Binary mask from Person 1 (Stage 2 output).
        open_ksize:       Opening kernel size (default raised to 5).
        erode_ksize:      Erosion kernel size.
        erode_iters:      Number of erosion iterations.
        dilate_ksize:     Dilation kernel size.
        dilate_iters:     Number of dilation iterations.
        close_kw:         Closing kernel width. None = auto (5 % of image width).
        close_kh:         Closing kernel height. None = auto (1.25 % of image height).
        min_area:         Minimum component area to keep.
        max_area:         Maximum component area to keep.
                          None = auto (15 % of image area) to eliminate large
                          background blobs (e.g. a character's body).
        min_width:        Minimum component bounding-box width.
        min_height:       Minimum component bounding-box height.
        max_aspect_ratio: Maximum width/height ratio for kept components
                          (raised to 30 to keep wide sign/banner text).

    Returns:
        Cleaned binary mask, uint8, {0, 255}.
    """
    h, w = seg_mask.shape[:2]
    image_area = h * w

    # Auto max_area: cap blobs at 70 % of image.
    if max_area is None:
        max_area = int(image_area * 0.70)

    # Step 1: light opening to remove salt-and-pepper noise.
    # Always use ksize=3 here; the heavier ksize=5 caused over-erosion on small text.
    mask = opening(seg_mask, ksize=3)

    # Step 2: PRE-filter - kill tall blobs (person silhouettes, large objects)
    # and geometry-failing components BEFORE measuring density or running erosion.
    # This ensures decisions are based on text-like content, not the body blob.
    mask = filter_components(
        mask,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_height=min_height,
        max_aspect_ratio=max_aspect_ratio,
    )

    # Step 3: measure density NOW (after body removed) to decide erosion strategy.
    #   Dense (> 30%): large bold text -> skip erosion (destroys letter interiors),
    #                  single dilation only.
    #   Sparse (<= 30%): normal/noisy mask -> full erode+dilate cycle.
    _mask_density = float(np.mean(mask > 0))

    if _mask_density >= 0.30:
        # Bold/large text: skip erosion, one dilation pass to connect strokes
        mask = dilate(mask, ksize=dilate_ksize, iterations=1)
    else:
        # Normal text or noisy mask: full erosion+dilation cycle
        mask = erode(mask, ksize=erode_ksize, iterations=erode_iters)
        mask = dilate(mask, ksize=dilate_ksize, iterations=dilate_iters)

    # Adaptive closing kernel: if a single component already covers > 25% of
    # the image after the pre-close filter (e.g. a person's silhouette or a
    # large background object), shrink the closing kernel dramatically so it
    # cannot bridge the gap between that blob and nearby text regions.
    _n, _labels, _stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    _largest_frac = 0.0
    _any_wide_blob = False
    for _lbl in range(1, _n):
        _frac = _stats[_lbl, cv2.CC_STAT_AREA] / image_area
        _bw = int(_stats[_lbl, cv2.CC_STAT_WIDTH])
        if _frac > _largest_frac:
            _largest_frac = _frac
        if _bw / w > 0.60:
            _any_wide_blob = True

    _effective_close_kw = close_kw
    _effective_close_kh = close_kh

    # Decide whether a large/over-merged blob is present that would cause the
    # default wide closing kernel to swallow text regions into background blobs.
    _large_blob_present = (_largest_frac > 0.15) or _any_wide_blob

    if _large_blob_present and close_kw is None and close_kh is None:
        # Two-pass closing strategy for images with large foreground objects
        # or already-wide blobs (meme text on busy background, person photo):
        #
        # Pass 1 — small SQUARE kernel: fills gaps *within* each character's
        #   stroke (e.g. reconnects the outline pixels of bold letters like
        #   'БАЗАРА' / 'НЕТ') without bridging across character spacing.
        #
        # Pass 2 — narrow HORIZONTAL kernel: joins adjacent characters in the
        #   same word.  Width is capped at w//30 (~3% of image) so it cannot
        #   reach across a half-image gap to merge with a person's body or
        #   with fence-texture noise on the other side.
        _sq = max(3, min(w, h) // 80)   # ~1.25% of short side, min 3 px
        mask = closing(mask, kw=_sq, kh=_sq)

        _effective_close_kw = max(7, w // 30)   # ~3% image width, min 7 px
        _effective_close_kh = max(2, h // 100)  # ~1% image height, min 2 px

    # close_kw/kh=None triggers auto-scaling inside closing()
    mask = closing(mask, kw=_effective_close_kw, kh=_effective_close_kh)

    # POST-CLOSE filter: remove anything that ballooned past max_area after merging
    mask = filter_components(
        mask,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_height=min_height,
        max_aspect_ratio=max_aspect_ratio,
    )
    return mask


def save_cleaned_mask(mask: np.ndarray, output_path: str | Path) -> None:
    """Save the cleaned binary mask to disk.

    Args:
        mask:        Cleaned binary mask (uint8, {0, 255}).
        output_path: Destination path (.png recommended for lossless output).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
