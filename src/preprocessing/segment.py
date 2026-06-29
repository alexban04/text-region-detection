"""
segment.py — Stage 2: Segmentation
Person 1 — Image Processing Specialist

Responsibilities:
  - Extract a binary mask that highlights text-like regions
  - Support multiple segmentation strategies (adaptive threshold, gradient, combined)
  - Pass the cleaned binary mask to Person 2 (morphology stage)

Output contract:
  seg_mask: np.ndarray  shape (H, W), dtype uint8, values in {0, 255}
"""

import cv2
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Segmentation strategies
# ──────────────────────────────────────────────

def segment_adaptive_threshold(
    enhanced: np.ndarray,
    block_size: int | None = None,
    c: int = 6,
) -> np.ndarray:
    """Segment text regions using adaptive (local) thresholding.

    Adaptive thresholding computes a threshold per pixel based on a local
    neighbourhood, making it robust to uneven lighting — common in scanned
    documents and photos of signs.

    block_size now defaults to None so it can be auto-scaled to ~2 % of
    the image's short side.  This avoids the fixed-15 value being too
    coarse for small images and too fine for large ones.

    The polarity (BINARY_INV vs BINARY) is chosen automatically: if more
    than 50 % of the mask is white after BINARY_INV, the image likely has
    light text on a dark background and we flip to BINARY instead.

    Args:
        enhanced:   Grayscale enhanced image, uint8.
        block_size: Size of the pixel neighbourhood (must be odd). None = auto.
        c:          Constant subtracted from the mean. Lowered to 6 (was 8)
                    to catch more lower-contrast text.

    Returns:
        Binary mask, uint8, {0, 255}.  255 = text candidate.
    """
    h, w = enhanced.shape[:2]
    if block_size is None:
        # Scale to ~2 % of the short side, keep odd, minimum 11
        block_size = max(11, (min(h, w) // 50) | 1)
    elif block_size % 2 == 0:
        block_size += 1  # must be odd

    mask_inv = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # dark text on light bg
        blockSize=block_size,
        C=c,
    )

    # Auto-flip: if more than half the mask is white, light text on dark bg
    if np.mean(mask_inv > 0) > 0.5:
        mask_inv = cv2.bitwise_not(mask_inv)

    return mask_inv


def segment_gradient(enhanced: np.ndarray, ksize: int = 3, threshold: int = 50) -> np.ndarray:
    """Segment text regions using the Sobel gradient magnitude.

    Text edges produce strong gradients.  This complements adaptive
    thresholding on images where text colour is similar to background
    (e.g. grey text on white paper).

    Threshold raised from 30 → 50 to reduce pickup of low-magnitude
    texture noise (fur, fabric, foliage) that mimics text edges.

    Args:
        enhanced:  Grayscale enhanced image, uint8.
        ksize:     Sobel kernel size (1, 3, 5, or 7).
        threshold: Gradient magnitude cutoff. Raised to 50 to suppress
                   texture noise while keeping sharp text edges.

    Returns:
        Binary mask, uint8, {0, 255}.  255 = strong-gradient (edge) region.
    """
    grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    _, mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    return mask


def segment_combined(
    enhanced: np.ndarray,
    block_size: int | None = None,
    c: int = 6,
    grad_ksize: int = 3,
    grad_threshold: int = 50,
) -> np.ndarray:
    """Combine adaptive threshold, gradient, and Otsu masks with a logical OR.

    Three complementary methods cover different text scenarios:
      - Adaptive threshold: uneven lighting, scanned docs, overlaid text.
      - Gradient magnitude: edges of text on busy backgrounds.
      - Otsu global threshold: high-contrast text (white on dark, dark on
        white) that adaptive thresholding may miss.

    Args:
        enhanced:       Grayscale enhanced image, uint8.
        block_size:     Adaptive threshold block size. None = auto-scale.
        c:              Adaptive threshold C constant.
        grad_ksize:     Sobel kernel size.
        grad_threshold: Gradient magnitude cutoff (raised to 50).

    Returns:
        Combined binary mask, uint8, {0, 255}.
    """
    mask_thresh = segment_adaptive_threshold(enhanced, block_size=block_size, c=c)
    mask_grad = segment_gradient(enhanced, ksize=grad_ksize, threshold=grad_threshold)

    # Otsu: catches high-contrast text (e.g. white logo on dark background)
    _, mask_otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask_otsu_inv = cv2.bitwise_not(mask_otsu)
    # Otsu polarity: keep the variant where foreground (white in mask) matches
    # dark pixels in the original image.  Dark text on light background is the
    # common case; we keep mask_otsu (BINARY) when the mean of the image where
    # mask_otsu is white is darker than where it is black — i.e. mask_otsu marks
    # the darker pixels as foreground.  This correctly handles the МАЛА ТОКМАЧКА
    # case where the sign letters are dark on a white board.
    mean_fg = float(np.mean(enhanced[mask_otsu > 0])) if np.any(mask_otsu > 0) else 128.0
    mean_bg = float(np.mean(enhanced[mask_otsu == 0])) if np.any(mask_otsu == 0) else 128.0
    if mean_fg <= mean_bg:
        # mask_otsu marks darker pixels — dark text on light bg: correct
        mask_otsu_best = mask_otsu
    else:
        # mask_otsu marks lighter pixels — flip so darker pixels are foreground
        mask_otsu_best = mask_otsu_inv

    combined = cv2.bitwise_or(mask_thresh, mask_grad)
    combined = cv2.bitwise_or(combined, mask_otsu_best)
    return combined


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

STRATEGIES = {
    "adaptive": segment_adaptive_threshold,
    "gradient": segment_gradient,
    "combined": segment_combined,
}


def segment(
    enhanced: np.ndarray,
    strategy: str = "combined",
    **kwargs,
) -> np.ndarray:
    """Segment an enhanced grayscale image into a binary text mask.

    Args:
        enhanced: Grayscale enhanced image from the enhance stage, uint8.
        strategy: One of "adaptive", "gradient", or "combined".
                  "combined" is recommended for general use.
        **kwargs: Extra keyword arguments forwarded to the chosen strategy
                  function (e.g. block_size=11, c=5).

    Returns:
        Binary mask, uint8, {0, 255}.  255 marks text-candidate pixels.

    Raises:
        ValueError: If an unknown strategy name is supplied.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}"
        )
    return STRATEGIES[strategy](enhanced, **kwargs)


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    """Save the binary segmentation mask to disk.

    Args:
        mask:        Binary mask (uint8, {0, 255}) to save.
        output_path: Destination path (.png recommended for lossless storage).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
