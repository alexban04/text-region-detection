"""
enhance.py — Stage 1: Image Enhancement
Person 1 — Image Processing Specialist

Responsibilities:
  - Load input images
  - Apply CLAHE, gamma correction, denoising
  - Convert to grayscale ready for segmentation
  - Save enhanced output image

Output contract:
  enhanced_img: np.ndarray  shape (H, W), dtype uint8, grayscale
"""

import cv2
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Core enhancement functions
# ──────────────────────────────────────────────

def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR format.

    Args:
        image_path: Path to the input image file.

    Returns:
        BGR image as np.ndarray (H, W, 3), uint8.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Handles both colour (3-channel) and already-greyscale (1-channel) inputs.

    Args:
        img: BGR image, shape (H, W, 3) or (H, W).

    Returns:
        Grayscale image, shape (H, W), uint8.
    """
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gamma_correction(gray: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Brighten or darken the image with gamma correction.

    gamma > 1  →  brighter (helps faint text on dark backgrounds)
    gamma < 1  →  darker

    Args:
        gray:  Grayscale image, uint8.
        gamma: Gamma value. Default 1.2 lifts mid-tones slightly.

    Returns:
        Gamma-corrected grayscale image, uint8.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, table)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE equalises local contrast while suppressing noise amplification —
    essential for images with uneven lighting (shadows, glare, poor scans).

    Args:
        gray:       Grayscale image, uint8.
        clip_limit: Threshold for contrast limiting. Higher = more contrast.
        tile_size:  Grid size for local histogram computation. Must divide
                    the image dimensions cleanly; 8 is a safe default.

    Returns:
        CLAHE-enhanced grayscale image, uint8.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def apply_denoising(gray: np.ndarray, h: int = 10) -> np.ndarray:
    """Remove noise using Non-Local Means Denoising.

    Preserves text edges better than Gaussian blur because it averages
    patches with similar texture rather than just nearby pixels.

    Args:
        gray: Grayscale image, uint8.
        h:    Filter strength. 10 is a balanced default.
              Higher values remove more noise but may blur fine text.

    Returns:
        Denoised grayscale image, uint8.
    """
    return cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=7, searchWindowSize=21)


def enhance(
    image_path: str | Path,
    gamma: float | None = None,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    denoise_h: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Full enhancement pipeline for a single image.

    Pipeline:
        load → grayscale → (adaptive) gamma correction → (conditional) CLAHE → denoising

    Changes from original:
      - gamma now defaults to None for automatic selection: dark images (mean
        brightness < 100) get gamma=1.4 to lift shadows; bright images skip
        correction (gamma=1.0) to avoid washing out the contrast CLAHE needs.
      - CLAHE is skipped on images that already have high contrast (std > 60)
        because applying it amplifies texture noise (fur, fabric, foliage) to
        the same magnitude as text edges, confusing the gradient segmenter.
      - denoise_h lowered from 10 → 5: h=10 blurs fine text strokes enough
        that the gradient mask misses them entirely.

    Args:
        image_path: Path to the input image.
        gamma:      Gamma value, or None for automatic selection.
        clahe_clip: CLAHE clip limit (only applied on low-contrast images).
        clahe_tile: CLAHE tile grid size.
        denoise_h:  Non-local means denoising strength (lowered to 5).

    Returns:
        Tuple of (original_bgr, enhanced_gray):
            original_bgr:  Original BGR image, shape (H, W, 3).
            enhanced_gray: Final enhanced grayscale image, shape (H, W).
    """
    original = load_image(image_path)
    gray = to_grayscale(original)

    # Adaptive gamma: only brighten dark images
    if gamma is None:
        mean_brightness = float(np.mean(gray))
        gamma = 1.4 if mean_brightness < 100 else 1.0

    gamma_corrected = apply_gamma_correction(gray, gamma=gamma)

    # Skip CLAHE on already high-contrast images to avoid amplifying texture
    contrast = float(np.std(gamma_corrected))
    if contrast < 60:
        clahe_out = apply_clahe(gamma_corrected, clip_limit=clahe_clip, tile_size=clahe_tile)
    else:
        clahe_out = gamma_corrected

    denoised = apply_denoising(clahe_out, h=denoise_h)
    return original, denoised


def save_enhanced(enhanced: np.ndarray, output_path: str | Path) -> None:
    """Save the enhanced grayscale image to disk.

    Args:
        enhanced:    Grayscale image to save.
        output_path: Destination file path (.png recommended).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)
