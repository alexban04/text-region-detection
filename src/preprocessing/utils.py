"""
utils.py — Shared Preprocessing Utilities
Person 1 — Image Processing Specialist

Helper functions used by both enhance.py and segment.py:
  - Batch processing over a folder of images
  - Side-by-side debug visualisation
  - Image validation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Iterator

from .enhance import enhance, save_enhanced
from .segment import segment, save_mask

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def is_valid_image_path(path: Path) -> bool:
    """Return True if the path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def validate_grayscale(img: np.ndarray, name: str = "image") -> None:
    """Raise ValueError if img is not a valid uint8 grayscale array."""
    if img.ndim != 2:
        raise ValueError(f"{name} must be 2D (grayscale), got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {img.dtype}")


def validate_mask(mask: np.ndarray, name: str = "mask") -> None:
    """Raise ValueError if mask is not a valid binary uint8 array."""
    validate_grayscale(mask, name)
    unique = np.unique(mask)
    if not set(unique).issubset({0, 255}):
        raise ValueError(f"{name} must contain only 0 and 255, got values: {unique}")


# ──────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────

def iter_image_paths(folder: str | Path) -> Iterator[Path]:
    """Yield all supported image paths from a folder (non-recursive).

    Args:
        folder: Directory to scan.

    Yields:
        Path objects for each supported image file found.

    Raises:
        NotADirectoryError: If folder does not exist or is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    for p in sorted(folder.iterdir()):
        if is_valid_image_path(p):
            yield p


def process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    strategy: str = "combined",
    enhance_kwargs: dict | None = None,
    segment_kwargs: dict | None = None,
) -> list[dict]:
    """Run the full enhance + segment pipeline on every image in a folder.

    Saves three files per image to output_folder:
      - <name>_original.png   (copy of original, for reference)
      - <name>_enhanced.png   (grayscale enhanced image)
      - <name>_mask.png       (binary segmentation mask)

    Args:
        input_folder:    Folder containing input images.
        output_folder:   Folder where outputs will be written (created if missing).
        strategy:        Segmentation strategy ("adaptive", "gradient", "combined").
        enhance_kwargs:  Optional kwargs forwarded to enhance().
        segment_kwargs:  Optional kwargs forwarded to segment().

    Returns:
        List of result dicts, one per image:
          {
            "name": str,
            "original_path": Path,
            "enhanced_path": Path,
            "mask_path": Path,
            "success": bool,
            "error": str | None,
          }
    """
    enhance_kwargs = enhance_kwargs or {}
    segment_kwargs = segment_kwargs or {}
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in iter_image_paths(input_folder):
        stem = img_path.stem
        result: dict = {
            "name": stem,
            "original_path": img_path,
            "enhanced_path": out / f"{stem}_enhanced.png",
            "mask_path": out / f"{stem}_mask.png",
            "success": False,
            "error": None,
        }
        try:
            original, enhanced = enhance(img_path, **enhance_kwargs)
            mask = segment(enhanced, strategy=strategy, **segment_kwargs)

            # Save original (for visualisation by Person 2)
            cv2.imwrite(str(out / f"{stem}_original.png"), original)
            save_enhanced(enhanced, result["enhanced_path"])
            save_mask(mask, result["mask_path"])

            result["success"] = True
        except Exception as exc:
            result["error"] = str(exc)

        results.append(result)

    return results


# ──────────────────────────────────────────────
# Debug visualisation
# ──────────────────────────────────────────────

def make_comparison_strip(
    original: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    labels: tuple[str, str, str] = ("Original", "Enhanced", "Segmentation Mask"),
) -> np.ndarray:
    """Build a horizontal strip showing original | enhanced | mask side by side.

    Useful for quick visual inspection during development.

    Args:
        original: BGR original image, (H, W, 3).
        enhanced: Grayscale enhanced image, (H, W).
        mask:     Binary mask, (H, W).
        labels:   Text labels drawn at the top of each panel.

    Returns:
        BGR strip image, shape (H, W*3, 3).
    """
    # Convert all panels to BGR for uniform concatenation
    panels = [
        original.copy(),
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
    ]

    # Resize panels to the same height (first image's height)
    h = panels[0].shape[0]
    resized = []
    for panel, label in zip(panels, labels):
        ph, pw = panel[:2] if panel.ndim == 2 else panel.shape[:2]
        scale = h / ph
        new_w = int(pw * scale)
        r = cv2.resize(panel, (new_w, h))

        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(r, label, (10, 30), font, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
        resized.append(r)

    return np.concatenate(resized, axis=1)


def save_comparison_strip(
    original: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a side-by-side comparison strip to disk.

    Args:
        original:    BGR original image.
        enhanced:    Grayscale enhanced image.
        mask:        Binary segmentation mask.
        output_path: Destination path for the strip image.
    """
    strip = make_comparison_strip(original, enhanced, mask)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), strip)
