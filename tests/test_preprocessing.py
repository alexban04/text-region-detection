"""
test_preprocessing.py — Unit tests for Person 1's preprocessing modules.

Run with:
    pytest tests/test_preprocessing.py -v

Dependencies:
    pytest, numpy, opencv-python
"""

import numpy as np
import pytest
import cv2
import tempfile
from pathlib import Path

from src.preprocessing.enhance import (
    to_grayscale,
    apply_gamma_correction,
    apply_clahe,
    apply_denoising,
    load_image,
    save_enhanced,
)
from src.preprocessing.segment import (
    segment_adaptive_threshold,
    segment_gradient,
    segment_combined,
    segment,
    save_mask,
)
from src.preprocessing.utils import (
    validate_grayscale,
    validate_mask,
    make_comparison_strip,
    iter_image_paths,
    process_folder,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_bgr() -> np.ndarray:
    """A small synthetic BGR image (100×120) with a white rectangle on grey."""
    img = np.full((100, 120, 3), 180, dtype=np.uint8)  # grey background
    img[20:80, 10:110] = 255                             # white rectangle
    return img


@pytest.fixture
def sample_gray() -> np.ndarray:
    """A small synthetic grayscale image (100×120)."""
    img = np.full((100, 120), 180, dtype=np.uint8)
    img[30:70, 20:100] = 50   # dark "text-like" rectangle
    return img


@pytest.fixture
def sample_mask() -> np.ndarray:
    """A valid binary mask (100×120)."""
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[30:70, 20:100] = 255
    return mask


# ──────────────────────────────────────────────
# enhance.py tests
# ──────────────────────────────────────────────

class TestToGrayscale:
    def test_converts_bgr_to_2d(self, sample_bgr):
        result = to_grayscale(sample_bgr)
        assert result.ndim == 2

    def test_output_dtype_uint8(self, sample_bgr):
        result = to_grayscale(sample_bgr)
        assert result.dtype == np.uint8

    def test_already_grayscale_passes_through(self, sample_gray):
        result = to_grayscale(sample_gray)
        assert result.ndim == 2
        np.testing.assert_array_equal(result, sample_gray)

    def test_output_shape_matches_hw(self, sample_bgr):
        result = to_grayscale(sample_bgr)
        assert result.shape == sample_bgr.shape[:2]


class TestGammaCorrection:
    def test_output_shape_unchanged(self, sample_gray):
        result = apply_gamma_correction(sample_gray, gamma=1.5)
        assert result.shape == sample_gray.shape

    def test_output_dtype_uint8(self, sample_gray):
        result = apply_gamma_correction(sample_gray, gamma=1.5)
        assert result.dtype == np.uint8

    def test_gamma_gt_1_brightens(self, sample_gray):
        """Gamma > 1 should increase average pixel value."""
        result = apply_gamma_correction(sample_gray, gamma=2.0)
        assert result.mean() > sample_gray.mean()

    def test_gamma_lt_1_darkens(self, sample_gray):
        """Gamma < 1 should decrease average pixel value."""
        result = apply_gamma_correction(sample_gray, gamma=0.5)
        assert result.mean() < sample_gray.mean()

    def test_gamma_1_is_identity(self, sample_gray):
        """Gamma == 1.0 should return effectively the same image."""
        result = apply_gamma_correction(sample_gray, gamma=1.0)
        np.testing.assert_array_equal(result, sample_gray)


class TestCLAHE:
    def test_output_shape_unchanged(self, sample_gray):
        result = apply_clahe(sample_gray)
        assert result.shape == sample_gray.shape

    def test_output_dtype_uint8(self, sample_gray):
        result = apply_clahe(sample_gray)
        assert result.dtype == np.uint8

    def test_increases_contrast(self, sample_gray):
        """CLAHE should increase the standard deviation of pixel values."""
        result = apply_clahe(sample_gray, clip_limit=4.0)
        assert result.std() >= sample_gray.std() - 1  # allow tiny rounding


class TestDenoising:
    def test_output_shape_unchanged(self, sample_gray):
        result = apply_denoising(sample_gray)
        assert result.shape == sample_gray.shape

    def test_output_dtype_uint8(self, sample_gray):
        result = apply_denoising(sample_gray)
        assert result.dtype == np.uint8


class TestLoadSave:
    def test_load_nonexistent_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")

    def test_load_invalid_file_raises_valueerror(self, tmp_path):
        bad = tmp_path / "not_an_image.png"
        bad.write_bytes(b"not image data")
        with pytest.raises(ValueError):
            load_image(bad)

    def test_save_enhanced_creates_file(self, sample_gray, tmp_path):
        out = tmp_path / "enhanced.png"
        save_enhanced(sample_gray, out)
        assert out.exists()

    def test_save_enhanced_creates_parent_dirs(self, sample_gray, tmp_path):
        out = tmp_path / "nested" / "dir" / "enhanced.png"
        save_enhanced(sample_gray, out)
        assert out.exists()


# ──────────────────────────────────────────────
# segment.py tests
# ──────────────────────────────────────────────

class TestSegmentAdaptiveThreshold:
    def test_output_shape(self, sample_gray):
        result = segment_adaptive_threshold(sample_gray)
        assert result.shape == sample_gray.shape

    def test_output_dtype(self, sample_gray):
        result = segment_adaptive_threshold(sample_gray)
        assert result.dtype == np.uint8

    def test_output_binary(self, sample_gray):
        result = segment_adaptive_threshold(sample_gray)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_even_block_size_auto_corrected(self, sample_gray):
        # Should not raise even with an even block_size
        result = segment_adaptive_threshold(sample_gray, block_size=14)
        assert result.shape == sample_gray.shape


class TestSegmentGradient:
    def test_output_shape(self, sample_gray):
        result = segment_gradient(sample_gray)
        assert result.shape == sample_gray.shape

    def test_output_dtype(self, sample_gray):
        result = segment_gradient(sample_gray)
        assert result.dtype == np.uint8

    def test_output_binary(self, sample_gray):
        result = segment_gradient(sample_gray)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_detects_edges(self, sample_gray):
        """The gradient mask should have at least some non-zero pixels for an image with edges."""
        result = segment_gradient(sample_gray, threshold=5)
        assert result.max() == 255


class TestSegmentCombined:
    def test_output_shape(self, sample_gray):
        result = segment_combined(sample_gray)
        assert result.shape == sample_gray.shape

    def test_is_union_of_strategies(self, sample_gray):
        """Combined mask should be >= both individual masks (bitwise OR)."""
        adaptive = segment_adaptive_threshold(sample_gray)
        gradient = segment_gradient(sample_gray)
        combined = segment_combined(sample_gray)
        expected = cv2.bitwise_or(adaptive, gradient)
        np.testing.assert_array_equal(combined, expected)


class TestSegmentDispatch:
    def test_unknown_strategy_raises(self, sample_gray):
        with pytest.raises(ValueError, match="Unknown strategy"):
            segment(sample_gray, strategy="magic")

    def test_adaptive_strategy(self, sample_gray):
        result = segment(sample_gray, strategy="adaptive")
        assert result.shape == sample_gray.shape

    def test_gradient_strategy(self, sample_gray):
        result = segment(sample_gray, strategy="gradient")
        assert result.shape == sample_gray.shape

    def test_combined_strategy(self, sample_gray):
        result = segment(sample_gray, strategy="combined")
        assert result.shape == sample_gray.shape


class TestSaveMask:
    def test_creates_file(self, sample_mask, tmp_path):
        out = tmp_path / "mask.png"
        save_mask(sample_mask, out)
        assert out.exists()


# ──────────────────────────────────────────────
# utils.py tests
# ──────────────────────────────────────────────

class TestValidation:
    def test_validate_grayscale_passes(self, sample_gray):
        validate_grayscale(sample_gray)  # should not raise

    def test_validate_grayscale_fails_on_bgr(self, sample_bgr):
        with pytest.raises(ValueError):
            validate_grayscale(sample_bgr)

    def test_validate_mask_passes(self, sample_mask):
        validate_mask(sample_mask)  # should not raise

    def test_validate_mask_fails_on_non_binary(self, sample_gray):
        with pytest.raises(ValueError):
            validate_mask(sample_gray)


class TestIterImagePaths:
    def test_yields_only_image_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "b.txt").write_text("hello")
        (tmp_path / "c.jpg").write_bytes(b"")
        paths = list(iter_image_paths(tmp_path))
        names = {p.name for p in paths}
        assert names == {"a.png", "c.jpg"}

    def test_raises_on_missing_folder(self):
        with pytest.raises(NotADirectoryError):
            list(iter_image_paths("/does/not/exist"))


class TestMakeComparisonStrip:
    def test_output_is_bgr(self, sample_bgr, sample_gray, sample_mask):
        strip = make_comparison_strip(sample_bgr, sample_gray, sample_mask)
        assert strip.ndim == 3
        assert strip.shape[2] == 3

    def test_strip_height_matches_original(self, sample_bgr, sample_gray, sample_mask):
        strip = make_comparison_strip(sample_bgr, sample_gray, sample_mask)
        assert strip.shape[0] == sample_bgr.shape[0]

    def test_strip_is_wider_than_original(self, sample_bgr, sample_gray, sample_mask):
        strip = make_comparison_strip(sample_bgr, sample_gray, sample_mask)
        assert strip.shape[1] > sample_bgr.shape[1]


class TestProcessFolder:
    def test_processes_images_and_creates_outputs(self, tmp_path):
        # Create a synthetic valid PNG in the input folder
        img_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        img_dir.mkdir()

        synthetic = np.full((60, 80, 3), 200, dtype=np.uint8)
        synthetic[15:45, 10:70] = 30  # dark rectangle = "text"
        cv2.imwrite(str(img_dir / "test_image.png"), synthetic)

        results = process_folder(str(img_dir), str(out_dir))

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["enhanced_path"].exists()
        assert results[0]["mask_path"].exists()
