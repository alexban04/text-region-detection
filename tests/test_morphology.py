"""
test_morphology.py — Unit tests for Person 2's modules.

Run with:
    pytest tests/test_morphology.py -v

Dependencies:
    pytest, numpy, opencv-python
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from src.morphology.clean import (
    erode,
    dilate,
    opening,
    closing,
    filter_components,
    clean,
    save_cleaned_mask,
)
from src.visualization.plot_stages import (
    draw_bounding_boxes,
    build_stage_grid,
    build_contact_sheet,
    save_stage_grid,
    save_contact_sheet,
    _to_bgr,
    _add_label,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def clean_mask() -> np.ndarray:
    """A binary mask with one clear text-like blob (100×150)."""
    mask = np.zeros((100, 150), dtype=np.uint8)
    mask[20:60, 30:120] = 255   # a single wide rectangle — text line
    return mask


@pytest.fixture
def noisy_mask() -> np.ndarray:
    """A binary mask with a text blob + scattered noise pixels."""
    mask = np.zeros((100, 150), dtype=np.uint8)
    mask[20:60, 30:120] = 255   # real text region
    # Add isolated noise specks
    for (r, c) in [(5, 5), (5, 140), (90, 5), (90, 140), (50, 145)]:
        mask[r, c] = 255
    return mask


@pytest.fixture
def sample_bgr() -> np.ndarray:
    """A small synthetic BGR image (100×150)."""
    img = np.full((100, 150, 3), 200, dtype=np.uint8)
    return img


@pytest.fixture
def sample_gray() -> np.ndarray:
    """A small synthetic grayscale image (100×150)."""
    return np.full((100, 150), 180, dtype=np.uint8)


@pytest.fixture
def sample_boxes() -> list[tuple[int, int, int, int]]:
    """Two realistic bounding boxes."""
    return [(30, 20, 90, 40), (30, 65, 60, 25)]


@pytest.fixture
def dummy_decision() -> dict:
    return {"text_found": True, "region_count": 2}


# ──────────────────────────────────────────────
# clean.py — individual operations
# ──────────────────────────────────────────────

class TestErode:
    def test_output_shape(self, clean_mask):
        assert erode(clean_mask).shape == clean_mask.shape

    def test_output_dtype(self, clean_mask):
        assert erode(clean_mask).dtype == np.uint8

    def test_output_binary(self, clean_mask):
        result = erode(clean_mask)
        assert set(np.unique(result)).issubset({0, 255})

    def test_shrinks_white_region(self, clean_mask):
        result = erode(clean_mask, ksize=5, iterations=2)
        assert result.sum() <= clean_mask.sum()


class TestDilate:
    def test_output_shape(self, clean_mask):
        assert dilate(clean_mask).shape == clean_mask.shape

    def test_output_dtype(self, clean_mask):
        assert dilate(clean_mask).dtype == np.uint8

    def test_output_binary(self, clean_mask):
        result = dilate(clean_mask)
        assert set(np.unique(result)).issubset({0, 255})

    def test_expands_white_region(self, clean_mask):
        result = dilate(clean_mask, ksize=5, iterations=2)
        assert result.sum() >= clean_mask.sum()


class TestOpening:
    def test_output_shape(self, noisy_mask):
        assert opening(noisy_mask).shape == noisy_mask.shape

    def test_output_dtype(self, noisy_mask):
        assert opening(noisy_mask).dtype == np.uint8

    def test_output_binary(self, noisy_mask):
        result = opening(noisy_mask)
        assert set(np.unique(result)).issubset({0, 255})

    def test_removes_isolated_pixels(self, noisy_mask):
        """Opening with a 5×5 kernel should kill single pixels."""
        result = opening(noisy_mask, ksize=5)
        # Check that the corner noise pixels are gone
        assert result[5, 5] == 0
        assert result[5, 140] == 0

    def test_preserves_large_region(self, noisy_mask):
        """The large text rectangle should survive opening."""
        result = opening(noisy_mask, ksize=3)
        # Centre of the text rectangle should still be white
        assert result[40, 75] == 255


class TestClosing:
    def test_output_shape(self, clean_mask):
        assert closing(clean_mask).shape == clean_mask.shape

    def test_output_dtype(self, clean_mask):
        assert closing(clean_mask).dtype == np.uint8

    def test_output_binary(self, clean_mask):
        result = closing(clean_mask)
        assert set(np.unique(result)).issubset({0, 255})

    def test_fills_small_gap(self):
        """Closing should fill a horizontal gap between two blobs."""
        mask = np.zeros((50, 100), dtype=np.uint8)
        mask[20:30, 10:40] = 255   # left word
        mask[20:30, 45:80] = 255   # right word  (5-pixel gap)
        result = closing(mask, kw=10, kh=3)
        # The gap should now be filled
        assert result[25, 42] == 255


class TestFilterComponents:
    def test_output_shape(self, noisy_mask):
        assert filter_components(noisy_mask).shape == noisy_mask.shape

    def test_output_dtype(self, noisy_mask):
        assert filter_components(noisy_mask).dtype == np.uint8

    def test_output_binary(self, noisy_mask):
        result = filter_components(noisy_mask)
        assert set(np.unique(result)).issubset({0, 255})

    def test_removes_small_components(self, noisy_mask):
        """Single-pixel noise should be removed with min_area=10."""
        result = filter_components(noisy_mask, min_area=10)
        assert result[5, 5] == 0

    def test_keeps_large_component(self, noisy_mask):
        """The large text rectangle must survive component filtering."""
        result = filter_components(noisy_mask, min_area=10)
        assert result[40, 75] == 255

    def test_max_area_removes_large_blobs(self):
        """max_area should discard components bigger than the limit."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 255   # big blob (area ≈ 6400)
        result = filter_components(mask, max_area=100)
        assert result.max() == 0

    def test_aspect_ratio_filter(self):
        """A very wide thin line should be removed by max_aspect_ratio."""
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[50:52, 10:190] = 255   # 2px tall × 180px wide → ratio = 90
        result = filter_components(mask, min_area=10, max_aspect_ratio=5.0)
        assert result.max() == 0


# ──────────────────────────────────────────────
# clean.py — full pipeline
# ──────────────────────────────────────────────

class TestClean:
    def test_output_shape(self, noisy_mask):
        result = clean(noisy_mask)
        assert result.shape == noisy_mask.shape

    def test_output_dtype(self, noisy_mask):
        result = clean(noisy_mask)
        assert result.dtype == np.uint8

    def test_output_binary(self, noisy_mask):
        result = clean(noisy_mask)
        assert set(np.unique(result)).issubset({0, 255})

    def test_removes_isolated_noise_pixels(self, noisy_mask):
        """Corner noise pixels must be gone after cleaning (even if closing grows the main blob)."""
        result = clean(noisy_mask, min_area=20)
        # The isolated corner pixels should have been wiped by opening + component filter
        assert result[5, 5] == 0
        assert result[5, 140] == 0
        assert result[90, 5] == 0

    def test_preserves_text_region(self, noisy_mask):
        """The centre of the text blob should survive a clean pass."""
        result = clean(noisy_mask)
        assert result[40, 75] == 255

    def test_all_zeros_mask(self):
        """A fully-black mask should pass through unchanged."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        result = clean(mask)
        assert result.max() == 0

    def test_save_cleaned_mask(self, clean_mask, tmp_path):
        out = tmp_path / "cleaned.png"
        save_cleaned_mask(clean_mask, out)
        assert out.exists()


# ──────────────────────────────────────────────
# plot_stages.py — helpers
# ──────────────────────────────────────────────

class TestToBgr:
    def test_grayscale_becomes_3channel(self, sample_gray):
        result = _to_bgr(sample_gray)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_bgr_passes_through(self, sample_bgr):
        result = _to_bgr(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_does_not_modify_original(self, sample_gray):
        original = sample_gray.copy()
        _to_bgr(sample_gray)
        np.testing.assert_array_equal(sample_gray, original)


class TestAddLabel:
    def test_output_shape_unchanged(self, sample_bgr):
        result = _add_label(sample_bgr, "Test Label")
        assert result.shape == sample_bgr.shape

    def test_does_not_modify_original(self, sample_bgr):
        original = sample_bgr.copy()
        _add_label(sample_bgr, "Test")
        np.testing.assert_array_equal(sample_bgr, original)


# ──────────────────────────────────────────────
# plot_stages.py — draw_bounding_boxes
# ──────────────────────────────────────────────

class TestDrawBoundingBoxes:
    def test_output_shape(self, sample_bgr, sample_boxes):
        result = draw_bounding_boxes(sample_bgr, sample_boxes)
        assert result.shape == sample_bgr.shape

    def test_output_dtype(self, sample_bgr, sample_boxes):
        result = draw_bounding_boxes(sample_bgr, sample_boxes)
        assert result.dtype == np.uint8

    def test_does_not_modify_original(self, sample_bgr, sample_boxes):
        original = sample_bgr.copy()
        draw_bounding_boxes(sample_bgr, sample_boxes)
        np.testing.assert_array_equal(sample_bgr, original)

    def test_accepts_grayscale_input(self, sample_gray, sample_boxes):
        result = draw_bounding_boxes(sample_gray, sample_boxes)
        assert result.ndim == 3

    def test_empty_boxes_returns_copy(self, sample_bgr):
        result = draw_bounding_boxes(sample_bgr, [])
        np.testing.assert_array_equal(result, sample_bgr)

    def test_boxes_change_pixels(self, sample_bgr, sample_boxes):
        """At least some pixels must differ after drawing green boxes."""
        result = draw_bounding_boxes(sample_bgr, sample_boxes)
        assert not np.array_equal(result, sample_bgr)


# ──────────────────────────────────────────────
# plot_stages.py — build_stage_grid
# ──────────────────────────────────────────────

class TestBuildStageGrid:
    def _make_grid(self, sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision):
        detection = draw_bounding_boxes(sample_bgr, sample_boxes)
        return build_stage_grid(
            original        = sample_bgr,
            enhanced        = sample_gray,
            seg_mask        = clean_mask,
            cleaned_mask    = clean_mask,
            detection_result= detection,
            decision        = dummy_decision,
            panel_w         = 100,
            panel_h         = 80,
            border          = 2,
        )

    def test_output_is_bgr(self, sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision):
        grid = self._make_grid(sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision)
        assert grid.ndim == 3
        assert grid.shape[2] == 3

    def test_output_dtype(self, sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision):
        grid = self._make_grid(sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision)
        assert grid.dtype == np.uint8

    def test_grid_taller_than_one_panel(self, sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision):
        grid = self._make_grid(sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision)
        # 2 rows → grid height > single panel height (80)
        assert grid.shape[0] > 80

    def test_grid_wider_than_one_panel(self, sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision):
        grid = self._make_grid(sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision)
        # 3 cols → grid width > single panel width (100)
        assert grid.shape[1] > 100

    def test_save_stage_grid(self, sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision, tmp_path):
        grid = self._make_grid(sample_bgr, sample_gray, clean_mask, sample_boxes, dummy_decision)
        out = tmp_path / "grid.png"
        save_stage_grid(grid, out)
        assert out.exists()

    def test_no_text_decision(self, sample_bgr, sample_gray, clean_mask, sample_boxes):
        """Grid should build without errors when text_found is False."""
        decision = {"text_found": False, "region_count": 0}
        detection = draw_bounding_boxes(sample_bgr, [])
        grid = build_stage_grid(
            original=sample_bgr, enhanced=sample_gray,
            seg_mask=clean_mask, cleaned_mask=clean_mask,
            detection_result=detection, decision=decision,
            panel_w=100, panel_h=80,
        )
        assert grid.shape[2] == 3


# ──────────────────────────────────────────────
# plot_stages.py — build_contact_sheet
# ──────────────────────────────────────────────

class TestBuildContactSheet:
    def _dummy_images(self, n=4):
        return [np.full((60, 80, 3), i * 30, dtype=np.uint8) for i in range(n)]

    def _dummy_labels(self, n=4):
        return [f"image_{i}" for i in range(n)]

    def test_output_is_bgr(self):
        sheet = build_contact_sheet(self._dummy_images(), self._dummy_labels(), cols=2)
        assert sheet.ndim == 3 and sheet.shape[2] == 3

    def test_output_dtype(self):
        sheet = build_contact_sheet(self._dummy_images(), self._dummy_labels(), cols=2)
        assert sheet.dtype == np.uint8

    def test_correct_number_of_rows(self):
        """4 images in 3 cols → 2 rows."""
        sheet = build_contact_sheet(
            self._dummy_images(4), self._dummy_labels(4),
            thumb_w=80, thumb_h=60, cols=3, border=2
        )
        expected_h = 2 * (60 + 2) + 2
        assert sheet.shape[0] == expected_h

    def test_single_image(self):
        sheet = build_contact_sheet(self._dummy_images(1), self._dummy_labels(1))
        assert sheet.ndim == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_contact_sheet([], [])

    def test_save_contact_sheet(self, tmp_path):
        sheet = build_contact_sheet(self._dummy_images(2), self._dummy_labels(2))
        out = tmp_path / "sheet.png"
        save_contact_sheet(sheet, out)
        assert out.exists()
