"""
test_detection.py — Unit tests for Person 3's modules.

Run with:
    pytest tests/test_detection.py -v
"""

import json
import numpy as np
import pytest
import cv2
from pathlib import Path

from src.detection.detect import (
    _box_area, _box_aspect, _iou,
    filter_boxes, find_text_boxes,
)
from src.detection.decide import (
    classify_region, make_decision,
    save_decision, load_decision,
)
from main import run_pipeline


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def mask_two_blobs():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[20:50,  20:200] = 255
    mask[80:110, 30:250] = 255
    return mask


@pytest.fixture
def mask_noisy():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[20:50,  20:200] = 255
    mask[80:110, 30:250] = 255
    for r, c in [(5, 5), (190, 290), (100, 5)]:
        mask[r, c] = 255
    return mask


@pytest.fixture
def empty_mask():
    return np.zeros((200, 300), dtype=np.uint8)


@pytest.fixture
def sample_bgr():
    return np.full((200, 300, 3), 200, dtype=np.uint8)


@pytest.fixture
def two_boxes():
    return [(20, 20, 180, 30), (30, 80, 220, 30)]


@pytest.fixture
def decision_two(two_boxes, sample_bgr):
    return make_decision(two_boxes, sample_bgr.shape)


# ──────────────────────────────────────────────
# detect.py helpers
# ──────────────────────────────────────────────

class TestBoxArea:
    def test_basic(self):
        assert _box_area((0, 0, 10, 20)) == 200

    def test_zero(self):
        assert _box_area((0, 0, 0, 10)) == 0


class TestBoxAspect:
    def test_wide(self):
        assert _box_aspect((0, 0, 100, 20)) == pytest.approx(5.0)

    def test_zero_height(self):
        assert _box_aspect((0, 0, 10, 0)) == float("inf")


class TestIou:
    def test_identical(self):
        b = (0, 0, 50, 50)
        assert _iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)

    def test_partial(self):
        iou = _iou((0, 0, 10, 10), (5, 0, 10, 10))
        assert 0.0 < iou < 1.0

    def test_symmetric(self):
        a, b = (0, 0, 20, 30), (10, 10, 20, 30)
        assert _iou(a, b) == pytest.approx(_iou(b, a))

    def test_zero_area(self):
        assert _iou((0, 0, 0, 0), (0, 0, 0, 0)) == pytest.approx(0.0)


# ──────────────────────────────────────────────
# filter_boxes
# ──────────────────────────────────────────────

class TestFilterBoxes:
    def test_removes_small(self):
        result = filter_boxes([(0, 0, 5, 5), (0, 0, 100, 30)], min_area=100)
        assert (0, 0, 100, 30) in result
        assert (0, 0, 5, 5) not in result

    def test_removes_over_max_area(self):
        result = filter_boxes([(0, 0, 200, 200), (0, 0, 50, 30)], max_area=5000)
        assert (0, 0, 50, 30) in result
        assert (0, 0, 200, 200) not in result

    def test_removes_thin(self):
        result = filter_boxes(
            [(0, 0, 200, 2), (0, 0, 200, 20)],
            min_area=10, max_aspect_ratio=20.0
        )
        assert (0, 0, 200, 20) in result
        assert (0, 0, 200, 2) not in result

    def test_nms_collapses_duplicates(self):
        a = (10, 10, 100, 30)
        b = (12, 10, 100, 30)
        result = filter_boxes([a, b], min_area=10, nms_iou_threshold=0.5)
        assert len(result) == 1

    def test_nms_keeps_non_overlapping(self):
        a = (10, 10, 100, 30)
        b = (10, 100, 100, 30)
        result = filter_boxes([a, b], min_area=10, nms_iou_threshold=0.5)
        assert len(result) == 2

    def test_empty_input(self):
        assert filter_boxes([]) == []

    def test_reading_order(self):
        boxes = [(50, 80, 100, 20), (10, 10, 100, 20), (60, 10, 100, 20)]
        result = filter_boxes(boxes, min_area=10)
        ys = [b[1] for b in result]
        assert ys == sorted(ys)

    def test_min_width(self):
        result = filter_boxes([(0, 0, 3, 30), (0, 0, 50, 30)], min_area=10, min_width=10)
        assert (0, 0, 50, 30) in result
        assert (0, 0, 3, 30) not in result

    def test_min_height(self):
        result = filter_boxes([(0, 0, 50, 2), (0, 0, 50, 20)], min_area=10, min_height=5)
        assert (0, 0, 50, 20) in result
        assert (0, 0, 50, 2) not in result


# ──────────────────────────────────────────────
# find_text_boxes
# ──────────────────────────────────────────────

class TestFindTextBoxes:
    def test_finds_two_blobs(self, mask_two_blobs):
        boxes = find_text_boxes(mask_two_blobs, min_area=100)
        assert len(boxes) == 2

    def test_empty_mask(self, empty_mask):
        assert find_text_boxes(empty_mask) == []

    def test_output_tuples(self, mask_two_blobs):
        for b in find_text_boxes(mask_two_blobs, min_area=100):
            assert len(b) == 4
            assert all(isinstance(v, int) for v in b)

    def test_boxes_in_bounds(self, mask_two_blobs):
        H, W = mask_two_blobs.shape
        for (x, y, w, h) in find_text_boxes(mask_two_blobs, min_area=100):
            assert 0 <= x and 0 <= y
            assert x + w <= W and y + h <= H

    def test_filters_noise(self, mask_noisy):
        for (_, _, w, h) in find_text_boxes(mask_noisy, min_area=100):
            assert w * h >= 100

    def test_reading_order(self, mask_two_blobs):
        boxes = find_text_boxes(mask_two_blobs, min_area=100)
        assert boxes[0][1] <= boxes[1][1]


# ──────────────────────────────────────────────
# classify_region
# ──────────────────────────────────────────────

class TestClassifyRegion:
    def test_word(self):
        assert classify_region((0, 0, 80, 30)) == "word"

    def test_line(self):
        assert classify_region((0, 0, 300, 30)) == "line"

    def test_block(self):
        assert classify_region((0, 0, 200, 150)) == "block"

    def test_zero_height(self):
        label = classify_region((0, 0, 100, 0))
        assert label in ("word", "line", "block")


# ──────────────────────────────────────────────
# make_decision
# ──────────────────────────────────────────────

class TestMakeDecision:
    def test_text_found_true(self, two_boxes, sample_bgr):
        assert make_decision(two_boxes, sample_bgr.shape)["text_found"] is True

    def test_text_found_false_empty(self, sample_bgr):
        assert make_decision([], sample_bgr.shape)["text_found"] is False

    def test_region_count(self, two_boxes, sample_bgr):
        assert make_decision(two_boxes, sample_bgr.shape)["region_count"] == 2

    def test_zero_count(self, sample_bgr):
        assert make_decision([], sample_bgr.shape)["region_count"] == 0

    def test_image_area(self, two_boxes, sample_bgr):
        H, W = sample_bgr.shape[:2]
        assert make_decision(two_boxes, sample_bgr.shape)["image_area"] == H * W

    def test_total_area(self, two_boxes, sample_bgr):
        expected = sum(w * h for (_, _, w, h) in two_boxes)
        assert make_decision(two_boxes, sample_bgr.shape)["total_area"] == expected

    def test_coverage_range(self, two_boxes, sample_bgr):
        d = make_decision(two_boxes, sample_bgr.shape)
        assert 0.0 <= d["coverage"] <= 1.0

    def test_verdict_has_count(self, two_boxes, sample_bgr):
        assert "2" in make_decision(two_boxes, sample_bgr.shape)["verdict"]

    def test_no_text_verdict(self, sample_bgr):
        assert "NO TEXT" in make_decision([], sample_bgr.shape)["verdict"]

    def test_region_schema(self, two_boxes, sample_bgr):
        for r in make_decision(two_boxes, sample_bgr.shape)["regions"]:
            for key in ("x", "y", "w", "h", "area", "label"):
                assert key in r

    def test_coverage_threshold_blocks_tiny(self):
        d = make_decision([(0, 0, 5, 5)], (10000, 10000), coverage_threshold=0.01)
        assert d["text_found"] is False

    def test_json_serialisable(self, two_boxes, sample_bgr):
        json.dumps(make_decision(two_boxes, sample_bgr.shape))


# ──────────────────────────────────────────────
# save / load decision
# ──────────────────────────────────────────────

class TestSaveLoad:
    def test_round_trip(self, decision_two, tmp_path):
        path = tmp_path / "d.json"
        save_decision(decision_two, path)
        loaded = load_decision(path)
        assert loaded["text_found"] == decision_two["text_found"]
        assert loaded["verdict"] == decision_two["verdict"]

    def test_creates_parents(self, decision_two, tmp_path):
        path = tmp_path / "a" / "b" / "d.json"
        save_decision(decision_two, path)
        assert path.exists()

    def test_valid_json(self, decision_two, tmp_path):
        path = tmp_path / "d.json"
        save_decision(decision_two, path)
        with open(path) as f:
            data = json.load(f)
        assert "verdict" in data


# ──────────────────────────────────────────────
# Integration — run_pipeline
# ──────────────────────────────────────────────

class TestRunPipeline:
    @pytest.fixture
    def synthetic_image(self, tmp_path):
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        img[40:70,  30:270] = 20
        img[100:130, 50:250] = 20
        path = tmp_path / "test_img.png"
        cv2.imwrite(str(path), img)
        return path

    def test_all_output_files_exist(self, synthetic_image, tmp_path):
        result = run_pipeline(synthetic_image, tmp_path / "out", verbose=False)
        for key, p in result["paths"].items():
            assert Path(p).exists(), f"Missing: {key}"

    def test_decision_keys(self, synthetic_image, tmp_path):
        d = run_pipeline(synthetic_image, tmp_path / "out", verbose=False)["decision"]
        for key in ("text_found", "region_count", "verdict"):
            assert key in d

    def test_text_found_is_bool(self, synthetic_image, tmp_path):
        d = run_pipeline(synthetic_image, tmp_path / "out", verbose=False)["decision"]
        assert isinstance(d["text_found"], bool)

    def test_detects_text_in_synthetic(self, synthetic_image, tmp_path):
        d = run_pipeline(
            synthetic_image, tmp_path / "out", min_area=50, verbose=False
        )["decision"]
        assert d["text_found"] is True

    def test_stage_grid_readable(self, synthetic_image, tmp_path):
        result = run_pipeline(synthetic_image, tmp_path / "out", verbose=False)
        grid = cv2.imread(str(result["paths"]["stage_grid"]))
        assert grid is not None and grid.ndim == 3
