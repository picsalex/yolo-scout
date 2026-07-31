"""Tests for src.visualization.iou module."""

from unittest.mock import Mock

import pytest

from yolo_scout.core.enums import DatasetTask
from yolo_scout.visualization.iou import (
    _compute_bbox_ious,
    _compute_polygon_ious,
    compute_iou_scores,
)


# Helper class to create dict-like objects that support both attribute and item access
class DictLikeObject(dict):
    """Dict-like object that supports both obj.attr and obj['attr'] syntax."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class TestComputeBboxIous:
    """Tests for _compute_bbox_ious function."""

    def test_compute_iou_for_overlapping_boxes(self):
        """Test IoU computation for overlapping bounding boxes."""
        # Create dict-like objects with overlapping bboxes
        obj1 = DictLikeObject()
        obj1.bounding_box = [0.2, 0.2, 0.4, 0.4]  # Box from (0.2,0.2) to (0.6,0.6)

        obj2 = DictLikeObject()
        obj2.bounding_box = [0.4, 0.4, 0.4, 0.4]  # Box from (0.4,0.4) to (0.8,0.8)

        objects = [obj1, obj2]

        _compute_bbox_ious(objects)

        # Both objects should have IoU > 0
        assert obj1["iou_score"] > 0
        assert obj2["iou_score"] > 0
        # IoU should be same for both
        assert obj1["iou_score"] == obj2["iou_score"]

    def test_compute_iou_for_non_overlapping_boxes(self):
        """Test IoU computation for non-overlapping bounding boxes."""
        obj1 = DictLikeObject()
        obj1.bounding_box = [0.1, 0.1, 0.2, 0.2]  # Box 1

        obj2 = DictLikeObject()
        obj2.bounding_box = [0.7, 0.7, 0.2, 0.2]  # Box 2 (far away)

        objects = [obj1, obj2]

        _compute_bbox_ious(objects)

        # Non-overlapping boxes should have IoU = 0
        assert obj1["iou_score"] == 0.0
        assert obj2["iou_score"] == 0.0

    def test_compute_iou_for_identical_boxes(self):
        """Test IoU computation for identical bounding boxes."""
        obj1 = DictLikeObject()
        obj1.bounding_box = [0.3, 0.3, 0.4, 0.4]

        obj2 = DictLikeObject()
        obj2.bounding_box = [0.3, 0.3, 0.4, 0.4]  # Same bbox

        objects = [obj1, obj2]

        _compute_bbox_ious(objects)

        # Identical boxes should have IoU = 1.0
        assert obj1["iou_score"] == pytest.approx(1.0, rel=1e-2)
        assert obj2["iou_score"] == pytest.approx(1.0, rel=1e-2)

    def test_compute_iou_single_box(self):
        """Test IoU computation for a single box (no overlaps)."""
        obj1 = DictLikeObject()
        obj1.bounding_box = [0.3, 0.3, 0.4, 0.4]

        objects = [obj1]

        _compute_bbox_ious(objects)

        # Single box should have IoU = 0 (no other boxes to overlap with)
        assert obj1["iou_score"] == 0.0

    def test_compute_iou_three_boxes(self):
        """Test IoU computation with three boxes."""
        obj1 = DictLikeObject()
        obj1.bounding_box = [0.2, 0.2, 0.3, 0.3]

        obj2 = DictLikeObject()
        obj2.bounding_box = [0.25, 0.25, 0.3, 0.3]  # Overlaps with obj1

        obj3 = DictLikeObject()
        obj3.bounding_box = [0.8, 0.8, 0.1, 0.1]  # Separate

        objects = [obj1, obj2, obj3]

        _compute_bbox_ious(objects)

        # obj1 and obj2 should have IoU > 0
        assert obj1["iou_score"] > 0
        assert obj2["iou_score"] > 0
        # obj3 should have IoU = 0 (no overlap)
        assert obj3["iou_score"] == 0.0

    def test_iou_score_is_rounded(self):
        """Test that IoU scores are rounded to 3 decimal places."""
        obj1 = DictLikeObject()
        obj1.bounding_box = [0.2, 0.2, 0.4, 0.4]

        obj2 = DictLikeObject()
        obj2.bounding_box = [0.3, 0.3, 0.4, 0.4]

        objects = [obj1, obj2]

        _compute_bbox_ious(objects)

        # Check that values are rounded
        iou_str = str(obj1["iou_score"])
        decimal_places = len(iou_str.split(".")[-1]) if "." in iou_str else 0
        assert decimal_places <= 3


class TestComputePolygonIous:
    """Tests for _compute_polygon_ious function."""

    def test_compute_iou_for_overlapping_polygons(self):
        """Test IoU computation for overlapping polygons."""
        obj1 = DictLikeObject()
        obj1.points = [[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]]

        obj2 = DictLikeObject()
        obj2.points = [[[0.4, 0.4], [0.8, 0.4], [0.8, 0.8], [0.4, 0.8]]]

        objects = [obj1, obj2]

        _compute_polygon_ious(objects)

        # Overlapping polygons should have IoU > 0
        assert obj1["iou_score"] > 0
        assert obj2["iou_score"] > 0

    def test_compute_iou_for_non_overlapping_polygons(self):
        """Test IoU computation for non-overlapping polygons."""
        obj1 = DictLikeObject()
        obj1.points = [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]]

        obj2 = DictLikeObject()
        obj2.points = [[[0.7, 0.7], [0.9, 0.7], [0.9, 0.9], [0.7, 0.9]]]

        objects = [obj1, obj2]

        _compute_polygon_ious(objects)

        # Non-overlapping polygons should have IoU = 0
        assert obj1["iou_score"] == 0.0
        assert obj2["iou_score"] == 0.0

    def test_compute_iou_with_invalid_polygon(self):
        """Test IoU computation with invalid polygon (< 3 points)."""
        obj1 = DictLikeObject()
        obj1.points = [[[0.2, 0.2], [0.6, 0.2]]]  # Only 2 points

        obj2 = DictLikeObject()
        obj2.points = [[[0.4, 0.4], [0.8, 0.4], [0.8, 0.8], [0.4, 0.8]]]

        objects = [obj1, obj2]

        # Should not raise error, just skip invalid polygon
        _compute_polygon_ious(objects)

        # Both should have IoU = 0 (invalid polygon skipped)
        assert obj1["iou_score"] == 0.0
        assert obj2["iou_score"] == 0.0


class TestComputeIouScores:
    """Tests for compute_iou_scores function."""

    def test_compute_iou_for_detection_task(self):
        """Test IoU computation for detection task."""
        # Create mock labels with detections
        labels = Mock()
        det1 = DictLikeObject()
        det1.bounding_box = [0.2, 0.2, 0.4, 0.4]
        det2 = DictLikeObject()
        det2.bounding_box = [0.4, 0.4, 0.4, 0.4]

        labels.get_field = Mock(return_value=[det1, det2])

        compute_iou_scores(labels, DatasetTask.DETECTION)

        # Should have computed IoU for both detections
        assert "iou_score" in det1
        assert "iou_score" in det2

    def test_compute_iou_for_segmentation_task(self):
        """Test IoU computation for segmentation task."""
        labels = Mock()
        poly1 = DictLikeObject()
        poly1.points = [[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]]
        poly2 = DictLikeObject()
        poly2.points = [[[0.7, 0.7], [0.9, 0.7], [0.9, 0.9], [0.7, 0.9]]]

        labels.get_field = Mock(return_value=[poly1, poly2])

        compute_iou_scores(labels, DatasetTask.SEGMENTATION)

        # Should have computed IoU for both polygons
        assert "iou_score" in poly1
        assert "iou_score" in poly2

    def test_compute_iou_for_obb_task(self):
        """Test IoU computation for OBB task."""
        labels = Mock()
        obb1 = DictLikeObject()
        obb1.points = [[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]]

        labels.get_field = Mock(return_value=[obb1])

        compute_iou_scores(labels, DatasetTask.OBB)

        # Should have computed IoU
        assert "iou_score" in obb1

    def test_compute_iou_with_none_labels(self):
        """Test that function handles None labels gracefully."""
        # Should not raise error
        compute_iou_scores(None, DatasetTask.DETECTION)

    def test_compute_iou_with_empty_objects(self):
        """Test IoU computation with empty objects list."""
        labels = Mock()
        labels.get_field = Mock(return_value=[])

        # Should not raise error
        compute_iou_scores(labels, DatasetTask.DETECTION)
