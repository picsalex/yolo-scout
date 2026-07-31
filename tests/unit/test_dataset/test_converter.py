"""Tests for src.dataset.converter module."""

from unittest.mock import Mock

import pytest

from yolo_scout.dataset.converter import (
    _create_detection,
    _create_keypoint,
    _create_obb,
    _create_polygon,
    create_detection_from_keypoint,
)


class TestCreateDetection:
    """Tests for _create_detection function."""

    def test_create_detection_with_valid_data(self):
        """Test creating detection with valid annotation data."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.2,
        }
        class_names = ["cat", "dog"]

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        assert detection is not None
        assert detection.label == "cat"
        assert len(detection.bounding_box) == 4
        assert "train" in detection.tags

        # Check bbox conversion (x_center - w/2, y_center - h/2, w, h)
        x, y, w, h = detection.bounding_box
        assert x == pytest.approx(0.35)  # 0.5 - 0.3/2
        assert y == pytest.approx(0.4)  # 0.5 - 0.2/2
        assert w == pytest.approx(0.3)
        assert h == pytest.approx(0.2)

    def test_create_detection_with_zero_width(self):
        """Test creating detection with zero width (edge case)."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.0,
            "height": 0.2,
        }
        class_names = ["cat"]

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        assert detection is not None
        assert detection.bounding_box[2] == pytest.approx(1.0 / 1000)

    def test_create_detection_with_zero_height(self):
        """Test creating detection with zero height (edge case)."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.0,
        }
        class_names = ["cat"]

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        assert detection is not None
        assert detection.bounding_box[3] == pytest.approx(1.0 / 800)

    def test_create_detection_clamping_bounds(self):
        """Test that bounding box is clamped to [0, 1] range."""
        anno = {
            "class_id": 0,
            "x_center": 0.05,
            "y_center": 0.05,
            "width": 0.3,  # Would extend beyond 0
            "height": 0.3,
        }
        class_names = ["cat"]

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        x, y, w, h = detection.bounding_box
        assert x >= 0.0
        assert y >= 0.0
        assert x + w <= 1.0
        assert y + h <= 1.0

    def test_create_detection_metadata_fields(self):
        """Test that detection has correct metadata fields."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.2,
        }
        class_names = ["cat"]

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        # Check metadata fields
        assert "area" in detection
        assert "aspect_ratio" in detection
        assert "width" in detection
        assert "height" in detection

        # Check values
        assert detection["area"] == int(0.3 * 1000 * 0.2 * 800)  # 48000
        assert detection["width"] == int(0.3 * 1000)  # 300
        assert detection["height"] == int(0.2 * 800)  # 160

    def test_create_detection_aspect_ratio(self):
        """Test aspect ratio calculation."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.4,
            "height": 0.2,
        }
        class_names = ["cat"]

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        # aspect_ratio = width / height = 0.4 / 0.2 = 2.0
        assert detection["aspect_ratio"] == pytest.approx(2.0)

    def test_create_detection_with_missing_class_name(self):
        """Test creating detection when class_id exceeds class_names length."""
        anno = {
            "class_id": 5,  # Out of range
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.2,
        }
        class_names = ["cat", "dog"]  # Only 2 classes

        detection = _create_detection(anno, class_names, 1000, 800, "train")

        # Should fallback to "class_5"
        assert detection.label == "class_5"


class TestCreateKeypoint:
    """Tests for _create_keypoint function."""

    def test_create_keypoint_with_visible_keypoints(self):
        """Test creating keypoint with all visible keypoints."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.4,
            "keypoints": [
                {"x": 0.5, "y": 0.4, "visibility": 2},
                {"x": 0.52, "y": 0.42, "visibility": 2},
                {"x": 0.48, "y": 0.42, "visibility": 2},
            ],
        }
        class_names = ["person"]

        keypoint = _create_keypoint(anno, class_names, 1000, 800, "train")

        assert keypoint is not None
        assert keypoint.label == "person"
        assert len(keypoint.points) == 3
        assert keypoint["num_keypoints"] == 3

    def test_create_keypoint_with_hidden_keypoints(self):
        """Test creating keypoint with some hidden keypoints (0, 0)."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.4,
            "keypoints": [
                {"x": 0.5, "y": 0.4, "visibility": 2},
                {"x": 0.0, "y": 0.0, "visibility": 0},  # Hidden
                {"x": 0.48, "y": 0.42, "visibility": 2},
            ],
        }
        class_names = ["person"]

        keypoint = _create_keypoint(anno, class_names, 1000, 800, "train")

        assert keypoint is not None
        assert len(keypoint.points) == 3
        # Only 2 visible keypoints
        assert keypoint["num_keypoints"] == 2
        # Hidden keypoint should be [-1, -1]
        assert keypoint.points[1] == [-1, -1]

    def test_create_keypoint_with_no_keypoints(self):
        """Test creating keypoint with no keypoint data."""
        anno = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.3,
            "height": 0.4,
            "keypoints": [],
        }
        class_names = ["person"]

        keypoint = _create_keypoint(anno, class_names, 1000, 800, "train")

        assert keypoint is None

    def test_create_keypoint_bbox_clamping(self):
        """Test that keypoint bbox is clamped to [0, 1]."""
        anno = {
            "class_id": 0,
            "x_center": 0.05,
            "y_center": 0.05,
            "width": 0.3,
            "height": 0.3,
            "keypoints": [{"x": 0.1, "y": 0.1, "visibility": 2}],
        }
        class_names = ["person"]

        keypoint = _create_keypoint(anno, class_names, 1000, 800, "train")

        bbox = keypoint.bounding_box
        assert bbox[0] >= 0.0  # x
        assert bbox[1] >= 0.0  # y
        assert bbox[0] + bbox[2] <= 1.0  # x + width
        assert bbox[1] + bbox[3] <= 1.0  # y + height


class TestCreatePolygon:
    """Tests for _create_polygon function."""

    def test_create_polygon_with_valid_points(self):
        """Test creating polygon with valid points."""
        anno = {
            "class_id": 0,
            "points": [
                {"x": 0.3, "y": 0.3},
                {"x": 0.7, "y": 0.3},
                {"x": 0.7, "y": 0.7},
                {"x": 0.3, "y": 0.7},
            ],
        }
        class_names = ["car"]

        polygon = _create_polygon(anno, class_names, 1000, 800, "train")

        assert polygon is not None
        assert polygon.label == "car"
        assert polygon.closed is True
        assert polygon.filled is True
        # Should have 5 points (4 + closing point)
        assert len(polygon.points[0]) == 5
        # First and last points should be the same (closed polygon)
        assert polygon.points[0][0] == polygon.points[0][-1]

    def test_create_polygon_with_few_points(self):
        """Test creating polygon with too few points."""
        anno = {
            "class_id": 0,
            "points": [
                {"x": 0.3, "y": 0.3},
                {"x": 0.7, "y": 0.3},
            ],
        }
        class_names = ["car"]

        polygon = _create_polygon(anno, class_names, 1000, 800, "train")

        assert polygon is None

    def test_create_polygon_metadata_fields(self):
        """Test that polygon has correct metadata fields."""
        anno = {
            "class_id": 0,
            "points": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 0.5},
                {"x": 0.0, "y": 0.5},
            ],
        }
        class_names = ["car"]

        polygon = _create_polygon(anno, class_names, 1000, 800, "train")

        assert "area" in polygon
        assert "num_keypoints" in polygon
        assert "width" in polygon
        assert "height" in polygon

        # Check approximate area (0.5 * 1000 * 0.5 * 800 = 200000)
        assert polygon["area"] > 0


class TestCreateOBB:
    """Tests for _create_obb function."""

    def test_create_obb_with_four_points(self):
        """Test creating OBB with exactly 4 points."""
        anno = {
            "class_id": 0,
            "points": [
                {"x": 0.3, "y": 0.3},
                {"x": 0.7, "y": 0.3},
                {"x": 0.7, "y": 0.7},
                {"x": 0.3, "y": 0.7},
            ],
        }
        class_names = ["ship"]

        obb = _create_obb(anno, class_names, 1000, 800, "train")

        assert obb is not None
        assert obb.label == "ship"
        assert obb.closed is True
        assert obb.filled is False
        # Should have 5 points (4 + closing point)
        assert len(obb.points[0]) == 5

    def test_create_obb_with_wrong_point_count(self):
        """Test creating OBB with wrong number of points."""
        anno = {
            "class_id": 0,
            "points": [
                {"x": 0.3, "y": 0.3},
                {"x": 0.7, "y": 0.3},
                {"x": 0.7, "y": 0.7},
            ],
        }
        class_names = ["ship"]

        obb = _create_obb(anno, class_names, 1000, 800, "train")

        assert obb is None


class TestCreateDetectionFromKeypoint:
    """Tests for create_detection_from_keypoint function."""

    def test_create_detection_from_keypoint(self):
        """Test converting a keypoint to a detection."""
        # Create a mock keypoint
        keypoint = Mock()
        keypoint.label = "person"
        keypoint.tags = ["train"]
        keypoint.get_field = Mock(
            side_effect=lambda x: {
                "bounding_box": [0.35, 0.45, 0.3, 0.2],
                "area": 48000,
                "num_keypoints": 17,
            }[x]
        )

        detection = create_detection_from_keypoint(
            keypoint=keypoint,
            image_width=1000,
            image_height=800,
            image_path="/path/to/image.jpg",
            label_path="/path/to/label.txt",
        )

        assert detection.label == "person"
        assert detection.tags == ["train"]
        assert detection["area"] == 48000
        assert detection["num_keypoints"] == 17
        assert detection["width"] == int(0.3 * 1000)
        assert detection["height"] == int(0.2 * 800)
        assert detection["image_path"] == "/path/to/image.jpg"
        assert detection["label_path"] == "/path/to/label.txt"
