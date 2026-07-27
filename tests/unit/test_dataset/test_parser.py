"""Tests for src.dataset.parser module."""

from yolo_scout.core.enums import DatasetTask
from yolo_scout.dataset.parser import (
    _parse_detection_line,
    _parse_obb_line,
    _parse_pose_line,
    _parse_segmentation_line,
    parse_yolo_annotation,
)


class TestParseDetectionLine:
    """Tests for _parse_detection_line function."""

    def test_parse_valid_detection_line(self):
        """Test parsing a valid detection annotation line."""
        line = "0 0.5 0.5 0.3 0.2"
        result = _parse_detection_line(line)

        assert result is not None
        assert result["class_id"] == 0
        assert result["x_center"] == 0.5
        assert result["y_center"] == 0.5
        assert result["width"] == 0.3
        assert result["height"] == 0.2

    def test_parse_detection_with_float_class_id(self):
        """Test parsing detection with various numeric formats."""
        line = "2 0.123 0.456 0.789 0.012"
        result = _parse_detection_line(line)

        assert result is not None
        assert result["class_id"] == 2
        assert result["x_center"] == 0.123
        assert result["y_center"] == 0.456

    def test_parse_detection_with_insufficient_values(self):
        """Test parsing detection line with too few values."""
        line = "0 0.5 0.5"  # Missing width and height
        result = _parse_detection_line(line)
        assert result is None

    def test_parse_detection_empty_line(self):
        """Test parsing empty detection line."""
        line = ""
        result = _parse_detection_line(line)
        assert result is None

    def test_parse_detection_with_extra_values(self):
        """Test parsing detection line with extra values (should still work)."""
        line = "0 0.5 0.5 0.3 0.2 0.9 0.8"
        result = _parse_detection_line(line)
        assert result is not None
        assert result["class_id"] == 0


class TestParsePoseLine:
    """Tests for _parse_pose_line function."""

    def test_parse_valid_pose_line_with_keypoints(self):
        """Test parsing a valid pose annotation with keypoints."""
        line = "0 0.5 0.5 0.3 0.4 0.6 0.7 2 0.65 0.75 2 0.7 0.8 1"
        result = _parse_pose_line(line)

        assert result is not None
        assert result["class_id"] == 0
        assert result["x_center"] == 0.5
        assert result["y_center"] == 0.5
        assert result["width"] == 0.3
        assert result["height"] == 0.4
        assert len(result["keypoints"]) == 3

        # Check first keypoint
        assert result["keypoints"][0]["x"] == 0.6
        assert result["keypoints"][0]["y"] == 0.7
        assert result["keypoints"][0]["visibility"] == 2

    def test_parse_pose_line_without_keypoints(self):
        """Test parsing pose line with only bbox (no keypoints)."""
        line = "0 0.5 0.5 0.3 0.4"
        result = _parse_pose_line(line)

        assert result is not None
        assert result["class_id"] == 0
        assert result["keypoints"] == []

    def test_parse_pose_line_with_incomplete_keypoint_triplet(self):
        """Test parsing pose line with incomplete keypoint data."""
        line = "0 0.5 0.5 0.3 0.4 0.6 0.7"  # Missing visibility
        result = _parse_pose_line(line)

        assert result is not None
        assert len(result["keypoints"]) == 0  # Incomplete triplet ignored

    def test_parse_pose_insufficient_values(self):
        """Test parsing pose line with insufficient values."""
        line = "0 0.5 0.5"  # Missing width and height
        result = _parse_pose_line(line)
        assert result is None


class TestParseSegmentationLine:
    """Tests for _parse_segmentation_line function."""

    def test_parse_valid_segmentation_line(self):
        """Test parsing a valid segmentation annotation."""
        line = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"
        result = _parse_segmentation_line(line)

        assert result is not None
        assert result["class_id"] == 0
        assert len(result["points"]) == 4

        # Check first point
        assert result["points"][0]["x"] == 0.1
        assert result["points"][0]["y"] == 0.1

        # Check last point
        assert result["points"][3]["x"] == 0.1
        assert result["points"][3]["y"] == 0.9

    def test_parse_segmentation_with_many_points(self):
        """Test parsing segmentation with many polygon points."""
        # Create a polygon with 10 points
        coords = " ".join([f"{i * 0.1} {i * 0.1}" for i in range(10)])
        line = f"0 {coords}"
        result = _parse_segmentation_line(line)

        assert result is not None
        assert result["class_id"] == 0
        assert len(result["points"]) == 10

    def test_parse_segmentation_with_few_points(self):
        """Test parsing segmentation with too few points (< 3)."""
        line = "0 0.1 0.1 0.9 0.1"  # Only 2 points
        result = _parse_segmentation_line(line)
        assert result is None

    def test_parse_segmentation_with_odd_coordinates(self):
        """Test parsing segmentation with odd number of coordinates."""
        line = "0 0.1 0.1 0.9 0.1 0.9"  # Odd number (5 values after class_id)
        result = _parse_segmentation_line(line)
        assert result is None


class TestParseOBBLine:
    """Tests for _parse_obb_line function."""

    def test_parse_valid_obb_line(self):
        """Test parsing a valid OBB annotation (4 corner points)."""
        line = "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7"
        result = _parse_obb_line(line)

        assert result is not None
        assert result["class_id"] == 0
        assert len(result["points"]) == 4

        # Check all four corners
        assert result["points"][0]["x"] == 0.3
        assert result["points"][0]["y"] == 0.3
        assert result["points"][3]["x"] == 0.3
        assert result["points"][3]["y"] == 0.7

    def test_parse_obb_with_incorrect_point_count(self):
        """Test parsing OBB with wrong number of points."""
        line = "0 0.3 0.3 0.7 0.3 0.7 0.7"  # Only 3 points
        result = _parse_obb_line(line)
        assert result is None

    def test_parse_obb_with_too_many_points(self):
        """Test parsing OBB with too many points."""
        line = "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4 0.5 0.5"  # 5 points
        result = _parse_obb_line(line)
        assert result is None


class TestParseYoloAnnotation:
    """Tests for parse_yolo_annotation function."""

    def test_parse_detection_file(self, tmp_path):
        """Test parsing a detection annotation file."""
        label_file = tmp_path / "detection.txt"
        label_file.write_text("0 0.5 0.5 0.3 0.2\n1 0.7 0.7 0.2 0.3\n")

        result = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)

        assert result is not None
        assert len(result) == 2
        assert result[0]["class_id"] == 0
        assert result[1]["class_id"] == 1

    def test_parse_segmentation_file(self, tmp_path):
        """Test parsing a segmentation annotation file."""
        label_file = tmp_path / "segmentation.txt"
        label_file.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")

        result = parse_yolo_annotation(str(label_file), DatasetTask.SEGMENTATION)

        assert result is not None
        assert len(result) == 1
        assert result[0]["class_id"] == 0
        assert len(result[0]["points"]) == 4

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty annotation file."""
        label_file = tmp_path / "empty.txt"
        label_file.write_text("")

        result = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)
        assert result is None

    def test_parse_file_with_blank_lines(self, tmp_path):
        """Test parsing file with blank lines and comments."""
        label_file = tmp_path / "with_blanks.txt"
        label_file.write_text("0 0.5 0.5 0.3 0.2\n\n  \n1 0.7 0.7 0.2 0.3\n")

        result = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)

        assert result is not None
        assert len(result) == 2

    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent file returns None."""
        result = parse_yolo_annotation("/nonexistent/file.txt", DatasetTask.DETECTION)
        assert result is None

    def test_parse_file_with_mixed_valid_invalid_lines(self, tmp_path):
        """Test parsing file with mix of valid and invalid lines."""
        label_file = tmp_path / "mixed.txt"
        label_file.write_text("0 0.5 0.5 0.3 0.2\ninvalid line\n1 0.7 0.7 0.2 0.3\n")

        result = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)

        # Should parse valid lines and skip invalid ones
        assert result is not None
        assert len(result) == 2

    def test_parse_pose_file(self, tmp_path):
        """Test parsing a pose annotation file."""
        label_file = tmp_path / "pose.txt"
        label_file.write_text("0 0.5 0.5 0.3 0.4 0.6 0.7 2 0.65 0.75 2\n")

        result = parse_yolo_annotation(str(label_file), DatasetTask.POSE)

        assert result is not None
        assert len(result) == 1
        assert result[0]["class_id"] == 0
        assert len(result[0]["keypoints"]) == 2

    def test_parse_obb_file(self, tmp_path):
        """Test parsing an OBB annotation file."""
        label_file = tmp_path / "obb.txt"
        label_file.write_text("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")

        result = parse_yolo_annotation(str(label_file), DatasetTask.OBB)

        assert result is not None
        assert len(result) == 1
        assert result[0]["class_id"] == 0
        assert len(result[0]["points"]) == 4
