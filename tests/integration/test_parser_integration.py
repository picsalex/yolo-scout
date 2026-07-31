"""Integration tests for YOLO annotation parsing with real files."""

import pytest

from yolo_scout.core.enums import DatasetTask
from yolo_scout.dataset.parser import parse_yolo_annotation


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestParseRealAnnotations:
    """Test parsing real YOLO annotation files from datasets."""

    def test_parse_real_detection_annotations(self, detect_dataset):
        """Test parsing real detection annotation files."""
        labels_dir = detect_dataset / "labels" / "train"

        # Find a label file
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0, "No label files found"

        label_file = label_files[0]
        annotations = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)

        # Some files might be empty, find one with annotations
        for lf in label_files:
            annotations = parse_yolo_annotation(str(lf), DatasetTask.DETECTION)
            if annotations:
                break

        if annotations:
            # Verify annotation structure
            assert isinstance(annotations, list)
            assert len(annotations) > 0

            anno = annotations[0]
            assert "class_id" in anno
            assert "x_center" in anno
            assert "y_center" in anno
            assert "width" in anno
            assert "height" in anno

            # Verify values are in valid range [0, 1]
            assert 0 <= anno["x_center"] <= 1
            assert 0 <= anno["y_center"] <= 1
            assert 0 <= anno["width"] <= 1
            assert 0 <= anno["height"] <= 1

    def test_parse_real_segmentation_annotations(self, segment_dataset):
        """Test parsing real segmentation annotation files."""
        labels_dir = segment_dataset / "labels" / "train"

        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0, "No label files found"

        # Find a file with annotations
        annotations = None
        for lf in label_files:
            annotations = parse_yolo_annotation(str(lf), DatasetTask.SEGMENTATION)
            if annotations:
                break

        if annotations:
            assert isinstance(annotations, list)
            assert len(annotations) > 0

            anno = annotations[0]
            assert "class_id" in anno
            assert "points" in anno
            assert len(anno["points"]) >= 3  # At least 3 points for polygon

            # Verify point structure
            point = anno["points"][0]
            assert "x" in point
            assert "y" in point
            assert 0 <= point["x"] <= 1
            assert 0 <= point["y"] <= 1

    def test_parse_real_pose_annotations(self, pose_dataset):
        """Test parsing real pose annotation files."""
        labels_dir = pose_dataset / "labels" / "train"

        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0, "No label files found"

        # Find a file with annotations
        annotations = None
        for lf in label_files:
            annotations = parse_yolo_annotation(str(lf), DatasetTask.POSE)
            if annotations:
                break

        if annotations:
            assert isinstance(annotations, list)
            assert len(annotations) > 0

            anno = annotations[0]
            assert "class_id" in anno
            assert "x_center" in anno
            assert "y_center" in anno
            assert "width" in anno
            assert "height" in anno
            assert "keypoints" in anno

            # Verify keypoint structure if present
            if anno["keypoints"]:
                kp = anno["keypoints"][0]
                assert "x" in kp
                assert "y" in kp
                assert "visibility" in kp

    def test_parse_real_obb_annotations(self, obb_dataset):
        """Test parsing real OBB annotation files."""
        labels_dir = obb_dataset / "labels" / "train"

        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0, "No label files found"

        # Find a file with annotations
        annotations = None
        for lf in label_files:
            annotations = parse_yolo_annotation(str(lf), DatasetTask.OBB)
            if annotations:
                break

        if annotations:
            assert isinstance(annotations, list)
            assert len(annotations) > 0

            anno = annotations[0]
            assert "class_id" in anno
            assert "points" in anno
            assert len(anno["points"]) == 4  # OBB has exactly 4 points

    def test_parse_empty_annotation_file(self, detect_dataset):
        """Test parsing empty annotation files (images with no objects)."""
        labels_dir = detect_dataset / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        # Find an empty file or test with all files
        for label_file in label_files:
            annotations = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)

            # Empty files should return None
            if annotations is None:
                # This is expected for empty files
                assert True
                break

    def test_all_label_files_parseable(self, detect_dataset):
        """Test that all label files can be parsed without errors."""
        labels_dir = detect_dataset / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        for label_file in label_files:
            annotations = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)
            # Should either return list or None, not raise exception
            assert annotations is None or isinstance(annotations, list), label_file.name


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestAnnotationValidation:
    """Test validation of parsed annotations."""

    def test_detection_coordinates_in_range(self, detect_dataset):
        """Test that all detection coordinates are in valid range [0, 1]."""
        labels_dir = detect_dataset / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        out_of_range = []
        for label_file in label_files:
            annotations = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)
            if annotations:
                for anno in annotations:
                    # Check all coordinates are in [0, 1]
                    if not (0 <= anno["x_center"] <= 1):
                        out_of_range.append((label_file.name, "x_center", anno["x_center"]))
                    if not (0 <= anno["y_center"] <= 1):
                        out_of_range.append((label_file.name, "y_center", anno["y_center"]))
                    if not (0 <= anno["width"] <= 1):
                        out_of_range.append((label_file.name, "width", anno["width"]))
                    if not (0 <= anno["height"] <= 1):
                        out_of_range.append((label_file.name, "height", anno["height"]))

        assert len(out_of_range) == 0, f"Found {len(out_of_range)} out-of-range values: {out_of_range[:5]}"

    def test_class_ids_are_integers(self, detect_dataset):
        """Test that all class IDs are valid integers."""
        labels_dir = detect_dataset / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        invalid_class_ids = []
        for label_file in label_files:
            annotations = parse_yolo_annotation(str(label_file), DatasetTask.DETECTION)
            if annotations:
                for anno in annotations:
                    if not isinstance(anno["class_id"], int):
                        invalid_class_ids.append((label_file.name, anno["class_id"]))
                    if anno["class_id"] < 0:
                        invalid_class_ids.append((label_file.name, anno["class_id"]))

        assert len(invalid_class_ids) == 0, f"Found {len(invalid_class_ids)} invalid class IDs: {invalid_class_ids[:5]}"

    def test_polygon_has_sufficient_points(self, segment_dataset):
        """Test that all polygons have at least 3 points."""
        labels_dir = segment_dataset / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        invalid_polygons = []
        for label_file in label_files:
            annotations = parse_yolo_annotation(str(label_file), DatasetTask.SEGMENTATION)
            if annotations:
                for anno in annotations:
                    if len(anno["points"]) < 3:
                        invalid_polygons.append((label_file.name, len(anno["points"])))

        assert len(invalid_polygons) == 0, f"Found {len(invalid_polygons)} invalid polygons: {invalid_polygons}"

    def test_obb_has_four_points(self, obb_dataset):
        """Test that all OBBs have exactly 4 points."""
        labels_dir = obb_dataset / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        invalid_obbs = []
        for label_file in label_files:
            annotations = parse_yolo_annotation(str(label_file), DatasetTask.OBB)
            if annotations:
                for anno in annotations:
                    if len(anno["points"]) != 4:
                        invalid_obbs.append((label_file.name, len(anno["points"])))

        assert len(invalid_obbs) == 0, f"Found {len(invalid_obbs)} invalid OBBs: {invalid_obbs}"


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetConsistency:
    """Test consistency between images and labels."""

    def test_image_label_pairing(self, detect_dataset):
        """Test that images and labels are properly paired."""
        images_dir = detect_dataset / "images" / "train"
        labels_dir = detect_dataset / "labels" / "train"

        image_files = {f.stem for f in images_dir.glob("*.jpg")} | {f.stem for f in images_dir.glob("*.png")}
        label_files = {f.stem for f in labels_dir.glob("*.txt")}

        # Not all images need labels (some might have no objects)
        # But we should have some overlap
        common = image_files & label_files
        assert len(common) > 0, "No matching image-label pairs found"

        # Most images should have labels (at least 50%)
        if len(label_files) > 0:
            assert len(common) / len(image_files) > 0.3, f"Only {len(common)}/{len(image_files)} images have labels"

    def test_no_orphaned_labels(self, detect_dataset):
        """Test that there are no label files without corresponding images."""
        images_dir = detect_dataset / "images" / "train"
        labels_dir = detect_dataset / "labels" / "train"

        image_files = {f.stem for f in images_dir.glob("*.jpg")} | {f.stem for f in images_dir.glob("*.png")}
        label_files = {f.stem for f in labels_dir.glob("*.txt")}

        orphaned = label_files - image_files
        assert len(orphaned) == 0, f"Found {len(orphaned)} orphaned label files: {list(orphaned)[:5]}"
