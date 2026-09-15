"""Integration tests for dataset loading pipeline."""

import os

import cv2
import fiftyone as fo
import numpy as np
import pytest

from yolo_scout.core.config import Config
from yolo_scout.core.constants import CORRUPTED_TAG
from yolo_scout.core.enums import DatasetTask, EmbeddingsModel
from yolo_scout.dataset.loader import load_yolo_dataset


def _make_config(data: str, task: DatasetTask, name: str, tmp_path) -> Config:
    return Config(
        data=data,
        task=task,
        name=name,
        reload=False,
        dataset_dir=str(tmp_path / "datasets"),
        skip_embeddings=True,
        model=EmbeddingsModel.OPENAI_CLIP,
        batch=16,
        mask_background=True,
        thumbnail_dir=str(tmp_path / "thumbnails"),
        thumbnail_width=100,
        skip_quality=True,
        port=5151,
        skip_launch=True,
        verbose=False,
    )


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetLoading:
    """Integration tests for dataset loading."""

    def test_load_detection_dataset(self, detect_dataset, tmp_path):
        """Test loading a real detection dataset."""
        dataset_name = "test_detect_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            assert dataset is not None
            assert dataset.name == dataset_name
            assert len(dataset) > 0

            sample = dataset.first()
            assert sample is not None
            assert "bounding_boxes" in sample
            assert "image_path" in sample
            assert "object_count" in sample

            assert "class_names" in dataset.info
            assert len(dataset.info["class_names"]) > 0

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_classification_dataset(self, classify_dataset, tmp_path):
        """Test loading a real classification dataset."""
        dataset_name = "test_classify_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(classify_dataset), DatasetTask.CLASSIFICATION, dataset_name, tmp_path)
        )

        try:
            assert dataset is not None
            assert len(dataset) > 0

            sample = dataset.first()
            assert sample is not None
            assert "cls_label" in sample
            assert sample["cls_label"] is not None

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_segmentation_dataset(self, segment_dataset, tmp_path):
        """Test loading a real segmentation dataset."""
        dataset_name = "test_segment_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(segment_dataset), DatasetTask.SEGMENTATION, dataset_name, tmp_path)
        )

        try:
            assert dataset is not None
            assert len(dataset) > 0

            sample = dataset.first()
            assert sample is not None
            assert "seg_polygons" in sample

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_pose_dataset(self, pose_dataset, tmp_path):
        """Test loading a real pose dataset."""
        dataset_name = "test_pose_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(pose_dataset), DatasetTask.POSE, dataset_name, tmp_path))

        try:
            assert dataset is not None
            assert len(dataset) > 0

            sample = dataset.first()
            assert sample is not None
            assert "pose_keypoints" in sample
            assert "bounding_boxes" in sample

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_obb_dataset(self, obb_dataset, tmp_path):
        """Test loading a real OBB dataset."""
        dataset_name = "test_obb_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(obb_dataset), DatasetTask.OBB, dataset_name, tmp_path))

        try:
            assert dataset is not None
            assert len(dataset) > 0

            sample = dataset.first()
            assert sample is not None
            assert "obb_bounding_boxes" in sample

        finally:
            fo.delete_dataset(dataset_name)

    def test_dataset_caching(self, detect_dataset, tmp_path):
        """Test that dataset loading is cached correctly."""
        dataset_name = "test_detect_caching"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        config = _make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path)

        dataset1 = load_yolo_dataset(config)
        dataset2 = load_yolo_dataset(config)

        try:
            assert dataset1.name == dataset2.name
            assert len(dataset1) == len(dataset2)

        finally:
            fo.delete_dataset(dataset_name)

    def test_force_reload(self, detect_dataset, tmp_path):
        """Test that deleting and reloading recreates the dataset."""
        dataset_name = "test_detect_force_reload"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        config = _make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path)
        load_yolo_dataset(config)

        fo.delete_dataset(dataset_name)
        dataset = load_yolo_dataset(config)

        try:
            assert dataset is not None
            assert len(dataset) > 0

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.integration
class TestCorruptedImageHandling:
    """Corrupted images must be kept and tagged, never silently dropped."""

    @staticmethod
    def _write_valid_jpeg(path: str) -> None:
        # Random noise, not a flat color: a uniform image JPEG-compresses to almost nothing,
        # leaving too little data for a truncated copy to keep a readable header.
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _build_split(self, tmp_path) -> str:
        """Build a minimal detection dataset with a valid, a truncated, and a garbage image."""
        data_dir = tmp_path / "corrupted_dataset"
        img_dir = data_dir / "images" / "train"
        img_dir.mkdir(parents=True)

        good_path = str(img_dir / "good.jpg")
        self._write_valid_jpeg(good_path)

        with open(good_path, "rb") as f:
            valid_bytes = bytearray(f.read())

        # Header-readable but truncated: PIL's `.load()` raises, but `Image.open().size` doesn't.
        truncated_path = str(img_dir / "truncated.jpg")
        with open(truncated_path, "wb") as f:
            f.write(valid_bytes[: len(valid_bytes) // 2])

        # Not a real image at all: even the header read fails.
        garbage_path = str(img_dir / "garbage.jpg")
        with open(garbage_path, "wb") as f:
            f.write(b"not an image" * 5)

        return str(data_dir)

    def test_corrupted_images_are_tagged_and_kept(self, tmp_path):
        """All three images must remain in the dataset; only the bad ones get tagged."""
        dataset_name = "test_corrupted_images"
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        data_dir = self._build_split(tmp_path)
        dataset = load_yolo_dataset(_make_config(data_dir, DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            assert len(dataset) == 3, "corrupted images must be kept, not dropped"

            corrupted = dataset.match_tags(CORRUPTED_TAG)
            clean = dataset.match_tags(CORRUPTED_TAG, bool=False)
            assert {os.path.basename(s.filepath) for s in corrupted} == {"truncated.jpg", "garbage.jpg"}
            assert {os.path.basename(s.filepath) for s in clean} == {"good.jpg"}

            # Truncated: header was readable, so metadata should still be present.
            truncated_sample = next(s for s in corrupted if os.path.basename(s.filepath) == "truncated.jpg")
            assert truncated_sample.metadata is not None

            # Garbage: not decodable at all, not even the header.
            garbage_sample = next(s for s in corrupted if os.path.basename(s.filepath) == "garbage.jpg")
            assert garbage_sample.metadata is None

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetStructure:
    """Test that loaded datasets have correct structure."""

    def test_detection_dataset_fields(self, detect_dataset, tmp_path):
        """Test that detection dataset has all expected fields."""
        dataset_name = "test_detect_fields"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            samples_with_detections = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_detections) > 0

            sample = samples_with_detections[0]
            detections = sample["bounding_boxes"]

            if detections and len(detections.detections) > 0:
                detection = detections.detections[0]
                assert hasattr(detection, "label")
                assert hasattr(detection, "bounding_box")
                assert "area" in detection
                assert "aspect_ratio" in detection
                assert "width" in detection
                assert "height" in detection
                assert "iou_score" in detection

        finally:
            fo.delete_dataset(dataset_name)

    def test_segmentation_dataset_fields(self, segment_dataset, tmp_path):
        """Test that segmentation dataset has all expected fields."""
        dataset_name = "test_segment_fields"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(segment_dataset), DatasetTask.SEGMENTATION, dataset_name, tmp_path)
        )

        try:
            samples_with_polygons = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_polygons) > 0

            sample = samples_with_polygons[0]
            polygons = sample["seg_polygons"]

            if polygons and len(polygons.polylines) > 0:
                polygon = polygons.polylines[0]
                assert hasattr(polygon, "label")
                assert hasattr(polygon, "points")
                assert "area" in polygon
                assert "num_keypoints" in polygon
                assert "width" in polygon
                assert "height" in polygon
                assert "iou_score" in polygon

        finally:
            fo.delete_dataset(dataset_name)

    def test_dataset_splits(self, detect_dataset, tmp_path):
        """Test that dataset properly loads different splits."""
        dataset_name = "test_detect_splits"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            all_tags = set()
            for sample in dataset:
                all_tags.update(sample.tags)

            assert len(all_tags) > 0
            assert any(tag in all_tags for tag in ["train", "val", "test"])

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetMetadata:
    """Test dataset metadata extraction and storage."""

    def test_class_names_stored(self, detect_dataset, tmp_path):
        """Test that class names are stored in dataset info."""
        dataset_name = "test_detect_metadata"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            assert "class_names" in dataset.info
            class_names = dataset.info["class_names"]
            assert isinstance(class_names, list)
            assert len(class_names) > 0

        finally:
            fo.delete_dataset(dataset_name)

    def test_image_metadata_extracted(self, detect_dataset, tmp_path):
        """Test that image metadata is extracted for samples."""
        dataset_name = "test_detect_img_metadata"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            sample = dataset.first()
            assert sample.metadata is not None
            assert hasattr(sample.metadata, "width")
            assert hasattr(sample.metadata, "height")
            assert sample.metadata.width > 0
            assert sample.metadata.height > 0

        finally:
            fo.delete_dataset(dataset_name)

    def test_object_count_computed(self, detect_dataset, tmp_path):
        """Test that object count is computed for each sample."""
        dataset_name = "test_detect_obj_count"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            for sample in dataset:
                assert "object_count" in sample
                assert isinstance(sample["object_count"], int)
                assert sample["object_count"] >= 0

                if sample["bounding_boxes"]:
                    actual_count = len(sample["bounding_boxes"].detections)
                    assert sample["object_count"] == actual_count
                else:
                    assert sample["object_count"] == 0

        finally:
            fo.delete_dataset(dataset_name)
