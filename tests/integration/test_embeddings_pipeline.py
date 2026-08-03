"""Integration tests for embeddings computation and storage."""

import fiftyone as fo
import pytest

from yolo_scout.core.config import Config
from yolo_scout.core.constants import DETECTION_FIELD, IMAGE_EMBEDDINGS_KEY, PATCH_EMBEDDINGS_KEY, get_field_name
from yolo_scout.core.enums import DatasetTask, EmbeddingsModel
from yolo_scout.dataset.loader import load_yolo_dataset
from yolo_scout.embeddings.computer import compute_embeddings
from yolo_scout.visualization.quality import compute_quality_metrics


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
@pytest.mark.slow
class TestImageEmbeddings:
    """Test image embeddings computation and storage."""

    def test_image_embeddings_computed_detection(self, detect_dataset, tmp_path):
        """Test that image embeddings are computed for detection dataset."""
        dataset_name = "test_img_emb_detect"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys, f"Image embeddings key not found. Available keys: {brain_keys}"

            brain_info = dataset.get_brain_info(IMAGE_EMBEDDINGS_KEY)
            assert brain_info is not None
            assert hasattr(brain_info, "config")

        finally:
            fo.delete_dataset(dataset_name)

    def test_image_embeddings_for_all_tasks(self, all_datasets, tmp_path):
        """Test that image embeddings work for all dataset tasks."""
        model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
        task_map = {
            "detect": DatasetTask.DETECTION,
            "classify": DatasetTask.CLASSIFICATION,
            "segment": DatasetTask.SEGMENTATION,
            "pose": DatasetTask.POSE,
            "obb": DatasetTask.OBB,
        }

        for task_name, dataset_path in all_datasets.items():
            dataset_name = f"test_img_emb_{task_name}"

            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)

            task = task_map[task_name]

            try:
                dataset = load_yolo_dataset(_make_config(str(dataset_path), task, dataset_name, tmp_path))
                compute_embeddings(dataset=dataset, dataset_task=task, model_kwargs=model_kwargs, batch_size=4)

                brain_keys = dataset.list_brain_runs()
                assert IMAGE_EMBEDDINGS_KEY in brain_keys, f"Image embeddings missing for {task_name}"

            finally:
                if dataset_name in fo.list_datasets():
                    fo.delete_dataset(dataset_name)

    def test_image_embeddings_visualization_exists(self, detect_dataset, tmp_path):
        """Test that visualization (UMAP) is computed for image embeddings."""
        dataset_name = "test_img_emb_viz"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            brain_info = dataset.get_brain_info(IMAGE_EMBEDDINGS_KEY)
            assert brain_info is not None

            config = brain_info.config
            assert hasattr(config, "method") or hasattr(config, "embeddings_field")

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestPatchEmbeddings:
    """Test patch embeddings computation and storage."""

    def test_patch_embeddings_computed_detection(self, detect_dataset, tmp_path):
        """Test that patch embeddings are computed for detection dataset."""
        dataset_name = "test_patch_emb_detect"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            brain_keys = dataset.list_brain_runs()
            assert PATCH_EMBEDDINGS_KEY in brain_keys, f"Patch embeddings key not found. Available keys: {brain_keys}"
            assert dataset.get_brain_info(PATCH_EMBEDDINGS_KEY) is not None

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_detection(self, detect_dataset, tmp_path):
        """Test that patch embeddings use bounding_boxes field for detection."""
        dataset_name = "test_patch_field_detect"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            patches_field = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY).config.patches_field
            assert patches_field == DETECTION_FIELD, f"Expected '{DETECTION_FIELD}', got '{patches_field}'"

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_segmentation(self, segment_dataset, tmp_path):
        """Test that patch embeddings use seg_polygons field for segmentation."""
        dataset_name = "test_patch_field_segment"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(segment_dataset), DatasetTask.SEGMENTATION, dataset_name, tmp_path)
        )

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.SEGMENTATION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            patches_field = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY).config.patches_field
            expected = get_field_name(DatasetTask.SEGMENTATION)
            assert patches_field == expected, f"Expected '{expected}', got '{patches_field}'"

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_pose(self, pose_dataset, tmp_path):
        """Test that patch embeddings use bounding_boxes field for pose (not keypoints)."""
        dataset_name = "test_patch_field_pose"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(pose_dataset), DatasetTask.POSE, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.POSE,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            patches_field = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY).config.patches_field
            assert patches_field == DETECTION_FIELD, f"Expected '{DETECTION_FIELD}' for pose, got '{patches_field}'"

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_obb(self, obb_dataset, tmp_path):
        """Test that patch embeddings use obb_bounding_boxes field for OBB."""
        dataset_name = "test_patch_field_obb"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(obb_dataset), DatasetTask.OBB, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.OBB,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            patches_field = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY).config.patches_field
            expected = get_field_name(DatasetTask.OBB)
            assert patches_field == expected, f"Expected '{expected}', got '{patches_field}'"

        finally:
            fo.delete_dataset(dataset_name)

    def test_classification_no_patch_embeddings(self, classify_dataset, tmp_path):
        """Test that classification dataset does not compute patch embeddings."""
        dataset_name = "test_patch_classify"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(classify_dataset), DatasetTask.CLASSIFICATION, dataset_name, tmp_path)
        )

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.CLASSIFICATION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys, "Image embeddings should exist for classification"
            assert PATCH_EMBEDDINGS_KEY not in brain_keys, "Patch embeddings should NOT exist for classification"

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestPatchVisualizationCorrectness:
    """Guards the manual UMAP points path against label mismatches.

    `compute_embeddings` bypasses FiftyOne's own UMAP call (its spectral init
    segfaults at scale) and builds the `points=` argument itself, keyed by patch
    id. If a future refactor breaks that id tracking, this should fail loudly
    here instead of silently mismatching points to patches in production.
    """

    def test_patch_points_match_dataset_labels(self, detect_dataset, tmp_path):
        dataset_name = "test_patch_points_correctness"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            brain_info = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY)
            assert brain_info.config.method == "manual"

            results = dataset.load_brain_results(PATCH_EMBEDDINGS_KEY)

            actual_label_ids = {
                detection.id
                for sample in dataset
                if sample[DETECTION_FIELD] is not None
                for detection in sample[DETECTION_FIELD].detections
            }

            assert len(results.points) == len(results.label_ids), "points and label_ids must be the same length"
            assert len(results.label_ids) == len(actual_label_ids), (
                f"Expected {len(actual_label_ids)} labeled points, got {len(results.label_ids)}"
            )
            assert set(results.label_ids) == actual_label_ids, (
                "label_ids returned by the visualization do not match the dataset's actual detection ids"
            )

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestInvalidPatchHandling:
    """Guards patch-level processing against one malformed patch corrupting its siblings.

    A patch with no bounding_box is expected to be excluded from embeddings/metrics
    entirely (not misassigned, not faked). FiftyOne's ODM also coerces an explicitly
    None bounding_box to an empty list on save, so the check must catch that case
    too - and one bad patch must not discard the rest of its sample's valid crops.
    """

    def test_invalid_patch_excluded_without_affecting_siblings(self, detect_dataset, tmp_path):
        dataset_name = "test_invalid_patch_handling"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            sample = dataset.first()
            bad_detection = sample[DETECTION_FIELD].detections[0]
            bad_id = bad_detection.id
            bad_detection.bounding_box = None
            sample.save()

            total_detections = sum(
                len(s[DETECTION_FIELD].detections) for s in dataset if s[DETECTION_FIELD] is not None
            )

            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            results = dataset.load_brain_results(PATCH_EMBEDDINGS_KEY)
            assert bad_id not in results.label_ids, "invalid patch should not get an embedding"
            assert len(results.points) == total_detections - 1, (
                "sibling patches in the same sample must still get embeddings"
            )

            compute_quality_metrics(dataset=dataset, dataset_task=DatasetTask.DETECTION, mask_background=True)
            dataset.reload()
            sample = dataset[sample.id]

            def has_metrics(detection):
                try:
                    return detection.get_field("blurriness") is not None
                except AttributeError:
                    return False

            bad_detection = next(d for d in sample[DETECTION_FIELD].detections if d.id == bad_id)
            siblings = [d for d in sample[DETECTION_FIELD].detections if d.id != bad_id]

            assert not has_metrics(bad_detection), "invalid patch should not get quality metrics"
            assert all(has_metrics(d) for d in siblings), "sibling patches must still get quality metrics"

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingsFieldMapping:
    """Test that embeddings use correct fields for each task type."""

    def test_field_mapping_consistency(self):
        """Test that field mapping is consistent across tasks."""
        assert get_field_name(DatasetTask.DETECTION) == DETECTION_FIELD
        assert get_field_name(DatasetTask.SEGMENTATION) != DETECTION_FIELD
        assert get_field_name(DatasetTask.OBB) != DETECTION_FIELD
        assert get_field_name(DatasetTask.POSE) != DETECTION_FIELD

    def test_all_non_classification_tasks_have_patches(self, all_datasets, tmp_path):
        """Test that all non-classification tasks compute patch embeddings."""
        model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
        task_map = {
            "detect": DatasetTask.DETECTION,
            "segment": DatasetTask.SEGMENTATION,
            "pose": DatasetTask.POSE,
            "obb": DatasetTask.OBB,
        }

        for task_name, task in task_map.items():
            dataset_name = f"test_patches_{task_name}"
            dataset_path = all_datasets[task_name]

            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)

            try:
                dataset = load_yolo_dataset(_make_config(str(dataset_path), task, dataset_name, tmp_path))
                compute_embeddings(dataset=dataset, dataset_task=task, model_kwargs=model_kwargs, batch_size=4)

                brain_keys = dataset.list_brain_runs()
                assert IMAGE_EMBEDDINGS_KEY in brain_keys, f"Missing image embeddings for {task_name}"
                assert PATCH_EMBEDDINGS_KEY in brain_keys, f"Missing patch embeddings for {task_name}"

            finally:
                if dataset_name in fo.list_datasets():
                    fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestBackgroundMasking:
    """Test background masking configuration for embeddings."""

    def test_mask_background_enabled_by_default(self, segment_dataset, tmp_path):
        """Test that background masking is enabled by default for segmentation."""
        dataset_name = "test_mask_bg_default"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(segment_dataset), DatasetTask.SEGMENTATION, dataset_name, tmp_path)
        )

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.SEGMENTATION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
                mask_background=True,
            )

            assert PATCH_EMBEDDINGS_KEY in dataset.list_brain_runs()

        finally:
            fo.delete_dataset(dataset_name)

    def test_mask_background_disabled(self, segment_dataset, tmp_path):
        """Test that background masking can be disabled for segmentation."""
        dataset_name = "test_mask_bg_disabled"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(
            _make_config(str(segment_dataset), DatasetTask.SEGMENTATION, dataset_name, tmp_path)
        )

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.SEGMENTATION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
                mask_background=False,
            )

            assert PATCH_EMBEDDINGS_KEY in dataset.list_brain_runs()

        finally:
            fo.delete_dataset(dataset_name)

    def test_mask_background_obb_task(self, obb_dataset, tmp_path):
        """Test that background masking works for OBB tasks."""
        dataset_name = "test_mask_bg_obb"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(obb_dataset), DatasetTask.OBB, dataset_name, tmp_path))

        try:
            for mask_bg in [True, False]:
                compute_embeddings(
                    dataset=dataset,
                    dataset_task=DatasetTask.OBB,
                    model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                    batch_size=4,
                    mask_background=mask_bg,
                )
                assert PATCH_EMBEDDINGS_KEY in dataset.list_brain_runs()

        finally:
            fo.delete_dataset(dataset_name)

    def test_mask_background_not_applied_to_detection(self, detect_dataset, tmp_path):
        """Test that mask_background parameter doesn't affect detection tasks."""
        dataset_name = "test_mask_bg_detect"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
                mask_background=False,
            )

            assert PATCH_EMBEDDINGS_KEY in dataset.list_brain_runs()

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingsBehavior:
    """Test embeddings computation behavior and edge cases."""

    def test_embeddings_with_small_batch_size(self, detect_dataset, tmp_path):
        """Test that embeddings work with small batch size."""
        dataset_name = "test_emb_small_batch"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=1,
            )

            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)

    def test_embeddings_with_samples_without_objects(self, detect_dataset, tmp_path):
        """Test that embeddings work even when some samples have no objects."""
        dataset_name = "test_emb_no_objects"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        dataset = load_yolo_dataset(_make_config(str(detect_dataset), DatasetTask.DETECTION, dataset_name, tmp_path))

        try:
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=EmbeddingsModel.OPENAI_CLIP.get_model_kwargs(),
                batch_size=4,
            )

            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)
