"""Load YOLO datasets into FiftyOne."""

import os
from dataclasses import dataclass

import fiftyone as fo
import yaml
from tqdm import tqdm

from yolo_scout.core.config import Config
from yolo_scout.core.constants import (
    DATASET_SPLITS,
    DETECTION_FIELD,
    SUPPORTED_IMAGE_FORMATS,
    THUMBNAIL_PATH_KEY,
    get_field_name,
)
from yolo_scout.core.enums import DatasetTask
from yolo_scout.dataset.converter import (
    create_detection_from_keypoint,
    yolo_to_fiftyone,
)
from yolo_scout.dataset.metadata import extract_image_metadata
from yolo_scout.dataset.parser import parse_yolo_annotation
from yolo_scout.utils.logger import logger
from yolo_scout.utils.path_utils import get_image_name
from yolo_scout.visualization.iou import compute_iou_scores


@dataclass
class SplitInfo:
    """Information about a dataset split."""

    name: str
    images_dir: str
    labels_dir: str | None


def load_yolo_dataset(config: Config) -> fo.Dataset:
    """Load a YOLO dataset into FiftyOne, returning a cached one if it exists."""
    dataset_path = config.data
    dataset_name = config.name
    dataset_task = config.task

    if dataset_name in fo.list_datasets():
        logger.info(f"Loading existing dataset '{dataset_name}'")
        return fo.load_dataset(name=dataset_name)

    logger.info(f"Creating new dataset '{dataset_name}' from path: {dataset_path} with task: {dataset_task.name}")

    # Validate dataset path
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Dataset path '{dataset_path}' does not exist or is not a directory")

    # Load class names
    class_names = _load_class_names(dataset_path=dataset_path, dataset_task=dataset_task)
    logger.info(f"Found {len(class_names)} classes: {class_names}")

    # Create FiftyOne dataset
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    if dataset_task != DatasetTask.CLASSIFICATION:
        # Discover splits
        splits = _discover_splits(dataset_path=dataset_path)

        if not splits:
            raise FileNotFoundError(
                "Dataset directory structure not recognized. Expected either:\n"
                "  - images/{split}/ and labels/{split}/, or\n"
                "  - {split}/images/ and {split}/labels/"
            )

        # Process each split
        for split in splits:
            _process_split(dataset=dataset, split=split, class_names=class_names, task=dataset_task)

    else:
        # Process classification dataset
        splits = _discover_classification_splits(dataset_path=dataset_path)
        for split in splits:
            _process_classification_split(dataset=dataset, split=split)

    _configure_dataset_fields(dataset=dataset, task=dataset_task)

    # Configure app settings
    dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
    dataset.app_config.grid_media_field = "thumbnail_path"

    dataset.info = {"class_names": class_names}
    dataset.save()

    logger.info(f"Dataset created with {len(dataset)} total samples")

    return dataset


def _load_class_names(dataset_path: str, dataset_task: DatasetTask) -> list[str]:
    """Load class names from data.yaml or dataset.yaml."""
    if dataset_task == DatasetTask.CLASSIFICATION:
        # Browse subdirectories in train/val/test for class names
        class_names = set()
        for split in DATASET_SPLITS:
            split_dir = os.path.join(dataset_path, split)
            if os.path.exists(split_dir):
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_dir):
                        class_names.add(class_name)

        return sorted(list(class_names))

    else:
        for yaml_name in ["data.yaml", "dataset.yaml"]:
            yaml_path = os.path.join(dataset_path, yaml_name)

            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f)
                        names = data.get("names", [])

                        if isinstance(names, dict):
                            return [names[i] for i in sorted(names.keys())]

                        return names
                except Exception as e:
                    logger.error(f"Failed to load class names from {yaml_path}: {e}")
                    raise
    return []


def _discover_splits(dataset_path: str) -> list[SplitInfo]:
    """Discover dataset splits in the directory structure."""
    splits = []

    images_base = os.path.join(dataset_path, "images")
    labels_base = os.path.join(dataset_path, "labels")

    # Structure 1: images/train, images/val, labels/train, labels/val
    if os.path.exists(images_base):
        for split_name in DATASET_SPLITS:
            images_dir = os.path.join(images_base, split_name)
            labels_dir = os.path.join(labels_base, split_name)

            if os.path.exists(images_dir):
                splits.append(
                    SplitInfo(
                        name=split_name,
                        images_dir=images_dir,
                        labels_dir=labels_dir if os.path.exists(labels_dir) else None,
                    )
                )

    # Structure 2: train/images, train/labels, val/images, val/labels
    else:
        for split_name in DATASET_SPLITS:
            split_dir = os.path.join(dataset_path, split_name)

            if os.path.exists(split_dir):
                images_dir = os.path.join(split_dir, "images")
                labels_dir = os.path.join(split_dir, "labels")

                if os.path.exists(images_dir):
                    splits.append(
                        SplitInfo(
                            name=split_name,
                            images_dir=images_dir,
                            labels_dir=labels_dir if os.path.exists(labels_dir) else None,
                        )
                    )

    return splits


def _discover_classification_splits(dataset_path: str) -> list[SplitInfo]:
    """Discover classification dataset splits."""
    splits = []

    for split_name in DATASET_SPLITS:
        split_dir = os.path.join(dataset_path, split_name)
        if os.path.exists(split_dir) and os.path.isdir(split_dir):
            splits.append(SplitInfo(name=split_name, images_dir=split_dir, labels_dir=None))

    return splits


def _process_split(
    dataset: fo.Dataset,
    split: SplitInfo,
    class_names: list[str],
    task: DatasetTask,
) -> None:
    """Process a single dataset split."""
    logger.info(f"\nProcessing {split.name} split:")

    # Get all image files
    image_files = sorted([f for f in os.listdir(split.images_dir) if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)])

    samples = []

    for img_file in tqdm(image_files, desc=f"Loading {split.name} images"):
        sample = _create_sample(img_file=img_file, split=split, class_names=class_names, task=task)
        if sample:
            samples.append(sample)

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} samples from {split.name} split")


def _create_sample(
    img_file: str,
    split: SplitInfo,
    class_names: list[str],
    task: DatasetTask,
) -> fo.Sample | None:
    """Create a FiftyOne sample from an image file."""
    image_path = os.path.join(split.images_dir, img_file)

    # Create sample
    sample = fo.Sample(filepath=image_path)
    sample.tags.append(split.name)

    try:
        # Add metadata
        sample.metadata = extract_image_metadata(filepath=image_path)

        # Get label path
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(split.labels_dir, label_file) if split.labels_dir else None

        # Parse annotations
        annotations = parse_yolo_annotation(label_path=label_path, task=task) if label_path else None
        object_count = 0

        # Convert to FiftyOne labels
        if annotations:
            labels = yolo_to_fiftyone(
                annotations=annotations,
                task=task,
                class_names=class_names,
                image_width=sample.metadata.width,
                image_height=sample.metadata.height,
                split=split.name,
                image_path=image_path,
                label_path=label_path,
            )

            if labels:
                field = get_field_name(task=task)
                compute_iou_scores(labels=labels, dataset_task=task)
                sample[field] = labels

                # For pose, also add bounding boxes
                if task == DatasetTask.POSE:
                    detection_labels = _create_detections_from_keypoints(
                        keypoints=labels,
                        image_width=sample.metadata.width,
                        image_height=sample.metadata.height,
                        image_path=image_path,
                        label_path=label_path,
                    )
                    compute_iou_scores(labels=detection_labels, dataset_task=DatasetTask.DETECTION)
                    sample[DETECTION_FIELD] = detection_labels

                object_count = _get_object_count(labels=labels)

        sample["image_path"] = image_path
        sample["label_path"] = label_path if label_path else None
        sample["image_name"] = get_image_name(image_path)
        sample["object_count"] = object_count

        return sample

    except Exception as e:
        logger.warning(f"Failed to process {img_file}: {e}")
        return None


def _create_detections_from_keypoints(
    keypoints: fo.Keypoints,
    image_width: int,
    image_height: int,
    image_path: str,
    label_path: str,
) -> fo.Detections:
    """Create detections from keypoints for pose estimation."""
    detections = []
    for kp in keypoints.keypoints:
        det = create_detection_from_keypoint(
            keypoint=kp,
            image_width=image_width,
            image_height=image_height,
            image_path=image_path,
            label_path=label_path,
        )
        detections.append(det)

    return fo.Detections(detections=detections)


def _get_object_count(labels: fo.Label | None) -> int:
    """Get the number of objects from labels."""
    if labels is None:
        return 0

    try:
        if hasattr(labels, "detections"):
            return len(labels.detections) if labels.detections else 0
        elif hasattr(labels, "polylines"):
            return len(labels.polylines) if labels.polylines else 0
        elif hasattr(labels, "keypoints"):
            return len(labels.keypoints) if labels.keypoints else 0
        elif hasattr(labels, "label"):
            return 1 if labels.label else 0

    except Exception:
        pass

    return 0


def _process_classification_split(
    dataset: fo.Dataset,
    split: SplitInfo,
) -> None:
    """Process a classification dataset split."""
    logger.info(f"\nProcessing {split.name} split:")

    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(split.images_dir) if os.path.isdir(os.path.join(split.images_dir, d))])

    if not class_dirs:
        logger.warning(f"No class directories found in {split.images_dir}, skipping")
        return

    logger.info(f"Found {len(class_dirs)} classes in {split.name}: {class_dirs}")

    samples = []

    for class_name in class_dirs:
        class_dir = os.path.join(split.images_dir, class_name)

        # Get all image files
        image_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)])

        for img_file in tqdm(image_files, desc=f"Loading {split.name}/{class_name} images"):
            image_path = os.path.join(class_dir, img_file)

            # Create sample
            sample = fo.Sample(filepath=image_path)

            try:
                sample.metadata = extract_image_metadata(filepath=image_path)

                # Add classification label
                field_name = get_field_name(task=DatasetTask.CLASSIFICATION)
                sample[field_name] = fo.Classification(label=class_name, tags=[split.name])

                sample["image_path"] = image_path
                sample["image_name"] = get_image_name(image_path)
                sample.tags.append(split.name)

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} samples from {split.name} split")


def _configure_dataset_fields(dataset: fo.Dataset, task: DatasetTask) -> None:
    """Add additional fields to the dataset based on the task."""

    dataset.add_sample_field(field_name="image_name", ftype=fo.StringField)
    dataset.add_sample_field(field_name="image_path", ftype=fo.StringField)
    dataset.add_sample_field(field_name="label_path", ftype=fo.StringField)

    dataset.add_sample_field(field_name="aspect_ratio", ftype=fo.FloatField)
    dataset.add_sample_field(field_name="blurriness", ftype=fo.FloatField)
    dataset.add_sample_field(field_name="brightness", ftype=fo.FloatField)
    dataset.add_sample_field(field_name="entropy", ftype=fo.FloatField)
    dataset.add_sample_field(field_name="object_count", ftype=fo.IntField)

    dataset.add_sample_field(field_name=THUMBNAIL_PATH_KEY, ftype=fo.StringField)

    if task == DatasetTask.CLASSIFICATION:
        return
    elif task in (DatasetTask.DETECTION, DatasetTask.POSE):
        field_name = DETECTION_FIELD
        base = f"{field_name}.detections"
    else:
        field_name = get_field_name(task=task)
        base = f"{field_name}.polylines"

    if field_name not in dataset.get_field_schema():
        logger.warning(f"No '{field_name}' field found because no annotations were loaded, skipping field setup")
        return

    dataset.add_sample_field(field_name=f"{base}.area", ftype=fo.IntField)
    dataset.add_sample_field(field_name=f"{base}.width", ftype=fo.IntField)
    dataset.add_sample_field(field_name=f"{base}.height", ftype=fo.IntField)
    dataset.add_sample_field(field_name=f"{base}.aspect_ratio", ftype=fo.FloatField)
    dataset.add_sample_field(field_name=f"{base}.blurriness", ftype=fo.FloatField)
    dataset.add_sample_field(field_name=f"{base}.brightness", ftype=fo.FloatField)
    dataset.add_sample_field(field_name=f"{base}.entropy", ftype=fo.FloatField)
    dataset.add_sample_field(field_name=f"{base}.iou_score", ftype=fo.FloatField)

    if task in (DatasetTask.POSE, DatasetTask.SEGMENTATION):
        dataset.add_sample_field(field_name=f"{base}.num_keypoints", ftype=fo.IntField)
