"""Compute image quality metrics for FiftyOne datasets."""

from typing import Dict, List, Tuple

import cv2
import fiftyone as fo
import numpy as np
from tqdm import tqdm

from yolo_scout.core.constants import DETECTION_FIELD, get_field_name
from yolo_scout.core.enums import DatasetTask
from yolo_scout.embeddings.preprocessing import iter_patch_crops, process_sample_patches
from yolo_scout.utils.logger import logger


def _blurriness(gray: np.ndarray) -> float:
    """
    Laplacian variance — lower = blurrier.
    Returns the inverse so higher = blurrier for easier interpretation.
    """
    return 1.0 / (1.0 + cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: np.ndarray) -> float:
    """
    Mean pixel intensity normalized to [0, 1].
    0 = fully dark, 1 = fully bright.
    """
    return float(gray.mean()) / 255.0


def _aspect_ratio(gray: np.ndarray) -> float:
    """
    Width-to-height ratio derived from the array shape.
    Values > 1 are wider than tall, < 1 are taller than wide.
    """
    h, w = gray.shape[:2]
    return round(w / h, 2) if h != 0 else 0.0


def _entropy(gray: np.ndarray) -> float:
    """
    Shannon entropy of the pixel intensity histogram.
    Higher = more texture/complexity, lower = uniform/flat regions.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    non_zero = hist[hist > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))


def _compute_patch_metrics(
    sample_data: Tuple[str, str, str, List, DatasetTask],
    background_color: Tuple[int, int, int] = (114, 114, 114),
    mask_background: bool = True,
) -> tuple[str, list[dict[str, float]]]:
    """Extract crops and compute their quality metrics inside a worker process."""
    sample_id, crops = process_sample_patches(
        sample_data, background_color=background_color, mask_background=mask_background
    )
    metrics = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        metrics.append(
            {
                "blurriness": _blurriness(gray),
                "brightness": _brightness(gray),
                "aspect_ratio": _aspect_ratio(gray),
                "entropy": _entropy(gray),
            }
        )
    return sample_id, metrics


def compute_quality_metrics(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    mask_background: bool,
) -> None:
    """Compute quality metrics for images and patches."""
    cv2.setNumThreads(1)
    logger.info("Computing quality metrics...")

    # Image-level
    for sample in tqdm(dataset, desc="Image metrics"):
        gray = cv2.imread(sample.filepath, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        sample["blurriness"] = _blurriness(gray)
        sample["brightness"] = _brightness(gray)
        sample["aspect_ratio"] = _aspect_ratio(gray)
        sample["entropy"] = _entropy(gray)
        sample.save()

    # Patch-level
    if dataset_task == DatasetTask.CLASSIFICATION:
        return

    patches_field = get_field_name(task=dataset_task)
    if dataset_task == DatasetTask.POSE:
        patches_field = DETECTION_FIELD

    is_detection_like = dataset_task in [DatasetTask.DETECTION, DatasetTask.POSE]

    def get_patches(sample):
        obj = sample[patches_field]
        if obj is None:
            return []
        return (obj.detections if is_detection_like else obj.polylines) or []

    metrics_stream = iter_patch_crops(
        dataset=dataset,
        patches_field=patches_field,
        dataset_task=dataset_task,
        mask_background=mask_background,
        worker_func=_compute_patch_metrics,
    )

    for sample_id, metrics_list in tqdm(metrics_stream, desc="Patch metrics"):
        sample = dataset[sample_id]
        patches = get_patches(sample)
        for patch, metrics in zip(patches, metrics_list):
            patch["blurriness"] = metrics["blurriness"]
            patch["brightness"] = metrics["brightness"]
            patch["aspect_ratio"] = metrics["aspect_ratio"]
            patch["entropy"] = metrics["entropy"]
        sample.save()

    logger.info("Quality metrics computed successfully")
