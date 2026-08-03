"""Compute image quality metrics for FiftyOne datasets."""

import cv2
import fiftyone as fo
import numpy as np
from tqdm import tqdm

from yolo_scout.core.constants import DETECTION_FIELD, get_field_name, get_patches_attr
from yolo_scout.core.enums import DatasetTask
from yolo_scout.embeddings.preprocessing import iter_patch_crops, limit_worker_cv2_threads, process_sample_patches
from yolo_scout.utils.logger import logger
from yolo_scout.utils.parallel import imap_workers

METRIC_NAMES = ("blurriness", "brightness", "aspect_ratio", "entropy")


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


def _metrics(gray: np.ndarray) -> dict[str, float]:
    """The quality metrics for one grayscale image or crop."""
    return {
        "blurriness": _blurriness(gray),
        "brightness": _brightness(gray),
        "aspect_ratio": _aspect_ratio(gray),
        "entropy": _entropy(gray),
    }


def _image_metrics(filepath: str) -> dict[str, float] | None:
    """Compute one image's quality metrics, or None if it cannot be read."""
    gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return _metrics(gray) if gray is not None else None


def _compute_image_metrics(dataset: fo.Dataset) -> None:
    """Compute image-level metrics in worker processes and write them in bulk.

    Decoding every image is the work here, so it belongs in the pool rather than in a serial
    loop; the results then go back as four `set_values` calls instead of one save per sample.
    """
    ids, filepaths = dataset.values(["id", "filepath"])

    results = imap_workers(_image_metrics, filepaths, initializer=limit_worker_cv2_threads)
    metrics = list(tqdm(results, total=len(filepaths), desc="Image metrics"))

    for name in METRIC_NAMES:
        values = {sample_id: m[name] for sample_id, m in zip(ids, metrics) if m is not None}
        dataset.set_values(name, values, key_field="id")


def _compute_patch_metrics(
    sample_data: tuple[str, str, list, DatasetTask],
    background_color: tuple[int, int, int] = (114, 114, 114),
    mask_background: bool = True,
) -> tuple[str, list[tuple[str, dict[str, float]]]]:
    """Extract crops and compute their quality metrics inside a worker process."""
    sample_id, crops = process_sample_patches(
        sample_data, background_color=background_color, mask_background=mask_background
    )
    metrics = [(patch_id, _metrics(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))) for patch_id, crop in crops]
    return sample_id, metrics


def compute_quality_metrics(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    mask_background: bool,
) -> None:
    """Compute quality metrics for images and patches."""
    logger.info("Computing quality metrics...")

    _compute_image_metrics(dataset=dataset)

    # Patch-level
    if dataset_task == DatasetTask.CLASSIFICATION:
        return

    patches_field = get_field_name(task=dataset_task)
    if dataset_task == DatasetTask.POSE:
        patches_field = DETECTION_FIELD

    patches_attr = get_patches_attr(task=dataset_task)

    def get_patches(sample) -> list:
        patches_obj = sample[patches_field]
        if patches_obj is None:
            return []
        return getattr(patches_obj, patches_attr, None) or []

    metrics_stream = iter_patch_crops(
        dataset=dataset,
        patches_field=patches_field,
        dataset_task=dataset_task,
        mask_background=mask_background,
        worker_func=_compute_patch_metrics,
        desc="Patch metrics",
    )

    for sample_id, metrics_list in metrics_stream:
        sample = dataset[sample_id]
        patches_by_id = {patch.id: patch for patch in get_patches(sample)}
        for patch_id, metrics in metrics_list:
            patch = patches_by_id.get(patch_id)
            if patch is None:
                continue
            for name in METRIC_NAMES:
                patch[name] = metrics[name]
        sample.save()

    logger.info("Quality metrics computed successfully")
