"""Image preprocessing for embeddings computation."""

from collections import deque
from collections.abc import Callable, Iterator
from functools import partial
from itertools import islice
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import cv2
import fiftyone as fo
import numpy as np
from tqdm import tqdm

from yolo_scout.core.enums import DatasetTask
from yolo_scout.utils.logger import logger

# Samples allowed in flight per worker. Pool.imap applies no backpressure, so workers
# race through the whole dataset and every crop sits in the parent until consumed -
# the exact leak this module streams to avoid.
PENDING_SAMPLES_PER_WORKER = 4


def create_mask_from_polyline(
    polyline_points: list[list[float]],
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Create a binary mask from polyline points.

    Args:
        polyline_points: List of normalized [x, y] coordinates (values in [0, 1])
        image_shape: Image shape as (height, width, channels)

    Returns:
        Binary mask as uint8 array (height, width) with 255 for object, 0 for background
    """
    height, width = image_shape[:2]

    # Convert normalized coordinates to pixel coordinates
    points_pixels = np.array(
        [[int(x * width), int(y * height)] for x, y in polyline_points],
        dtype=np.int32,
    )

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill polygon
    cv2.fillPoly(mask, [points_pixels], 255)

    return mask


def apply_background_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Apply background masking to an image.

    Args:
        image: Input image as (H, W, C) numpy array
        mask: Binary mask as (H, W) array with 255 for object, 0 for background
        background_color: RGB tuple for background fill color

    Returns:
        Masked image with background replaced by background_color
    """
    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create background image
    background = np.full_like(image, background_color, dtype=np.uint8)

    # Blend: where mask is 255, keep original; where mask is 0, use background
    masked_image = np.where(mask_3ch == 255, image, background)

    return masked_image.astype(np.uint8)


def get_bbox_from_polyline(
    polyline_points: list[list[float]],
) -> tuple[float, float, float, float]:
    """
    Compute normalized bounding box from polyline points.

    Args:
        polyline_points: List of normalized [x, y] coordinates

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in normalized coordinates [0, 1]
    """
    if not polyline_points:
        return 0.0, 0.0, 1.0, 1.0

    x_coords = [pt[0] for pt in polyline_points]
    y_coords = [pt[1] for pt in polyline_points]

    x_min = max(0.0, min(x_coords))
    y_min = max(0.0, min(y_coords))
    x_max = min(1.0, max(x_coords))
    y_max = min(1.0, max(y_coords))

    return x_min, y_min, x_max, y_max


def normalize_bbox(
    bbox: list[float],
) -> tuple[float, float, float, float]:
    """
    Convert FiftyOne bbox format to (x_min, y_min, x_max, y_max).

    Args:
        bbox: FiftyOne bbox as [x_top_left, y_top_left, width, height]

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in normalized coordinates
    """
    x, y, w, h = bbox
    return x, y, x + w, y + h


def crop_to_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Crop image to bounding box coordinates.

    Args:
        image: Input image as (H, W, C) array
        bbox: Normalized bbox as (x_min, y_min, x_max, y_max)
        image_shape: Original image shape (height, width)

    Returns:
        Cropped image
    """
    height, width = image_shape[:2]

    x_min, y_min, x_max, y_max = bbox

    # Convert to pixel coordinates
    x1 = int(x_min * width)
    y1 = int(y_min * height)
    x2 = int(x_max * width)
    y2 = int(y_max * height)

    # Ensure valid crop dimensions
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return image[y1:y2, x1:x2]


def create_crop_for_detection(
    image: np.ndarray,
    bbox: list[float],
) -> np.ndarray:
    """
    Create a crop for detection/pose tasks (no masking, just bbox crop).

    Args:
        image: Original image as (H, W, C) array
        bbox: FiftyOne bbox as [x_top_left, y_top_left, width, height]

    Returns:
        Cropped image
    """
    normalized_bbox = normalize_bbox(bbox)
    return crop_to_bbox(image, normalized_bbox, image.shape)


def create_masked_crop_for_polyline(
    image: np.ndarray,
    polyline_points: list[list[float]],
    background_color: tuple[int, int, int] = (114, 114, 114),
    mask_background: bool = True,
) -> np.ndarray:
    """
    Create a masked and cropped image for a polyline (segment/obb tasks).

    Args:
        image: Original image as (H, W, C) array
        polyline_points: Normalized polyline coordinates
        background_color: RGB background color for masking
        mask_background: Whether to mask the background (default: True)

    Returns:
        Cropped and masked image
    """
    # Remove duplicate last point if polyline is closed
    if len(polyline_points) > 1 and polyline_points[0] == polyline_points[-1]:
        polyline_points = polyline_points[:-1]

    if len(polyline_points) < 3:
        # Invalid polygon, return a small blank image
        return np.full((10, 10, 3), background_color, dtype=np.uint8)

    # Apply background masking if enabled
    if mask_background:
        # Create mask
        mask = create_mask_from_polyline(polyline_points, image.shape)

        # Apply background masking
        masked_image = apply_background_mask(image, mask, background_color)
    else:
        # No masking, use original image
        masked_image = image

    # Get bounding box and crop
    bbox = get_bbox_from_polyline(polyline_points)
    cropped_image = crop_to_bbox(masked_image, bbox, image.shape)

    return cropped_image


def _limit_worker_cv2_threads() -> None:
    # Otherwise each of the cpu_count() - 1 worker processes also tries to use every core.
    cv2.setNumThreads(1)


def _imap_bounded(
    pool: Pool,
    func: Callable[[tuple], tuple[str, list]],
    items: list[tuple],
    max_pending: int,
) -> Iterator[tuple[str, list]]:
    """Like `pool.imap`, but never lets more than `max_pending` results pile up unconsumed.

    `pool.imap` submits the whole iterable as fast as the workers can drain it and caches
    every result in the parent, so a consumer slower than the workers (model inference vs.
    cropping) ends up holding the entire dataset in memory. Submitting one replacement task
    per consumed result keeps the workers fed without letting the backlog grow.
    """
    remaining = iter(items)
    pending = deque(pool.apply_async(func, (item,)) for item in islice(remaining, max_pending))

    while pending:
        result = pending.popleft().get()
        for item in islice(remaining, 1):
            pending.append(pool.apply_async(func, (item,)))
        yield result


def process_sample_patches(
    sample_data: tuple[str, str, list, DatasetTask],
    background_color: tuple[int, int, int] = (114, 114, 114),
    mask_background: bool = True,
) -> tuple[str, list[tuple[str, np.ndarray]]]:
    """
    Process a single sample to extract patch crops.
    This function is designed to be called by worker processes.

    Args:
        sample_data: Tuple of (sample_id, filepath, patches_list, task)
        background_color: Background color for masking (segment/obb only)
        mask_background: Whether to mask the background for segment/obb tasks (default: True)

    Returns:
        Tuple of (sample_id, list_of_(patch_id, crop)). Patches with invalid geometry
        (missing bounding_box or points) are excluded entirely, not placeholder-filled -
        callers must match crops back to patches by patch_id, not by list position.
    """
    sample_id, filepath, patches_list, task = sample_data

    try:
        # Load image once
        image = cv2.imread(filepath)
        if image is None:
            return sample_id, []

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Broad by design: runs in worker processes, one bad sample must not kill the pool
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load image {filepath}: {e}")
        return sample_id, []

    crops = []

    # Process based on task type. Each patch is cropped independently so one malformed
    # patch can't discard the rest of the sample's otherwise-valid crops.
    if task in [DatasetTask.DETECTION, DatasetTask.POSE]:
        # For detection/pose: just crop to bbox
        for patch in patches_list:
            if not patch.bounding_box:
                continue
            try:
                crop = create_crop_for_detection(image, patch.bounding_box)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to crop patch {patch.id} in {filepath}: {e}")
                continue
            crops.append((patch.id, crop))

    elif task in [DatasetTask.SEGMENTATION, DatasetTask.OBB]:
        # For segmentation/obb: optionally mask background and crop
        for patch in patches_list:
            if not patch.points or len(patch.points) == 0:
                continue
            try:
                crop = create_masked_crop_for_polyline(image, patch.points[0], background_color, mask_background)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to crop patch {patch.id} in {filepath}: {e}")
                continue
            crops.append((patch.id, crop))

    return sample_id, crops


def iter_patch_crops(
    dataset: fo.Dataset,
    patches_field: str,
    dataset_task: DatasetTask,
    background_color: tuple[int, int, int] = (114, 114, 114),
    mask_background: bool = True,
    worker_func: Callable[..., tuple[str, list]] = process_sample_patches,
    desc: str = "Extracting crops",
) -> Iterator[tuple[str, list]]:
    """
    Stream per-sample worker results from a dataset with multiprocessing, one sample at a time.

    Yields incrementally instead of materializing every crop in memory at once, and bounds
    how far the workers may run ahead of the consumer, so a caller slower than the pool does
    not accumulate the whole dataset. `worker_func` runs inside the worker processes alongside
    crop extraction (default: returns the raw crops themselves; pass a different worker to also
    compute per-crop results there instead of afterward).

    Args:
        dataset: FiftyOne dataset
        patches_field: Field name containing patches
        dataset_task: Dataset task type
        background_color: RGB background color for masking (segment/obb only)
        mask_background: Whether to mask the background for segment/obb tasks (default: True)
        worker_func: Called per sample as (sample_data, background_color=..., mask_background=...)
        desc: Progress bar label, since this is the only bar the caller's loop gets

    Yields:
        Tuple of (sample_id, list_of_results) for each sample that has patches
    """
    # Prepare sample data for workers
    sample_data_list = []

    for sample in dataset.select_fields([patches_field, "filepath"]):
        patches_obj = sample[patches_field]

        if patches_obj is None:
            continue

        # Get patches based on task type
        if dataset_task in [DatasetTask.DETECTION, DatasetTask.POSE]:
            patches_list = patches_obj.detections if hasattr(patches_obj, "detections") else []
        elif dataset_task in [DatasetTask.SEGMENTATION, DatasetTask.OBB]:
            patches_list = patches_obj.polylines if hasattr(patches_obj, "polylines") else []
        else:
            patches_list = []

        if not patches_list:
            continue

        sample_data_list.append((sample.id, sample.filepath, patches_list, dataset_task))

    if not sample_data_list:
        logger.warning("No patches found in dataset")
        return

    process_func = partial(
        worker_func,
        background_color=background_color,
        mask_background=mask_background,
    )

    workers = max(1, cpu_count() - 1)

    with Pool(processes=workers, initializer=_limit_worker_cv2_threads) as pool:
        for sample_id, results in tqdm(
            _imap_bounded(pool, process_func, sample_data_list, workers * PENDING_SAMPLES_PER_WORKER),
            total=len(sample_data_list),
            desc=desc,
        ):
            if results:
                yield sample_id, results
