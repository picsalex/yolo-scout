"""Generate thumbnails for FiftyOne visualization."""

import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.image as foui

from yolo_scout.core.constants import CORRUPTED_TAG, THUMBNAIL_PATH_KEY
from yolo_scout.utils.logger import logger


def generate_thumbnails(dataset: fo.Dataset, thumbnail_dir_path: str, thumbnail_width: int) -> None:
    """
    Generate optimized thumbnails using transform_images.

    Args:
        dataset: The FiftyOne dataset to process
        thumbnail_dir_path: Directory to save thumbnails
        thumbnail_width: Width of the generated thumbnails in pixels
    """
    logger.info(f"Output directory: {thumbnail_dir_path}")

    # Create thumbnail directory if it doesn't exist
    try:
        os.makedirs(thumbnail_dir_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create thumbnail directory: {e}")
        raise

    try:
        filepaths = [p for p in dataset.values("filepath") if p]
        if not filepaths:
            logger.error(
                "Thumbnail generation failed: the dataset appears to be empty. "
                "If this dataset was loaded from cache, try re-running with --reload."
            )
            return

        common_base = os.path.commonpath([os.path.dirname(p) for p in filepaths])
        clean_view = dataset.match_tags(CORRUPTED_TAG, bool=False)

        # Resize along the largest dimension; set -1 on the smallest so FiftyOne
        # computes it automatically (handles both landscape and portrait images).
        for size, orientation_filter in [
            (
                (thumbnail_width, -1),
                fo.ViewField("metadata.width") >= fo.ViewField("metadata.height"),
            ),
            (
                (-1, thumbnail_width),
                fo.ViewField("metadata.height") > fo.ViewField("metadata.width"),
            ),
        ]:
            oriented_view = clean_view.match(orientation_filter)
            if len(oriented_view) > 0:
                foui.transform_images(
                    oriented_view,
                    size=size,
                    output_dir=thumbnail_dir_path,
                    rel_dir=common_base,
                    output_field=THUMBNAIL_PATH_KEY,
                )

        dataset.info["thumbnail_width"] = thumbnail_width
        dataset.save()
        logger.info("Thumbnails generated successfully")

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        raise


def delete_thumbnails(dataset_name: str, thumbnail_dir: str) -> None:
    """Delete thumbnails folder associated with a dataset.

    Args:
        dataset_name: Name of the dataset whose thumbnails should be deleted
        thumbnail_dir: Base directory where thumbnails are stored (default: "thumbnails")
    """
    # Construct the full path to the dataset's thumbnail directory
    thumbnail_path = Path(thumbnail_dir) / dataset_name

    if thumbnail_path.exists() and thumbnail_path.is_dir():
        try:
            shutil.rmtree(thumbnail_path)
            logger.info(f"Deleted thumbnails directory: {thumbnail_path}")
        except OSError as e:
            logger.warning(f"Failed to delete thumbnails directory {thumbnail_path}: {e}")
