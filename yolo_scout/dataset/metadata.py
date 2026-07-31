"""Extract metadata from images."""

import os

import fiftyone as fo
from PIL import Image

from yolo_scout.utils.logger import logger


def extract_image_metadata(filepath: str) -> fo.ImageMetadata:
    """
    Extract all metadata for an image.

    Args:
        filepath: Path to the image file

    Returns:
        FiftyOne ImageMetadata object
    """
    try:
        width, height = get_dimensions(filepath)
        channels = get_channel_count(filepath)
        size_bytes = get_size_bytes(filepath)
        mime_type = get_mime_type(filepath)

        return fo.ImageMetadata(
            width=width,
            height=height,
            num_channels=channels,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )
    except Exception as e:
        logger.error(f"Failed to extract metadata for {filepath}: {e}")
        raise


def get_dimensions(filepath: str) -> tuple[int, int]:
    """Get image dimensions (width, height)."""
    with Image.open(filepath) as img:
        return img.size


def get_channel_count(filepath: str) -> int:
    """Get number of channels in an image."""
    with Image.open(filepath) as img:
        return len(img.getbands())


def get_size_bytes(filepath: str) -> int:
    """Get image file size in bytes."""
    return os.path.getsize(filepath)


def get_mime_type(filepath: str) -> str:
    """Get image MIME type."""
    with Image.open(filepath) as img:
        return img.get_format_mimetype()
