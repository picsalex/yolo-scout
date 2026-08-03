"""Constants used throughout the application."""

import matplotlib.pyplot as plt

from yolo_scout.core.enums import DatasetTask

# Field names for different annotation types
DETECTION_FIELD = "bounding_boxes"
KEYPOINTS_FIELD = "pose_keypoints"
SEGMENTATION_FIELD = "seg_polygons"
CLASSIFICATION_FIELD = "cls_label"
OBB_FIELD = "obb_bounding_boxes"

# Brain keys for embeddings
IMAGE_EMBEDDINGS_KEY = "images_embeddings"
PATCH_EMBEDDINGS_KEY = "patches_embeddings"
THUMBNAIL_PATH_KEY = "thumbnail_path"

# Ultralytics color palette
ULTRALYTICS_COLORS = [
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
]

# Dataset split options
DATASET_SPLITS = [
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "train2017",
    "val2017",
    "test2017",
]

SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".tiff")


def get_field_name(task: DatasetTask) -> str:
    """
    Get the appropriate field name based on the dataset task.

    Args:
        task: The dataset task (classify, detect, segment, pose, obb)

    Returns:
        The corresponding field name in the FiftyOne dataset
    """
    if task == DatasetTask.CLASSIFICATION:
        return CLASSIFICATION_FIELD
    elif task == DatasetTask.DETECTION:
        return DETECTION_FIELD
    elif task == DatasetTask.SEGMENTATION:
        return SEGMENTATION_FIELD
    elif task == DatasetTask.POSE:
        return KEYPOINTS_FIELD
    elif task == DatasetTask.OBB:
        return OBB_FIELD
    else:
        raise ValueError(f"Unsupported dataset task: {task}")


def get_patches_attr(task: DatasetTask) -> str | None:
    """
    Get the attribute holding the patch list on a task's label field.

    Args:
        task: The dataset task

    Returns:
        "detections", "polylines", or None for tasks that have no patches
    """
    if task in (DatasetTask.DETECTION, DatasetTask.POSE):
        return "detections"
    if task in (DatasetTask.SEGMENTATION, DatasetTask.OBB):
        return "polylines"
    return None


def get_color_palette(labels: list[str]) -> list[dict[str, str]]:
    """
    Use the ultralytics color palette to generate a list of distinct colors.
    If more colors are needed than available in the palette, generate additional colors using a colormap.

    Args:
        labels: A list of label names.

    Returns:
        A list of dicts with 'value' and 'color' keys for each label.
    """
    num_labels = len(labels)
    palette = []

    for i in range(num_labels):
        if i < len(ULTRALYTICS_COLORS):
            color_hex = f"#{ULTRALYTICS_COLORS[i]}"
        else:
            # Generate additional colors using a colormap
            cmap = plt.get_cmap("hsv")
            color = cmap(i / num_labels)
            color_hex = f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
        palette.append({"value": labels[i], "color": color_hex})

    return palette
