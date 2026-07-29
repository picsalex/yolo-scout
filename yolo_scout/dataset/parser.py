"""Parse YOLO annotation files."""

import os

from yolo_scout.core.enums import DatasetTask
from yolo_scout.utils.logger import logger


def parse_yolo_annotation(
    label_path: str,
    task: DatasetTask,
) -> list[dict] | None:
    """
    Parse a YOLO annotation file into a list of annotation dictionaries.

    Args:
        label_path: Path to the YOLO label file
        task: Dataset task type

    Returns:
        List of annotation dictionaries, or None if file doesn't exist
    """
    if not os.path.exists(label_path):
        return None

    try:
        with open(label_path, "r") as f:
            annotations = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if task == DatasetTask.DETECTION:
                    anno = _parse_detection_line(line)
                elif task == DatasetTask.POSE:
                    anno = _parse_pose_line(line)
                elif task == DatasetTask.SEGMENTATION:
                    anno = _parse_segmentation_line(line)
                elif task == DatasetTask.OBB:
                    anno = _parse_obb_line(line)
                else:
                    continue

                if anno:
                    annotations.append(anno)

            return annotations if annotations else None

    except (OSError, ValueError, IndexError) as e:
        logger.warning(f"Failed to parse annotations from {label_path}: {e}")
        return None


def _parse_detection_line(line: str) -> dict | None:
    """
    Parse a detection annotation line.
    Format: class_id x_center y_center width height
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    return {
        "class_id": int(parts[0]),
        "x_center": float(parts[1]),
        "y_center": float(parts[2]),
        "width": float(parts[3]),
        "height": float(parts[4]),
    }


def _parse_pose_line(line: str) -> dict | None:
    """
    Parse a pose annotation line.
    Format: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    # Parse bbox
    anno = {
        "class_id": int(parts[0]),
        "x_center": float(parts[1]),
        "y_center": float(parts[2]),
        "width": float(parts[3]),
        "height": float(parts[4]),
        "keypoints": [],
    }

    # Parse keypoints (groups of 3: x, y, visibility)
    keypoint_data = parts[5:]
    for i in range(0, len(keypoint_data), 3):
        if i + 2 < len(keypoint_data):
            kp_x = float(keypoint_data[i])
            kp_y = float(keypoint_data[i + 1])
            kp_v = float(keypoint_data[i + 2])
            anno["keypoints"].append({"x": kp_x, "y": kp_y, "visibility": kp_v})

    return anno


def _parse_segmentation_line(line: str) -> dict | None:
    """
    Parse a segmentation annotation line.
    Format: class_id x1 y1 x2 y2 x3 y3 ...
    """
    parts = line.split()
    if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
        return None

    points = []
    coords = parts[1:]

    if len(coords) % 2 != 0:
        return None

    for i in range(0, len(coords), 2):
        x = float(coords[i])
        y = float(coords[i + 1])
        points.append({"x": x, "y": y})

    return {
        "class_id": int(parts[0]),
        "points": points,
    }


def _parse_obb_line(line: str) -> dict | None:
    """
    Parse an OBB (Oriented Bounding Box) annotation line.
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    """
    parts = line.split()
    if len(parts) != 9:  # class_id + 8 coordinates (4 points)
        return None

    points = []
    for i in range(1, 9, 2):
        x = float(parts[i])
        y = float(parts[i + 1])
        points.append({"x": x, "y": y})

    return {
        "class_id": int(parts[0]),
        "points": points,
    }
