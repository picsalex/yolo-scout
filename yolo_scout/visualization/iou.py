"""Compute IoU scores for overlapping annotations."""

import fiftyone as fo
from shapely.geometry import Polygon, box

from yolo_scout.core.enums import DatasetTask
from yolo_scout.utils.logger import logger


def compute_iou_scores(labels: fo.Label, dataset_task: DatasetTask) -> None:
    """
    Set the iou_score property for each object in the labels based on IoU overlap.
    For each object, stores the maximum IoU value with any other object in the image.

    Args:
        labels: The labels containing detections or polylines
        dataset_task: The dataset task (detect, segment, obb)
    """
    if labels is None:
        return

    try:
        # Extract objects based on task type
        if dataset_task == DatasetTask.DETECTION:
            objects = labels.get_field("detections")
            if not objects:
                return

            _compute_bbox_ious(objects)

        elif dataset_task == DatasetTask.SEGMENTATION or dataset_task == DatasetTask.OBB:
            objects = labels.get_field("polylines")
            if not objects:
                return

            _compute_polygon_ious(objects)

    except Exception as e:
        logger.warning(f"Failed to compute IoU scores: {e}")


def _compute_bbox_ious(objects):
    """Compute IoU scores for bounding boxes."""
    n = len(objects)
    iou_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            bbox_i = objects[i].bounding_box  # [x_top_left, y_top_left, width, height]
            bbox_j = objects[j].bounding_box

            # Create Shapely box objects (x_min, y_min, x_max, y_max)
            box_i = box(
                bbox_i[0],
                bbox_i[1],
                bbox_i[0] + bbox_i[2],
                bbox_i[1] + bbox_i[3],
            )
            box_j = box(
                bbox_j[0],
                bbox_j[1],
                bbox_j[0] + bbox_j[2],
                bbox_j[1] + bbox_j[3],
            )

            # Calculate IoU
            intersection = box_i.intersection(box_j).area
            union = box_i.union(box_j).area
            iou = intersection / union if union > 0 else 0.0

            iou_matrix[i][j] = iou
            iou_matrix[j][i] = iou

    # Set iou_score property for each object
    for i in range(n):
        max_iou = 0.0
        for j in range(n):
            if i != j:
                max_iou = max(max_iou, iou_matrix[i][j])
        objects[i]["iou_score"] = round(max_iou, 3)


def _compute_polygon_ious(objects):
    """Compute IoU scores for polygons."""
    n = len(objects)
    iou_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Get polygon points [[[x1, y1], [x2, y2], ...]]
            points_i = objects[i].points[0] if objects[i].points else []
            points_j = objects[j].points[0] if objects[j].points else []

            if len(points_i) < 3 or len(points_j) < 3:
                continue

            try:
                # Create Shapely polygons
                poly_i = Polygon(points_i)
                poly_j = Polygon(points_j)

                # Ensure polygons are valid
                if not poly_i.is_valid:
                    poly_i = poly_i.buffer(0)
                if not poly_j.is_valid:
                    poly_j = poly_j.buffer(0)

                # Calculate IoU
                intersection = poly_i.intersection(poly_j).area
                union = poly_i.union(poly_j).area
                iou = intersection / union if union > 0 else 0.0

                iou_matrix[i][j] = iou
                iou_matrix[j][i] = iou

            except Exception:
                # Skip invalid polygons
                continue

    # Set iou_score property for each object
    for i in range(n):
        max_iou = 0.0
        for j in range(n):
            if i != j:
                max_iou = max(max_iou, iou_matrix[i][j])
        objects[i]["iou_score"] = round(max_iou, 3)
