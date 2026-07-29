"""Convert YOLO annotations to FiftyOne labels."""

import fiftyone as fo
from shapely.errors import ShapelyError
from shapely.geometry import Polygon

from yolo_scout.core.enums import DatasetTask
from yolo_scout.utils.path_utils import get_image_name


def yolo_to_fiftyone(
    annotations: list[dict] | None,
    task: DatasetTask,
    class_names: list[str],
    image_width: int,
    image_height: int,
    split: str,
    image_path: str,
    label_path: str,
) -> fo.Label | None:
    """
    Convert YOLO annotations to FiftyOne labels.

    Args:
        annotations: List of parsed YOLO annotation dictionaries
        task: Dataset task type
        class_names: List of class names
        image_width: Image width in pixels
        image_height: Image height in pixels
        split: Dataset split name
        image_path: Path to image file
        label_path: Path to label file

    Returns:
        FiftyOne label object or None
    """
    if not annotations:
        return None

    if task == DatasetTask.DETECTION:
        detections = [_create_detection(anno, class_names, image_width, image_height, split) for anno in annotations]
        detections = [d for d in detections if d]

        # Add paths to each detection
        for det in detections:
            det["image_path"] = image_path
            det["label_path"] = label_path
            det["image_name"] = get_image_name(image_path)

        return fo.Detections(detections=detections) if detections else None

    elif task == DatasetTask.POSE:
        keypoints = [_create_keypoint(anno, class_names, image_width, image_height, split) for anno in annotations]
        keypoints = [k for k in keypoints if k]

        # Add paths to each keypoint
        for kp in keypoints:
            kp["image_path"] = image_path
            kp["label_path"] = label_path
            kp["image_name"] = get_image_name(image_path)

        return fo.Keypoints(keypoints=keypoints) if keypoints else None

    elif task == DatasetTask.SEGMENTATION:
        polygons = [_create_polygon(anno, class_names, image_width, image_height, split) for anno in annotations]
        polygons = [p for p in polygons if p]

        # Add paths to each polygon
        for poly in polygons:
            poly["image_path"] = image_path
            poly["label_path"] = label_path
            poly["image_name"] = get_image_name(image_path)

        return fo.Polylines(polylines=polygons) if polygons else None

    elif task == DatasetTask.OBB:
        obbs = [_create_obb(anno, class_names, image_width, image_height, split) for anno in annotations]
        obbs = [o for o in obbs if o]

        # Add paths to each obb
        for obb in obbs:
            obb["image_path"] = image_path
            obb["label_path"] = label_path
            obb["image_name"] = get_image_name(image_path)

        return fo.Polylines(polylines=obbs) if obbs else None

    return None


def _create_detection(
    anno: dict,
    class_names: list[str],
    image_width: int,
    image_height: int,
    split: str,
) -> fo.Detection | None:
    """Create a FiftyOne Detection from parsed annotation."""
    x_center = anno["x_center"]
    y_center = anno["y_center"]
    width = anno["width"]
    height = anno["height"]
    class_id = anno["class_id"]

    # Handle zero dimensions
    if width == 0:
        width = 1 / image_width
    if height == 0:
        height = 1 / image_height

    # Convert YOLO format (x_center, y_center, width, height) to FiftyOne format
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2

    # Clamp to valid bounds
    x_top_left = max(0.0, min(1.0, x_top_left))
    y_top_left = max(0.0, min(1.0, y_top_left))
    width = max(0.0, min(1.0 - x_top_left, width))
    height = max(0.0, min(1.0 - y_top_left, height))

    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

    detection = fo.Detection(
        label=label,
        bounding_box=[x_top_left, y_top_left, width, height],
        tags=[split],
    )

    detection["area"] = int(width * image_width * height * image_height)
    detection["width"] = int(width * image_width)
    detection["height"] = int(height * image_height)
    detection["aspect_ratio"] = round(width / height, 2)

    return detection


def _create_keypoint(
    anno: dict,
    class_names: list[str],
    image_width: int,
    image_height: int,
    split: str,
) -> fo.Keypoint | None:
    """Create a FiftyOne Keypoint from parsed annotation."""
    x_center = anno["x_center"]
    y_center = anno["y_center"]
    width = anno["width"]
    height = anno["height"]
    class_id = anno["class_id"]

    # Handle zero dimensions
    if width == 0:
        width = 1 / image_width
    if height == 0:
        height = 1 / image_height

    # Convert bbox to FiftyOne format
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2

    # Clamp to valid bounds
    x_top_left = max(0.0, min(1.0, x_top_left))
    y_top_left = max(0.0, min(1.0, y_top_left))
    width = max(0.0, min(1.0 - x_top_left, width))
    height = max(0.0, min(1.0 - y_top_left, height))

    # Parse keypoints
    points = []
    num_keypoints = 0

    for kp in anno.get("keypoints", []):
        kp_x = max(0.0, min(1.0, kp["x"]))
        kp_y = max(0.0, min(1.0, kp["y"]))

        if kp_x != 0 and kp_y != 0:
            points.append([kp_x, kp_y])
            num_keypoints += 1
        else:
            # Hidden keypoint
            points.append([-1, -1])

    if not points:
        return None

    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

    keypoint = fo.Keypoint(
        label=label,
        points=points,
        bounding_box=[x_top_left, y_top_left, width, height],
        tags=[split],
    )

    keypoint["area"] = int(width * image_width * height * image_height)
    keypoint["num_keypoints"] = num_keypoints

    return keypoint


def _create_polygon(
    anno: dict,
    class_names: list[str],
    image_width: int,
    image_height: int,
    split: str,
) -> fo.Polyline | None:
    """Create a FiftyOne Polyline (polygon) from parsed annotation."""
    points_data = anno.get("points", [])
    if len(points_data) < 3:
        return None

    class_id = anno["class_id"]

    # Convert to FiftyOne format
    points = []
    points_pixels = []
    min_x, min_y = 1.0, 1.0
    max_x, max_y = 0.0, 0.0

    for pt in points_data:
        x = max(0.0, min(1.0, pt["x"]))
        y = max(0.0, min(1.0, pt["y"]))
        points.append([x, y])
        points_pixels.append([x * image_width, y * image_height])

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Close the polygon if not already closed
    if points[0] != points[-1]:
        points.append(points[0])

    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

    polygon = fo.Polyline(
        label=label,
        points=[points],
        closed=True,
        filled=True,
        tags=[split],
    )

    # Calculate area using Shapely
    try:
        poly_points = points_pixels[:-1] if points_pixels[0] == points_pixels[-1] else points_pixels
        polygon["area"] = int(Polygon(poly_points).area)

    except (ValueError, ShapelyError):
        polygon["area"] = 0

    polygon["num_keypoints"] = len(points)
    polygon["width"] = int((max_x - min_x) * image_width)
    polygon["height"] = int((max_y - min_y) * image_height)

    return polygon


def _create_obb(
    anno: dict,
    class_names: list[str],
    image_width: int,
    image_height: int,
    split: str,
) -> fo.Polyline | None:
    """Create a FiftyOne Polyline (OBB) from parsed annotation."""
    points_data = anno.get("points", [])
    if len(points_data) != 4:
        return None

    class_id = anno["class_id"]

    # Convert to FiftyOne format
    points = []
    points_pixels = []

    for pt in points_data:
        x = max(0.0, min(1.0, pt["x"]))
        y = max(0.0, min(1.0, pt["y"]))
        points.append([x, y])
        points_pixels.append([x * image_width, y * image_height])

    # Close the polygon
    points.append(points[0])

    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

    obb = fo.Polyline(
        label=label,
        points=[points],
        closed=True,
        filled=False,
        tags=[split],
    )

    # Calculate area using Shapely
    try:
        poly_points = points_pixels[:-1] if points_pixels[0] == points_pixels[-1] else points_pixels
        obb["area"] = int(Polygon(poly_points).area)

    except (ValueError, ShapelyError):
        obb["area"] = 0

    # Calculate width and height from first two points
    obb["width"] = int(
        ((points_pixels[1][0] - points_pixels[0][0]) ** 2 + (points_pixels[1][1] - points_pixels[0][1]) ** 2) ** 0.5
    )
    obb["height"] = int(
        ((points_pixels[2][0] - points_pixels[1][0]) ** 2 + (points_pixels[2][1] - points_pixels[1][1]) ** 2) ** 0.5
    )

    return obb


def create_detection_from_keypoint(
    keypoint: fo.Keypoint,
    image_width: int,
    image_height: int,
    image_path: str,
    label_path: str,
) -> fo.Detection:
    """
    Create a detection from a keypoint (for pose estimation).

    Args:
        keypoint: FiftyOne Keypoint object
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        image_path: Path to image file
        label_path: Path to label file

    Returns:
        FiftyOne Detection object
    """
    bbox = keypoint.get_field("bounding_box")

    detection = fo.Detection(
        label=keypoint.label,
        bounding_box=bbox,
        tags=keypoint.tags,
    )

    # Copy over metadata using proper field access
    detection["area"] = keypoint.get_field("area")
    detection["num_keypoints"] = keypoint.get_field("num_keypoints")
    detection["width"] = int(bbox[2] * image_width)
    detection["height"] = int(bbox[3] * image_height)
    detection["image_path"] = image_path
    detection["label_path"] = label_path
    detection["image_name"] = get_image_name(image_path)

    return detection
