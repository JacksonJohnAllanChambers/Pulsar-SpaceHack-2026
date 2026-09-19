"""Crop classified vessel detections into the downlink priority queues."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


MIN_PADDING_PX = 64
MAX_PADDING_PX = 160
PADDING_HULL_LENGTHS = 1.5


def padding_pixels(
    detection: Dict[str, Any], gsd_m: float, minimum: int = MIN_PADDING_PX, maximum: int = MAX_PADDING_PX
) -> int:
    """Return bounded context padding sized to the detected vessel's hull."""
    hull_length_m = float(detection.get("hull_length_m") or 0.0)
    scaled_padding = int(np.ceil(PADDING_HULL_LENGTHS * hull_length_m / gsd_m)) if gsd_m > 0 else minimum
    return max(minimum, min(maximum, scaled_padding))


def crop_detection(image: np.ndarray, detection: Dict[str, Any], gsd_m: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a padded target window, clipping crop bounds to the source image."""
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")

    try:
        center_x, center_y = detection["apex_px"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("detection requires a two-value apex_px") from error

    height, width = image.shape[:2]
    padding = padding_pixels(detection, gsd_m)
    left = max(0, int(np.floor(float(center_x) - padding)))
    top = max(0, int(np.floor(float(center_y) - padding)))
    right = min(width, int(np.ceil(float(center_x) + padding + 1)))
    bottom = min(height, int(np.ceil(float(center_y) + padding + 1)))
    if left >= right or top >= bottom:
        raise ValueError("detection apex_px is outside the source image")
    return image[top:bottom, left:right].copy(), (left, top, right, bottom)


def destination_for_detection(detection: Dict[str, Any], priority_dir: Path, less_priority_dir: Path) -> Path:
    """Route unmatched vessels to priority; AIS-correlated vessels to the lower-priority queue."""
    return less_priority_dir if detection.get("matched_vessel") is not None else priority_dir


def output_filename(detection: Dict[str, Any], index: int) -> str:
    """Use a self-contained location contract for deliverable dark-vessel alerts."""
    target_id = str(detection.get("detection_id") or f"vessel_{index:03d}")
    if detection.get("classification") != "DARK_VESSEL":
        return f"{target_id}.jpg"
    coordinates = detection.get("world_coordinates") or {}
    try:
        latitude = float(coordinates["latitude"])
        longitude = float(coordinates["longitude"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dark-vessel detection requires world_coordinates") from error
    scene_id = str(detection.get("scene_id") or "scene")
    return f"alert-v1__{target_id}__{scene_id}__lat-{latitude:.6f}__lon-{longitude:.6f}__DARK_VESSEL.jpg"


def _display_image(crop: np.ndarray) -> np.ndarray:
    """Convert single- or multi-band crops to a JPEG-compatible BGR image."""
    if crop.ndim == 2:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if crop.ndim != 3:
        raise ValueError("crop must be a grayscale or multi-band image")
    if crop.shape[2] == 1:
        return cv2.cvtColor(crop[:, :, 0], cv2.COLOR_GRAY2BGR)
    if crop.shape[2] >= 3:
        return crop[:, :, :3]
    raise ValueError("crop must contain at least one band")


def write_vessel_crops(
    image: np.ndarray,
    detections: Iterable[Dict[str, Any]],
    priority_dir: Path,
    less_priority_dir: Path,
    gsd_m: float,
) -> List[Path]:
    """Write one JPEG crop per detection and return paths in detection order."""
    priority_dir = Path(priority_dir)
    less_priority_dir = Path(less_priority_dir)
    priority_dir.mkdir(parents=True, exist_ok=True)
    less_priority_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for index, detection in enumerate(detections, start=1):
        crop, _ = crop_detection(image, detection, gsd_m)
        destination = destination_for_detection(detection, priority_dir, less_priority_dir)
        output_path = destination / output_filename(detection, index)
        if not cv2.imwrite(str(output_path), _display_image(crop)):
            raise OSError(f"failed to write crop: {output_path}")
        outputs.append(output_path)
    return outputs


def route_classified_targets(
    scenes: Iterable[Dict[str, Any]],
    detections: Iterable[Dict[str, Any]],
    queue_root: Path,
) -> List[Path]:
    """Write classified target crops using their source scene arrays."""
    scenes_by_id = {scene["id"]: scene for scene in scenes}
    targets_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for detection in detections:
        targets_by_scene.setdefault(str(detection["scene_id"]), []).append(detection)

    priority_dir = Path(queue_root) / "priority"
    less_priority_dir = Path(queue_root) / "nonPriority"
    outputs = []
    for scene_id, scene_detections in targets_by_scene.items():
        scene = scenes_by_id.get(scene_id)
        if scene is None or "array" not in scene:
            raise ValueError(f"source scene is unavailable for {scene_id}")
        outputs.extend(
            write_vessel_crops(
                scene["array"], scene_detections, priority_dir, less_priority_dir, float(scene["gsd_m"])
            )
        )
    return outputs