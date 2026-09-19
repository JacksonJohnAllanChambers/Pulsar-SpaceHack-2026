import importlib.util
from pathlib import Path

import cv2
import numpy as np


PROCESS_PATH = Path(__file__).parents[1] / "src" / "pyFlows" / "process.py"
SPEC = importlib.util.spec_from_file_location("process", PROCESS_PATH)
process = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(process)


def test_crop_detection_applies_bounded_hull_padding():
    image = np.zeros((400, 400, 4), dtype=np.uint8)
    detection = {"apex_px": [200, 200], "hull_length_m": 800}

    crop, bounds = process.crop_detection(image, detection, gsd_m=10.0)

    assert bounds == (80, 80, 321, 321)
    assert crop.shape == (241, 241, 4)


def test_write_vessel_crops_routes_by_ais_match(tmp_path):
    image = np.full((300, 300, 4), 120, dtype=np.uint8)
    priority_dir = tmp_path / "priority"
    less_priority_dir = tmp_path / "nonPriority"
    detections = [
        {"detection_id": "DARK", "apex_px": [12, 15], "hull_length_m": 30, "matched_vessel": None},
        {"detection_id": "KNOWN", "apex_px": [280, 280], "hull_length_m": 30, "matched_vessel": "123456789"},
    ]

    outputs = process.write_vessel_crops(image, detections, priority_dir, less_priority_dir, gsd_m=10.0)

    assert outputs == [priority_dir / "DARK.jpg", less_priority_dir / "KNOWN.jpg"]
    assert all(path.is_file() for path in outputs)
    assert all(cv2.imread(str(path)) is not None for path in outputs)


def test_route_classified_targets_uses_source_scenes(tmp_path):
    scene = {"id": "SCENE_01", "array": np.full((200, 200, 4), 80, dtype=np.uint8), "gsd_m": 10.0}
    detection = {"detection_id": "SCENE_01_T001", "scene_id": "SCENE_01", "apex_px": [100, 100],
                 "hull_length_m": 30, "matched_vessel": None}

    outputs = process.route_classified_targets([scene], [detection], tmp_path / "queues")

    assert outputs == [tmp_path / "queues" / "priority" / "SCENE_01_T001.jpg"]