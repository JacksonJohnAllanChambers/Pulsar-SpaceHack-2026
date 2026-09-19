import random
import time
from pathlib import Path

if __package__:
    from .file_moves import move_image_with_xml
else:
    from file_moves import move_image_with_xml


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def should_use_non_priority_queue():
    return random.random() >= 0.25


def run(base_dir, poll_interval=1):
    ship_detected_dir = Path(base_dir) / "shipDetected"
    priority_queue_dir = base_dir / "downlink" / "queues" / "priority"
    non_priority_queue_dir = base_dir / "downlink" / "queues" / "nonPriority"
    ship_detected_dir.mkdir(parents=True, exist_ok=True)
    priority_queue_dir.mkdir(parents=True, exist_ok=True)
    non_priority_queue_dir.mkdir(parents=True, exist_ok=True)

    while True:
        for image_path in ship_detected_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                queue_dir = non_priority_queue_dir if should_use_non_priority_queue() else priority_queue_dir
                move_image_with_xml(image_path, queue_dir)
        time.sleep(poll_interval)