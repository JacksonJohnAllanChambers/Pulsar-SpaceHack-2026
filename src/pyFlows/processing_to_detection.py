import time
from pathlib import Path
from typing import Tuple

if __package__:
    from .file_moves import move_image_with_xml
else:
    from file_moves import move_image_with_xml


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def process_image(image_path) -> Tuple[bool, bool]:
    """Return whether a ship was detected and whether it matched an AIS record."""
    return False, False


def destination_for_result(ship_detected, ais_matched, priority_dir, less_priority_dir, no_ship_dir):
    if not ship_detected:
        return no_ship_dir
    return less_priority_dir if ais_matched else priority_dir


def move_expired_images(no_ship_detected_dir, deleted_dir, ttl_seconds):
    now = time.time()
    for image_path in no_ship_detected_dir.iterdir():
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
            and now - image_path.stat().st_mtime >= ttl_seconds
        ):
            move_image_with_xml(image_path, deleted_dir)


def run(base_dir, poll_interval=1, no_ship_ttl_seconds=60):
    base_dir = Path(base_dir)
    processing_dir = base_dir / "processing"
    priority_queue_dir = base_dir / "downlink" / "queues" / "priority"
    less_priority_queue_dir = base_dir / "downlink" / "queues" / "nonPriority"
    no_ship_detected_dir = base_dir / "noShipDetected"
    deleted_dir = base_dir / "deleted"
    processing_dir.mkdir(parents=True, exist_ok=True)
    priority_queue_dir.mkdir(parents=True, exist_ok=True)
    less_priority_queue_dir.mkdir(parents=True, exist_ok=True)
    no_ship_detected_dir.mkdir(parents=True, exist_ok=True)
    deleted_dir.mkdir(parents=True, exist_ok=True)

    while True:
        move_expired_images(no_ship_detected_dir, deleted_dir, no_ship_ttl_seconds)
        for image_path in processing_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                ship_detected, ais_matched = process_image(image_path)
                destination = destination_for_result(
                    ship_detected,
                    ais_matched,
                    priority_queue_dir,
                    less_priority_queue_dir,
                    no_ship_detected_dir,
                )
                move_image_with_xml(image_path, destination)
        time.sleep(poll_interval)