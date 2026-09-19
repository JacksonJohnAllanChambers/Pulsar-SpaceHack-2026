import time
from pathlib import Path

if __package__:
    from .file_moves import move_image_with_xml
else:
    from file_moves import move_image_with_xml


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def process_image(image_path):
    return False


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
    processing_dir = Path(base_dir) / "processing"
    ship_detected_dir = Path(base_dir) / "shipDetected"
    no_ship_detected_dir = Path(base_dir) / "noShipDetected"
    deleted_dir = Path(base_dir) / "deleted"
    processing_dir.mkdir(parents=True, exist_ok=True)
    ship_detected_dir.mkdir(parents=True, exist_ok=True)
    no_ship_detected_dir.mkdir(parents=True, exist_ok=True)
    deleted_dir.mkdir(parents=True, exist_ok=True)

    while True:
        move_expired_images(
            no_ship_detected_dir,
            deleted_dir,
            no_ship_ttl_seconds,
        )
        for image_path in processing_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                destination = ship_detected_dir if process_image(image_path) else no_ship_detected_dir
                destination_path = destination / image_path.name
                move_image_with_xml(image_path, destination)
                if destination == no_ship_detected_dir:
                    destination_path.touch()
        time.sleep(poll_interval)