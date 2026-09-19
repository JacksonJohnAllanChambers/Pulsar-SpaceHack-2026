import time
from pathlib import Path

if __package__:
    from .file_moves import move_image_with_xml
else:
    from file_moves import move_image_with_xml


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRANSFER_RATE_BYTES_PER_SECOND = 50 * 1024 * 1024
PRIORITY_IMAGES_PER_LESS_PRIORITY_IMAGE = 3


def downlink_image(image_path, sent_dir, rate_bytes_per_second=TRANSFER_RATE_BYTES_PER_SECOND):
    transfer_seconds = image_path.stat().st_size / rate_bytes_per_second
    time.sleep(transfer_seconds)
    move_image_with_xml(image_path, sent_dir)


def next_image(source_dir):
    for image_path in source_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            return image_path
    return None


def next_downlink_image(
    priority_queue_dir,
    less_priority_queue_dir,
    no_ship_detected_dir,
    consecutive_priority_images,
):
    priority_image = next_image(priority_queue_dir)
    less_priority_image = next_image(less_priority_queue_dir)

    if priority_image and (
        not less_priority_image
        or consecutive_priority_images < PRIORITY_IMAGES_PER_LESS_PRIORITY_IMAGE
    ):
        return priority_image, consecutive_priority_images + 1
    if less_priority_image:
        return less_priority_image, 0

    return next_image(no_ship_detected_dir), consecutive_priority_images


def run(base_dir, poll_interval=1):
    base_dir = Path(base_dir)
    priority_queue_dir = base_dir / "downlink" / "queues" / "priority"
    less_priority_queue_dir = base_dir / "downlink" / "queues" / "nonPriority"
    no_ship_detected_dir = base_dir / "noShipDetected"
    sent_dir = base_dir / "sent"
    priority_queue_dir.mkdir(parents=True, exist_ok=True)
    less_priority_queue_dir.mkdir(parents=True, exist_ok=True)
    no_ship_detected_dir.mkdir(parents=True, exist_ok=True)
    sent_dir.mkdir(parents=True, exist_ok=True)
    consecutive_priority_images = 0

    while True:
        image_path, consecutive_priority_images = next_downlink_image(
            priority_queue_dir,
            less_priority_queue_dir,
            no_ship_detected_dir,
            consecutive_priority_images,
        )
        if image_path is None:
            time.sleep(poll_interval)
            continue
        downlink_image(image_path, sent_dir)