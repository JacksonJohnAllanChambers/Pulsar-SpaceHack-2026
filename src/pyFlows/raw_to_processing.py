import time
from pathlib import Path

if __package__:
    from .file_moves import move_image_with_xml
else:
    from file_moves import move_image_with_xml


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def run(base_dir, poll_interval=1):
    raw_images_dir = Path(base_dir) / "rawImages"
    processing_dir = Path(base_dir) / "processing"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    processing_dir.mkdir(parents=True, exist_ok=True)

    while True:
        for image_path in raw_images_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                move_image_with_xml(image_path, processing_dir)
        time.sleep(poll_interval)