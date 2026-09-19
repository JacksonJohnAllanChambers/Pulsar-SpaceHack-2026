import shutil
from pathlib import Path


def move_image_with_xml(image_path, destination_dir):
    image_path = Path(image_path)
    destination_dir = Path(destination_dir)
    shutil.move(image_path, destination_dir / image_path.name)

    xml_path = image_path.with_suffix(".xml")
    if xml_path.is_file():
        shutil.move(xml_path, destination_dir / xml_path.name)