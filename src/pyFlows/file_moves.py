import shutil
from pathlib import Path

from applet.core.crypto import encrypt_file


def move_image_with_xml(image_path, destination_dir):
    image_path = Path(image_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if "downlink" in destination_dir.parts:
        encrypt_file(image_path, destination_dir / f"{image_path.name}.enc")
    else:
        shutil.move(image_path, destination_dir / image_path.name)

    xml_path = image_path.with_suffix(".xml")
    if xml_path.is_file():
        if "downlink" in destination_dir.parts:
            encrypt_file(xml_path, destination_dir / f"{xml_path.name}.enc")
        else:
            shutil.move(xml_path, destination_dir / xml_path.name)