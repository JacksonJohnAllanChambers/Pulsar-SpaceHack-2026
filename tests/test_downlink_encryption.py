from applet.core.crypto import decrypt_file
from src.pyFlows.file_moves import move_image_with_xml
from src.pyFlows.timed_downlink import downlink_image


def test_downlink_queue_and_sent_files_are_encrypted(tmp_path):
    source_dir = tmp_path / "processing"
    queue_dir = tmp_path / "downlink" / "queues" / "priority"
    sent_dir = tmp_path / "sent"
    source_dir.mkdir()
    sent_dir.mkdir()
    image_path = source_dir / "contact.jpg"
    image_path.write_bytes(b"jpeg")
    image_path.with_suffix(".xml").write_text("<contact />", encoding="utf-8")

    move_image_with_xml(image_path, queue_dir)

    queued_image = queue_dir / "contact.jpg.enc"
    queued_xml = queue_dir / "contact.xml.enc"
    assert not image_path.exists()
    assert not image_path.with_suffix(".xml").exists()
    assert decrypt_file(queued_image) == b"jpeg"
    assert decrypt_file(queued_xml) == b"<contact />"

    downlink_image(queued_image, sent_dir, rate_bytes_per_second=1024**3)

    assert not queued_image.exists()
    assert not queued_xml.exists()
    assert decrypt_file(sent_dir / "contact.jpg.enc") == b"jpeg"
    assert decrypt_file(sent_dir / "contact.xml.enc") == b"<contact />"