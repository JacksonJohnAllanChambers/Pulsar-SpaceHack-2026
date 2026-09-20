from xml.etree import ElementTree

from applet.core.crypto import decrypt_file
from ground.fleets import FLEETS, dispatch_sent_alerts, nearest_fleet, parse_alert_filename


def test_fleet_seeds_cover_every_defined_real_data_aoi():
    assert len(FLEETS) == 21
    assert len({fleet["id"] for fleet in FLEETS}) == len(FLEETS)


def test_nearest_fleet_assigns_long_beach_contact():
    fleet = nearest_fleet(33.68, -118.17)
    assert fleet["id"] == "longbeach"
    assert fleet["distance_nm"] < 5


def test_dispatch_sent_alerts_parses_and_persists_dark_vessel(tmp_path):
    filename = "alert-v1__S2_LONGBEACH_T001__S2_LONGBEACH__lat-33.680000__lon--118.170000__DARK_VESSEL.jpg"
    (tmp_path / "sent").mkdir()
    (tmp_path / "sent" / filename).write_bytes(b"jpeg")
    (tmp_path / "sent" / f"{filename}.queue-origin").write_text("priority", encoding="utf-8")
    alert_path = tmp_path / "alerts.json"
    output_dir = tmp_path / "fleet_alerts"

    assert parse_alert_filename(tmp_path / "sent" / filename)["detection_id"] == "S2_LONGBEACH_T001"
    alerts = dispatch_sent_alerts(tmp_path / "sent", alert_path, output_dir)

    assert len(alerts) == 1
    assert alerts[0]["fleet"]["id"] == "longbeach"
    ping_dir = output_dir / "longbeach" / "ping_S2_LONGBEACH_T001"
    assert not (ping_dir / "image.jpg").exists()
    assert decrypt_file(ping_dir / "image.jpg.enc") == b"jpeg"
    info = ElementTree.fromstring(decrypt_file(ping_dir / "info.xml.enc"))
    assert info.findtext("fleet/name") == "Long Beach Response"
    assert info.findtext("scene_id") == "S2_LONGBEACH"
    assert dispatch_sent_alerts(tmp_path / "sent", alert_path, output_dir) == []


def test_dispatch_sent_alerts_ignores_non_priority_dark_vessel(tmp_path):
    filename = "alert-v1__S2_LONGBEACH_T001__S2_LONGBEACH__lat-33.680000__lon--118.170000__DARK_VESSEL.jpg"
    sent_dir = tmp_path / "sent"
    sent_dir.mkdir()
    (sent_dir / filename).write_bytes(b"jpeg")
    (sent_dir / f"{filename}.queue-origin").write_text("nonPriority", encoding="utf-8")

    assert dispatch_sent_alerts(sent_dir, tmp_path / "alerts.json", tmp_path / "fleet_alerts") == []