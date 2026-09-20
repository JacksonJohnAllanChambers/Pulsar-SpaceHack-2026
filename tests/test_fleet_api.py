import pytest

from applet.core.crypto import encrypt_bytes

# ground/ is the ground station, not flight code, so FastAPI is a dev dependency and is absent
# from the judges' container. Without this guard the module fails to import, and pytest aborts
# the WHOLE run on a collection error -- so one ground-side test would take every flight test
# with it in exactly the environment we most need them to run.
pytest.importorskip("fastapi", reason="ground console dependency; not installed in the flight image")

from ground import server  # noqa: E402


def reset_transfer_tracker(monkeypatch, root):
    monkeypatch.setattr(server, "TRANSFER_ROOT", str(root))
    monkeypatch.setattr(server, "_transfer_previous", {})
    monkeypatch.setattr(server, "_transfer_events", server.deque(maxlen=server.TRANSFER_EVENT_LIMIT))


def test_fleet_alert_api_returns_seeded_fleets(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "FLEET_ALERT_PATH", str(tmp_path / "alerts.json"))
    response = server.fleet_alerts()

    assert len(response["fleets"]) == 21
    assert response["alerts"] == []


def test_dispatch_api_reads_sent_folder(monkeypatch, tmp_path):
    sent_dir = tmp_path / "sent"
    sent_dir.mkdir()
    filename = "alert-v1__S2_LONGBEACH_T001__S2_LONGBEACH__lat-33.680000__lon--118.170000__DARK_VESSEL.jpg"
    (sent_dir / filename).write_bytes(b"jpeg")
    (sent_dir / f"{filename}.queue-origin").write_text("priority", encoding="utf-8")
    monkeypatch.setattr(server, "SENT_DIR", str(sent_dir))
    monkeypatch.setattr(server, "FLEET_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setattr(server, "FLEET_OUTPUT_DIR", str(tmp_path / "fleet_alerts"))

    response = server.dispatch_fleet_alerts()

    assert response["dispatched"][0]["fleet"]["id"] == "longbeach"
    assert len(response["fleets"]) == 21
    assert (tmp_path / "fleet_alerts" / "longbeach" / "ping_S2_LONGBEACH_T001" / "info.xml.enc").is_file()


def test_transfer_state_observes_queue_to_sent_move_and_alert_metadata(monkeypatch, tmp_path):
    root = tmp_path / "src"
    priority = root / "downlink" / "queues" / "priority"
    sent = root / "sent"
    priority.mkdir(parents=True)
    sent.mkdir()
    filename = "alert-v1__S2_LONGBEACH_T001__S2_LONGBEACH__lat-33.680000__lon--118.170000__DARK_VESSEL.jpg"
    (priority / filename).write_bytes(b"jpeg")
    reset_transfer_tracker(monkeypatch, root)

    first = server.transfer_state()

    assert [stage["id"] for stage in first["stages"]] == ["priority", "standard", "sent"]
    assert first["files"][0]["alert"]["detection_id"] == "S2_LONGBEACH_T001"
    assert first["events"][0]["event"] == "queued_priority"

    (priority / filename).replace(sent / filename)
    second = server.transfer_state()

    assert second["files"][0]["stage"] == "sent"
    assert second["events"][0]["event"] == "downlinked"
    assert second["events"][0]["from_stage"] == "priority"


def test_transfer_image_rejects_traversal(monkeypatch, tmp_path):
    root = tmp_path / "src"
    (root / "sent").mkdir(parents=True)
    reset_transfer_tracker(monkeypatch, root)

    with pytest.raises(server.HTTPException) as error:
        server.transfer_image("sent", "../secret.jpg")

    assert error.value.status_code == 400


def test_transfer_state_reads_fleet_ping(monkeypatch, tmp_path):
    root = tmp_path / "src"
    ping = root / "fleet_alerts" / "longbeach" / "ping_S2_LONGBEACH_T001"
    ping.mkdir(parents=True)
    (ping / "image.jpg.enc").write_bytes(encrypt_bytes(b"jpeg", associated_data=b"image.jpg"))
    info = (
        "<fleet_ping><detection_id>S2_LONGBEACH_T001</detection_id><scene_id>S2_LONGBEACH</scene_id>"
        "<classification>DARK_VESSEL</classification><latitude>33.68</latitude><longitude>-118.17</longitude>"
        "<distance_nm>3.2</distance_nm><source_filename>alert.jpg</source_filename>"
        "<dispatched_at>2026-09-19T00:00:00Z</dispatched_at><fleet><id>longbeach</id>"
        "<name>Long Beach Response</name><latitude>33.71</latitude><longitude>-118.20</longitude>"
        "</fleet></fleet_ping>"
    )
    (ping / "info.xml.enc").write_bytes(encrypt_bytes(info.encode("utf-8"), associated_data=b"info.xml"))
    reset_transfer_tracker(monkeypatch, root)

    state = server.transfer_state()

    assert state["fleet_pings"][0]["fleet"]["name"] == "Long Beach Response"
    assert state["fleet_pings"][0]["image_url"].endswith("/image.jpg.enc")