import pytest

# ground/ is the ground station, not flight code, so FastAPI is a dev dependency and is absent
# from the judges' container. Without this guard the module fails to import, and pytest aborts
# the WHOLE run on a collection error -- so one ground-side test would take every flight test
# with it in exactly the environment we most need them to run.
pytest.importorskip("fastapi", reason="ground console dependency; not installed in the flight image")

from ground import server  # noqa: E402


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
    monkeypatch.setattr(server, "SENT_DIR", str(sent_dir))
    monkeypatch.setattr(server, "FLEET_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setattr(server, "FLEET_OUTPUT_DIR", str(tmp_path / "fleet_alerts"))

    response = server.dispatch_fleet_alerts()

    assert response["dispatched"][0]["fleet"]["id"] == "longbeach"
    assert len(response["fleets"]) == 21
    assert (tmp_path / "fleet_alerts" / "longbeach" / "ping_S2_LONGBEACH_T001" / "info.xml").is_file()