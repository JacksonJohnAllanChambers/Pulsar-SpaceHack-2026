"""
End-to-end: the hackathon's hard rules (no crash, bounded output, determinism) as tests.
"""

import os
import json
import hashlib
import tarfile

from applet.config import AppletConfig
from applet.runner import run_pass


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_end_to_end_pass(bundle_dir, tmp_path):
    context, telemetry, downlink = run_pass(bundle_dir, str(tmp_path / "out"), AppletConfig())

    ids = {s["id"]: s for s in context["screened_scenes"]}
    assert not ids["SCENE_02"]["quality_metrics"]["is_usable"]  # overcast -> rejected before detection
    assert any("CLOUD_COVER" in r for r in ids["SCENE_02"]["quality_metrics"]["rejection_reasons"])
    assert ids["SCENE_01"]["quality_metrics"]["is_usable"]
    assert any(s.get("id") == "SCENE_04" for s in context["rejected_scenes"])  # corrupt file survived
    assert "MISSING_BANDS_NIR" in ids["SCENE_08"]["quality_metrics"]["warnings"]

    assert context["dark_vessels_count"] >= 1
    assert context["confirmed_known_count"] >= 1

    tar_path = downlink["downlink_tarball_path"]
    assert downlink["within_budget"] and os.path.getsize(tar_path) < 500 * 1024
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        assert "downlink/tactical_intelligence.geojson" in names
        assert "downlink/scene_report.json" in names and "downlink/manifest.json" in names
        geo = json.load(tar.extractfile("downlink/tactical_intelligence.geojson"))
    assert geo["type"] == "FeatureCollection" and len(geo["features"]) >= len(context["classified_targets"])

    # Regression: the original scaffold wrote an empty telemetry file
    with open(downlink["telemetry_path"], "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["status"] == "SUCCESS" and saved["wall_clock_time_s"] > 0
    assert [s["stage"] for s in saved["stages"]][:2] == ["Ingest+Validate", "ImageQualityScreener"]
    assert saved["peak_memory_mb"] < 14 * 1024
    assert saved["data_reduction_ratio"] > 50


def test_output_is_byte_identical_across_runs(bundle_dir, tmp_path):
    _, _, a = run_pass(bundle_dir, str(tmp_path / "a"), AppletConfig())
    _, _, b = run_pass(bundle_dir, str(tmp_path / "b"), AppletConfig())
    assert _sha(a["downlink_tarball_path"]) == _sha(b["downlink_tarball_path"])


def test_known_traffic_does_not_spend_chip_budget(bundle_dir, tmp_path):
    context, _, downlink = run_pass(bundle_dir, str(tmp_path / "out"), AppletConfig())
    with tarfile.open(downlink["downlink_tarball_path"], "r:gz") as tar:
        chips = {os.path.basename(n)[:-4] for n in tar.getnames() if n.endswith(".jpg")}
    known = {t["detection_id"] for t in context["classified_targets"] if t["classification"] == "CONFIRMED_KNOWN_VESSEL"}
    assert chips and not (chips & known)


def test_screening_track_runs_without_detector(bundle_dir, tmp_path):
    context, telemetry, _ = run_pass(bundle_dir, str(tmp_path / "out"), AppletConfig(), track="track4")
    assert telemetry["status"] == "SUCCESS"
    assert context["classified_targets"] == []
    assert all("quality_metrics" in s for s in context["screened_scenes"])


def test_physics_only_mode_when_verifier_disabled(bundle_dir, tmp_path):
    cfg = AppletConfig()
    cfg.verifier.enabled = False
    context, telemetry, _ = run_pass(bundle_dir, str(tmp_path / "out"), cfg)
    assert telemetry["verifier"]["status"] == "DISABLED"
    assert context["classified_targets"]


def test_queue_crops_stay_with_the_pass_and_are_timed(bundle_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    stale = out / "queues" / "priority" / "PREVIOUS_PASS.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    context, telemetry, _ = run_pass(bundle_dir, str(out), AppletConfig())

    crops = list((out / "queues").rglob("*.jpg"))
    assert len(crops) == len(context["classified_targets"]) and not stale.exists()
    assert not (tmp_path / "src").exists()  # never relative to the working directory
    assert "QueueRouter" in json.dumps(telemetry)

    config = AppletConfig()
    config.downlink.write_queues = False
    run_pass(bundle_dir, str(tmp_path / "quiet"), config)
    assert not (tmp_path / "quiet" / "queues").exists()
