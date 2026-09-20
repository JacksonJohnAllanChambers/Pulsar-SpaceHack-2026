import json
import hashlib

from applet.config import AppletConfig
from applet.runner import run_pass
from applet.core.unknown_memory import UnknownContactMemory


def test_memory_requires_repeated_dates_and_persists(tmp_path):
    path = tmp_path / "unknown.json"
    memory = UnknownContactMemory(str(path), radius_nm=0.1, required_passes=3)

    memory.record_pass([{"latitude": 44.0, "longitude": -63.0}, {"latitude": 44.0001, "longitude": -63.0001}], "2024-11-01")
    memory.save()
    assert not memory.is_suppressed(44.0, -63.0)

    reloaded = UnknownContactMemory(str(path), radius_nm=0.1, required_passes=3)
    reloaded.record_pass([{"latitude": 44.0002, "longitude": -63.0002}], "2024-11-06")
    reloaded.save()
    assert not reloaded.is_suppressed(44.0, -63.0)

    reloaded = UnknownContactMemory(str(path), radius_nm=0.1, required_passes=3)
    reloaded.record_pass([{"latitude": 44.0003, "longitude": -63.0003}], "2024-11-11")
    reloaded.save()
    assert reloaded.is_suppressed(44.0, -63.0)
    assert not reloaded.is_suppressed(45.0, -63.0)
    assert not reloaded.is_suppressed(44.0, -63.0, [{"latitude": 44.0, "longitude": -63.0}])

    with path.open(encoding="utf-8") as stream:
        assert json.load(stream)["entries"][0]["passes"] == 3


def test_reprocessing_the_same_acquisition_teaches_it_nothing(tmp_path):
    memory = UnknownContactMemory(str(tmp_path / "unknown.json"), radius_nm=0.1, required_passes=3)
    for _ in range(10):
        memory.record_pass([{"latitude": 44.0, "longitude": -63.0}], "2024-11-01")
    memory.record_pass([{"latitude": 44.0, "longitude": -63.0}], None)  # undated: a re-run and a revisit look alike
    assert memory.entries[0]["passes"] == 1
    assert not memory.is_suppressed(44.0, -63.0)


def test_an_area_that_stops_recurring_is_forgotten(tmp_path):
    memory = UnknownContactMemory(str(tmp_path / "unknown.json"), radius_nm=0.1, required_passes=3, max_age_days=180)
    for day in ("2024-01-01", "2024-01-06", "2024-01-11"):
        memory.record_pass([{"latitude": 44.0, "longitude": -63.0}], day)
    assert memory.is_suppressed(44.0, -63.0)
    memory.record_pass([{"latitude": 10.0, "longitude": 10.0}], "2024-12-01")
    assert not memory.is_suppressed(44.0, -63.0)


def test_memory_on_keeps_repeat_runs_byte_identical(bundle_dir, tmp_path):
    # The failure this guards: counted per run, the fourth pass over one bundle reported zero contacts
    config = AppletConfig()
    config.ais_correlation.unknown_memory_enabled = True
    seen = set()
    for _ in range(5):
        context, telemetry, downlink = run_pass(bundle_dir, str(tmp_path / "out"), config)
        with open(downlink["downlink_tarball_path"], "rb") as f:
            seen.add((len(context["classified_targets"]), hashlib.sha256(f.read()).hexdigest()))
    assert len(seen) == 1 and next(iter(seen))[0] > 0
    assert telemetry["funnel"]["suppressed_persistent"] == 0
