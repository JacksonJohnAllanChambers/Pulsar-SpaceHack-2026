import json

from applet.core.unknown_memory import UnknownContactMemory


def test_memory_requires_repeated_passes_and_persists(tmp_path):
    path = tmp_path / "unknown.json"
    memory = UnknownContactMemory(str(path), radius_nm=0.1, required_passes=3)

    memory.record_pass([{"latitude": 44.0, "longitude": -63.0}, {"latitude": 44.0001, "longitude": -63.0001}])
    memory.save()
    assert not memory.is_suppressed(44.0, -63.0)

    reloaded = UnknownContactMemory(str(path), radius_nm=0.1, required_passes=3)
    reloaded.record_pass([{"latitude": 44.0002, "longitude": -63.0002}])
    reloaded.save()
    assert not reloaded.is_suppressed(44.0, -63.0)

    reloaded = UnknownContactMemory(str(path), radius_nm=0.1, required_passes=3)
    reloaded.record_pass([{"latitude": 44.0003, "longitude": -63.0003}])
    reloaded.save()
    assert reloaded.is_suppressed(44.0, -63.0)
    assert not reloaded.is_suppressed(45.0, -63.0)
    assert not reloaded.is_suppressed(44.0, -63.0, [{"latitude": 44.0, "longitude": -63.0}])

    with path.open(encoding="utf-8") as stream:
        assert json.load(stream)["entries"][0]["passes"] == 3