"""
Unit tests for edge telemetry profiler.
"""

import time
from applet.core.telemetry import EdgeTelemetryTracker


def test_telemetry_tracker_records_metrics():
    with EdgeTelemetryTracker(label="test_tracker") as tracker:
        time.sleep(0.02)
        # Allocate some memory to register delta
        dummy_data = [i for i in range(100000)]
        tracker.set_io_metrics(input_bytes=1000000, output_bytes=10000)

    summary = tracker.get_summary()
    assert summary["label"] == "test_tracker"
    assert summary["wall_clock_time_s"] >= 0.015
    assert summary["peak_memory_mb"] > 0
    assert summary["input_raw_sensor_bytes"] == 1000000
    assert summary["output_size_bytes"] == 10000
    assert summary["data_reduction_ratio"] == 100.0
    assert summary["status"] == "SUCCESS"


def test_stage_timings_are_recorded_in_order():
    with EdgeTelemetryTracker(label="stages") as tracker:
        with tracker.stage("first"):
            time.sleep(0.01)
        with tracker.stage("second"):
            pass
    stages = tracker.get_summary()["stages"]
    assert [s["stage"] for s in stages] == ["first", "second"]
    assert stages[0]["seconds"] >= 0.005
