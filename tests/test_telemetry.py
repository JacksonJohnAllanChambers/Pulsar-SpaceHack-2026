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


def test_cpu_time_is_measured_under_emulation():
    """
    Regression: psutil reads every stage as free under QEMU.

    `/proc/self/stat` reports utime and stime as 0 inside an emulated linux/arm64 container, so
    `psutil.cpu_times()` returns zeros and the whole judging run reported "0.0 of 6 cores busy"
    -- which is also what the governor was sizing its power estimate from. A stage that spins
    the CPU must charge for it on every platform we run on.
    """
    with EdgeTelemetryTracker(label="cpu") as tracker:
        with tracker.stage("busy"):
            deadline = time.perf_counter() + 0.20
            while time.perf_counter() < deadline:
                pass
    summary = tracker.get_summary()
    busy = summary["stages"][0]
    assert busy["cpu_seconds"] >= 0.10, "a 0.2 s spin must report CPU time, not zero"
    assert busy["cores_busy"] >= 0.5
    assert summary["cpu_time_s"] >= busy["cpu_seconds"] * 0.9
    # Natively both clocks agree, so nothing is flagged; in the emulated container psutil reads
    # zero against a hot process clock and the flag goes up.
    assert summary["cpu_time_includes_emulator"] is False
