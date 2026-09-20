"""
Telemetry and Resource Profiler for Jetson Orin NX Edge Execution.
Monitors memory footprint (RSS), CPU load, per-stage latency, and downlink data reduction.
"""

import os
import time
import threading
import psutil
from contextlib import contextmanager
from typing import Dict, Any, List

JETSON_MEMORY_BUDGET_MB = 14 * 1024
JETSON_CPU_CORES = 6


class EdgeTelemetryTracker:
    """
    Context manager and profiler for monitoring edge compute constraints.
    Peak RSS is sampled on a background thread so short-lived allocations inside a
    stage are not missed; the sampler only reads /proc and never touches pipeline state.
    """

    def __init__(self, label: str = "applet_execution", sample_interval_s: float = 0.02):
        self.label = label
        self.process = psutil.Process(os.getpid())
        self.sample_interval_s = sample_interval_s
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.start_memory_mb: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.input_file_bytes: int = 0
        self.input_raw_bytes: int = 0
        self.output_bytes: int = 0
        self.stages: List[Dict[str, Any]] = []
        self.records: Dict[str, Any] = {}
        self._cpu_start = None
        self._stop = threading.Event()
        self._sampler = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory_mb = self.start_memory_mb
        self._cpu_start = self.process.cpu_times()
        self._stop.clear()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()
        return self

    def _sample_loop(self):
        while not self._stop.wait(self.sample_interval_s):
            self.update_peak_memory()

    def update_peak_memory(self):
        try:
            current = self.process.memory_info().rss / (1024 * 1024)
        except psutil.Error:
            return
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current

    @contextmanager
    def stage(self, name: str):
        """
        Time one stage, and measure how much of the CPU budget it actually used.

        `cores_busy` is CPU-seconds over wall-seconds for this stage alone. It is a real
        measurement -- the container gives us honest `cpu_times()` -- and it is what the
        EdgeGovernor turns into a power estimate, so a stage that parallelises well reads
        as hotter than one that blocks on I/O. A cumulative average would smear that out
        and the governor would react to the wrong thing.
        """
        t0 = time.perf_counter()
        cpu0 = self.process.cpu_times()
        try:
            yield
        finally:
            self.update_peak_memory()
            seconds = time.perf_counter() - t0
            cpu1 = self.process.cpu_times()
            cpu_seconds = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
            self.stages.append({
                "stage": name,
                "seconds": round(seconds, 4),
                "cpu_seconds": round(cpu_seconds, 4),
                "cores_busy": round(cpu_seconds / max(seconds, 1e-4), 2),
            })

    def set_io_metrics(self, input_bytes: int, output_bytes: int, input_raw_bytes: int = 0):
        """input_bytes = files as stored; input_raw_bytes = uncompressed sensor samples."""
        self.input_file_bytes = input_bytes
        self.input_raw_bytes = input_raw_bytes or input_bytes
        self.output_bytes = output_bytes
        if self.records:
            self.records.update(self._io_records())

    def _io_records(self) -> Dict[str, Any]:
        raw, out = self.input_raw_bytes, self.output_bytes
        return {
            "input_file_bytes": self.input_file_bytes,
            "input_raw_sensor_bytes": raw,
            "input_size_mb": round(raw / (1024 * 1024), 3),
            "output_size_bytes": out,
            "output_size_kb": round(out / 1024, 2),
            "data_reduction_ratio": round(raw / out, 1) if out > 0 else 0.0,
            "bandwidth_savings_pct": round((1.0 - out / raw) * 100.0, 3) if raw > 0 and out > 0 else 0.0,
        }

    def snapshot(self, status: str = "RUNNING") -> Dict[str, Any]:
        now = self.end_time or time.perf_counter()
        duration = max(now - self.start_time, 1e-4)
        cpu = self.process.cpu_times()
        cpu_s = (cpu.user + cpu.system) - (self._cpu_start.user + self._cpu_start.system)
        rec = {
            "label": self.label,
            "wall_clock_time_s": round(duration, 4),
            "cpu_time_s": round(cpu_s, 4),
            "avg_cores_busy": round(cpu_s / duration, 2),
            "cpu_core_budget": JETSON_CPU_CORES,
            "start_memory_mb": round(self.start_memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "memory_budget_mb": JETSON_MEMORY_BUDGET_MB,
            "memory_budget_used_pct": round(100.0 * self.peak_memory_mb / JETSON_MEMORY_BUDGET_MB, 2),
            "stages": list(self.stages),
            "status": status,
        }
        rec.update(self._io_records())
        return rec

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
        self.update_peak_memory()
        self.records = self.snapshot("FAILED" if exc_type else "SUCCESS")

    def get_summary(self) -> Dict[str, Any]:
        return self.records or self.snapshot()

    def print_report(self):
        print_telemetry_report(self.get_summary())


def print_telemetry_report(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print(f"  EDGE TELEMETRY REPORT: {r.get('label')}")
    print("=" * 64)
    print(f"  Execution time     {r['wall_clock_time_s']} s  (CPU {r['cpu_time_s']} s, "
          f"{r['avg_cores_busy']} of {r['cpu_core_budget']} cores busy)")
    for s in r["stages"]:
        print(f"    - {s['stage']:<28}{s['seconds']:>9.4f} s")
    print(f"  Peak RAM (RSS)     {r['peak_memory_mb']} MB  ({r['memory_budget_used_pct']}% of 14 GB budget)")
    print(f"  Raw sensor input   {r['input_size_mb']} MB")
    print(f"  Downlink package   {r['output_size_kb']} KB")
    print(f"  Data reduction     {r['data_reduction_ratio']}x  ({r['bandwidth_savings_pct']}% bandwidth saved)")
    print(f"  Status             {r['status']}")
    print("=" * 64 + "\n")
