"""
Benchmark harness: repeated passes over a bundle, reported against the Orin NX envelope.

    python scripts/benchmark.py --input data/sample_bundle --iterations 5
    python scripts/benchmark.py --full-swath            # renders + runs a 4096x4096 scene

Numbers from the QEMU-emulated container measure "does it fit", not "how fast is a Jetson":
the prep guide is explicit that emulated ARM is far slower than the real CPU complex. Run
this natively on an ARM64 host (Apple Silicon, Graviton, a real Jetson) for timing claims.
"""

import os
import sys
import json
import argparse
import platform
import statistics
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="data/sample_bundle")
    ap.add_argument("--output", "-o", default="data/outputs/benchmark")
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--iterations", "-n", type=int, default=5)
    ap.add_argument("--full-swath", action="store_true", help="benchmark a freshly rendered 4096x4096 bundle")
    ap.add_argument("--no-verifier", action="store_true")
    args = ap.parse_args()

    config = AppletConfig.load_from_yaml(args.config)
    if args.no_verifier:
        config.verifier.enabled = False

    input_dir = args.input
    tmp = None
    if args.full_swath:
        tmp = tempfile.TemporaryDirectory()
        input_dir = tmp.name
        print("[INFO] Rendering full-swath bundle (4096x4096, 19.4 km)...")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "generate_synthetic_data.py"),
                        "--output", input_dir, "--full-swath"], check=True, capture_output=True)

    runs = []
    for i in range(args.iterations):
        _, telemetry, _ = run_pass(input_dir, args.output, config)
        runs.append(telemetry)
        print(f"[RUN {i + 1}/{args.iterations}] {telemetry['wall_clock_time_s']:.3f} s | "
              f"peak {telemetry['peak_memory_mb']:.0f} MB | {telemetry['avg_cores_busy']:.2f} cores | "
              f"{telemetry['output_size_kb']:.1f} KB out")

    # first run pays for imports, the ONNX session and page faults: report it separately
    warm = runs[1:] if len(runs) > 1 else runs
    wall = [r["wall_clock_time_s"] for r in warm]
    last = runs[-1]
    mpx = last["funnel"].get("pixels_screened", 0) / 1e6
    stage_names = [s["stage"] for s in last["stages"]]
    stage_mean = {n: statistics.mean(r["stages"][k]["seconds"] for r in warm) for k, n in enumerate(stage_names)}

    report = {
        "host": {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
                 "cpu_count": os.cpu_count()},
        "iterations": args.iterations,
        "cold_start_s": runs[0]["wall_clock_time_s"],
        "warm_mean_s": round(statistics.mean(wall), 4),
        "warm_stdev_s": round(statistics.pstdev(wall), 4),
        "megapixels_per_pass": round(mpx, 2),
        "throughput_mpx_per_s": round(mpx / statistics.mean(wall), 2),
        "peak_memory_mb_max": max(r["peak_memory_mb"] for r in runs),
        "memory_budget_used_pct": max(r["memory_budget_used_pct"] for r in runs),
        "avg_cores_busy": round(statistics.mean(r["avg_cores_busy"] for r in warm), 2),
        "stage_mean_s": {k: round(v, 4) for k, v in stage_mean.items()},
        "raw_sensor_mb": last["input_size_mb"], "downlink_kb": last["output_size_kb"],
        "data_reduction_ratio": last["data_reduction_ratio"],
        "verifier": last["verifier"],
        "deterministic_output_size": len({r["output_size_bytes"] for r in runs}) == 1,
    }

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "benchmark_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  BENCHMARK  ({report['host']['machine']} / {report['host']['system']}, {args.iterations} passes)")
    print("=" * 64)
    print(f"  Cold start            {report['cold_start_s']:.3f} s")
    print(f"  Warm pass             {report['warm_mean_s']:.3f} s  (+- {report['warm_stdev_s']:.3f})")
    print(f"  Throughput            {report['throughput_mpx_per_s']} Mpx/s over {report['megapixels_per_pass']} Mpx")
    for name, sec in report["stage_mean_s"].items():
        print(f"    - {name:<26}{sec:>8.4f} s")
    print(f"  Peak RAM              {report['peak_memory_mb_max']:.0f} MB ({report['memory_budget_used_pct']}% of 14 GB)")
    print(f"  Cores busy            {report['avg_cores_busy']} of 6")
    print(f"  Raw -> downlink       {report['raw_sensor_mb']} MB -> {report['downlink_kb']} KB "
          f"({report['data_reduction_ratio']}x)")
    print(f"  Output size stable    {report['deterministic_output_size']}")
    print("=" * 64)
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
