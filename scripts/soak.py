"""
Long-run soak: the same process handles pass after pass, the way an orbiting one would.

Every other measurement here starts a fresh Python and exits, which hides exactly the faults
that matter over a mission: memory that grows a little each pass, latency that creeps as the
part warms, or output that stops being reproducible after the first run. A satellite's applet
is a long-lived process, so this runs `run_pass` in-process N times and reports drift rather
than an average.

    python scripts/soak.py -i data/real/s2_us_bundle -n 40
    python scripts/soak.py -i data/sample_bundle -n 200 --quiet

Fails loudly (exit 1) if RSS grows past --max-growth-mb, if the last quartile is more than
--max-slowdown slower than the first, or if the downlink stops being byte-identical.
"""

import os
import sys
import gc
import json
import time
import hashlib
import argparse
import tempfile
import statistics
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quartile_medians(values: List[float]):
    q = max(1, len(values) // 4)
    return statistics.median(values[:q]), statistics.median(values[-q:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="data/real/s2_us_bundle")
    ap.add_argument("--iterations", "-n", type=int, default=40)
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--max-growth-mb", type=float, default=250.0,
                    help="RSS growth from the first pass to the last that counts as a leak")
    ap.add_argument("--max-slowdown", type=float, default=0.15,
                    help="fractional slowdown of the last quartile vs the first that counts as throttling")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = AppletConfig.load_from_yaml(args.config)
    times: List[float] = []
    rss: List[float] = []
    digests = set()
    contacts = set()

    print(f"soak: {args.iterations} passes over {args.input}, one process\n")
    print(f"{'pass':>5}{'wall s':>9}{'RSS MB':>10}{'contacts':>10}")
    with tempfile.TemporaryDirectory() as out_dir:
        for i in range(args.iterations):
            t0 = time.perf_counter()
            context, telemetry, downlink = run_pass(args.input, out_dir, config)
            dt = time.perf_counter() - t0

            # Collect before sampling so what we measure is retained memory, not garbage
            # that simply has not been swept yet.
            gc.collect()
            times.append(dt)
            rss.append(rss_mb())
            contacts.add(len(context.get("classified_targets", [])))
            tar = downlink.get("downlink_tarball_path")
            if tar and os.path.exists(tar):
                digests.add(sha(tar))
            if not args.quiet or i in (0, args.iterations - 1):
                print(f"{i + 1:>5}{dt:>9.3f}{rss[-1]:>10.0f}{max(contacts):>10}")

    t_first, t_last = quartile_medians(times)
    growth = rss[-1] - rss[0]
    slowdown = (t_last - t_first) / t_first if t_first else 0.0

    print(f"\n{'=' * 60}")
    print(f"  passes                {len(times)}")
    print(f"  wall clock            median {statistics.median(times):.3f} s, "
          f"min {min(times):.3f}, max {max(times):.3f}")
    print(f"  first vs last quartile{t_first:>8.3f} s -> {t_last:.3f} s  ({slowdown:+.1%})")
    print(f"  RSS                   {rss[0]:.0f} MB -> {rss[-1]:.0f} MB  ({growth:+.0f} MB), "
          f"peak {max(rss):.0f} MB")
    print(f"  contacts per pass     {sorted(contacts)}")
    print(f"  distinct downlinks    {len(digests)} {'(byte-identical)' if len(digests) == 1 else '(NOT STABLE)'}")

    problems = []
    if growth > args.max_growth_mb:
        problems.append(f"RSS grew {growth:.0f} MB over {len(times)} passes (limit {args.max_growth_mb:.0f})")
    if slowdown > args.max_slowdown:
        problems.append(f"last quartile {slowdown:.1%} slower than the first (limit {args.max_slowdown:.0%})")
    if len(digests) > 1:
        problems.append(f"{len(digests)} distinct downlink tarballs: output is not reproducible across passes")
    if len(contacts) > 1:
        problems.append(f"contact count varied across passes: {sorted(contacts)}")

    print(f"{'=' * 60}")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        return 1
    print("  PASS  no leak, no drift, byte-identical output across every pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
