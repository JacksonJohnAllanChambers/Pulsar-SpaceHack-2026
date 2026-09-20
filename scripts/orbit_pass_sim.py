"""
What the EdgeGovernor costs, and what it buys.

Two experiments, one script:

  COST   Run the real pipeline on a real bundle, pinned to each rung of the cascade
         ladder in turn, and measure what each degradation actually costs in detections,
         runtime and downlink bytes. Everything in this table is MEASURED -- same code,
         same data, only the profile changes.

  BUY    Simulate an orbit's worth of continuous processing in two orbits -- a mid-beta
         SSO with a real eclipse, and a dawn-dusk SSO with none -- and show the governor
         picking a different steady-state answer for each. This half is MODELLED. The
         temperature is a prediction; see applet/core/thermal.py for the assumptions and
         why we could not validate them without hardware.

Usage:
    python scripts/orbit_pass_sim.py --input data/eval_bundle
    python scripts/orbit_pass_sim.py --input data/eval_bundle --json out/governor.json
"""

import argparse
import collections
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.core.governor import CASCADE_LADDER, EdgeGovernor, simulate_orbit  # noqa: E402
from applet.core.thermal import OrbitModel, describe_budget  # noqa: E402
from applet.runner import run_pass  # noqa: E402

ORBITS = {
    "mid-beta SSO": OrbitModel(),
    "dawn-dusk SSO": OrbitModel(eclipse_fraction=0.0),
}


def _one_pass(input_dir, profile):
    """One pinned run. Returns (wall, cpu_s, cores, telemetry, downlink, context)."""
    workdir = tempfile.mkdtemp(prefix="gov_" + profile.name + "_")
    try:
        config = AppletConfig()
        config.thermal.pin_profile = profile.name
        # Queue crops are a scheduler convenience and would add unrelated I/O to the
        # timing we are trying to compare. Off for all four rungs, equally.
        config.downlink.write_queues = False

        started = time.perf_counter()
        context, telemetry, downlink = run_pass(input_dir, workdir, config)
        wall = time.perf_counter() - started
        return wall, telemetry["cpu_time_s"], telemetry["avg_cores_busy"], telemetry, downlink, context
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def measure_profiles(input_dir, repeats=1):
    """
    Run the real cascade `repeats` times per rung. The only thing that varies is the profile.

    Repeats exist because the two halves of this table have very different reliability, and
    printing them at the same precision would be dishonest:

      EXACT   candidates, dark vessels, AIS-confirmed, downlink bytes. Identical every run --
              the pipeline is deterministic given a profile.
      NOISY   wall-clock and CPU-seconds. The trap is that within ONE invocation the repeats
              often look tight, so the ranges printed below read as trustworthy. They are not:
              re-running the whole script moved our FULL wall-clock median across 6.51 / 8.33 /
              9.73 s on the same machine and the same data. Drift between invocations dwarfs the
              spread inside one. Two careful attempts disagreed in SIGN -- five sequential
              repeats put REDUCED 30 % below FULL in CPU-seconds, six interleaved rounds put it
              11 % above.

    So timings are printed with their observed range and labelled unmeasured, and the ranges are
    printed precisely so a reader can see that for themselves. Quote the exact columns. Do NOT
    quote a speed-up or a slow-down from this script unless you have run it on a quiet machine
    and the ranges have actually separated.
    """
    rows = []
    for profile in CASCADE_LADDER:
        walls, cpus, cores = [], [], []
        telemetry = downlink = context = None
        for _ in range(max(1, repeats)):
            wall, cpu_s, core, telemetry, downlink, context = _one_pass(input_dir, profile)
            walls.append(wall)
            cpus.append(cpu_s)
            cores.append(core)

        classes = collections.Counter(
            t.get("classification") for t in context["classified_targets"])
        rows.append({
            "profile": profile.name,
            "power_mode_w": profile.power_mode_w,
            "repeats": len(walls),
            "wall_s": round(statistics.median(walls), 2),
            "wall_min_s": round(min(walls), 2),
            "wall_max_s": round(max(walls), 2),
            "cpu_s": round(statistics.median(cpus), 2),
            "cpu_min_s": round(min(cpus), 2),
            "cpu_max_s": round(max(cpus), 2),
            "avg_cores_busy": round(statistics.median(cores), 2),
            "peak_rss_mb": telemetry["peak_memory_mb"],
            "candidates": telemetry["funnel"].get("candidates", 0),
            "physics_accepted": telemetry["funnel"].get("physics_accepted", 0),
            "verified": telemetry["funnel"].get("verified", 0),
            "dark_vessels": classes.get("DARK_VESSEL", 0),
            "ais_confirmed": classes.get("CONFIRMED_KNOWN_VESSEL", 0),
            "kinematic_mismatch": classes.get("AIS_KINEMATIC_MISMATCH", 0),
            "bundle_bytes": downlink["final_bundle_bytes"],
            "chips": downlink["included_chips_count"],
            "verifier_status": telemetry["verifier"]["status"],
            "rationale": profile.rationale,
        })
    return rows


def run_orbit(orbit, minutes):
    """Build a governor for this orbit and hand it to the shared simulator."""
    config = AppletConfig()
    config.thermal.governor_enabled = True
    config.thermal.eclipse_fraction = orbit.eclipse_fraction
    return simulate_orbit(EdgeGovernor.from_config(config), minutes)


def print_cost_table(rows):
    reps = rows[0]["repeats"] if rows else 1
    print("\n" + "=" * 100)
    print("  COST OF DEGRADATION -- real pipeline, real bundle, only the cascade profile varies")
    print("=" * 100)

    print("\n  EXACT -- deterministic given a profile, identical every run:")
    header = ("  {:<9}{:>4}{:>8}{:>7}{:>7}{:>7}{:>6}{:>7}{:>13}"
              .format("profile", "W", "cand", "phys", "verif", "dark", "AIS", "chips", "downlink KB"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print("  {:<9}{:>4}{:>8}{:>7}{:>7}{:>7}{:>6}{:>7}{:>13.1f}".format(
            r["profile"], r["power_mode_w"], r["candidates"], r["physics_accepted"],
            r["verified"], r["dark_vessels"], r["ais_confirmed"], r["chips"],
            r["bundle_bytes"] / 1024))

    print("\n  NOISY -- median of {} run{} on a shared dev box, range in brackets.".format(
        reps, "" if reps == 1 else "s"))
    print("  These ranges cover ONE invocation and can look reassuringly tight. They are not.")
    print("  Repeating the whole invocation moved our FULL median across 6.51 / 8.33 / 9.73 s on the")
    print("  same machine and the same data -- drift between invocations is far larger than the")
    print("  spread inside one, and larger than the gaps between rows. Treat the timing block as")
    print("  unmeasured: not zero, and not necessarily the sign you expect.")
    header2 = "  {:<9}{:>22}{:>22}{:>7}{:>9}".format("profile", "wall s (range)", "CPU s (range)",
                                                     "cores", "RSS MB")
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for r in rows:
        wall = "{:.2f} [{:.2f}-{:.2f}]".format(r["wall_s"], r["wall_min_s"], r["wall_max_s"])
        cpu = "{:.2f} [{:.2f}-{:.2f}]".format(r["cpu_s"], r["cpu_min_s"], r["cpu_max_s"])
        print("  {:<9}{:>22}{:>22}{:>7.2f}{:>9.1f}".format(
            r["profile"], wall, cpu, r["avg_cores_busy"], r["peak_rss_mb"]))

    base = rows[0]
    print()
    print("  What a rung changes for certain is the POWER CAP and the worker count it selects --")
    print("  those are config, not measurements. Whether that shows up as wall-clock here is a")
    print("  separate question, and on this box the answer is that we cannot tell.")
    print("  >> Do not quote a speed number from a single invocation of this script. <<")
    if reps < 5:
        print("  NOTE: only {} repeat(s). Use --repeats 5 or more.".format(reps))
    print()
    for r in rows[1:]:
        print("  {:<9} AIS-confirmed {}/{}, dark-vessel alerts {} vs {} ({:.1f}x), "
              "downlink {:.1f} KB vs {:.1f} KB, power cap {} -> {} W".format(
                  r["profile"], r["ais_confirmed"], base["ais_confirmed"],
                  r["dark_vessels"], base["dark_vessels"],
                  r["dark_vessels"] / max(base["dark_vessels"], 1),
                  r["bundle_bytes"] / 1024, base["bundle_bytes"] / 1024,
                  base["power_mode_w"], r["power_mode_w"]))
        print("            " + r["rationale"])
        print()


def print_orbit_results(results):
    budget = describe_budget()
    print("\n" + "=" * 100)
    print("  THERMAL BUDGET -- MODELLED, never validated on hardware")
    print("=" * 100)
    print(f"  Radiator {budget['radiator_area_m2']} m2 rejects {budget['radiated_w_at_27c']} W at 27 C;"
          f" direct sun adds {budget['solar_load_w']} W back")
    print(f"  Sustainable SoC power:  eclipse {budget['sustainable_soc_w_eclipse']} W"
          f"   |   sunlit {budget['sustainable_soc_w_sunlit']} W")
    print(f"  We act at {budget['throttle_c']} C; Orin's own TJ_max is 105 C")

    print("\n" + "=" * 100)
    print("  GOVERNOR BEHAVIOUR -- continuous processing, same applet, two orbits")
    print("=" * 100)
    for name, r in results.items():
        verdict = "THROTTLED" if r["throttled"] else "stayed under the limit"
        print(f"\n  {name}  (eclipse {r['eclipse_fraction']:.0%}, {r['minutes']} min continuous)")
        print(f"    peak junction     {r['peak_junction_c']} C  -> {verdict}")
        print(f"    settled on        {r['final_profile']}")
        print(f"    rungs used        {', '.join(r['profiles_used'])}  ({r['transitions']} transitions)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/eval_bundle", help="input bundle to measure against")
    parser.add_argument("--minutes", type=int, default=400,
                        help="minutes of continuous processing to simulate per orbit")
    parser.add_argument("--json", help="write the full result, including the thermal trace, here")
    parser.add_argument("--repeats", type=int, default=1,
                        help="passes per profile; timings are the median with the range shown")
    parser.add_argument("--skip-cost", action="store_true",
                        help="only run the thermal simulation (no pipeline runs)")
    args = parser.parse_args()

    result = {"thermal_budget": describe_budget(), "validated_on_hardware": False}

    if not args.skip_cost:
        if not os.path.isdir(args.input):
            parser.error(f"input bundle not found: {args.input}")
        print(f"[INFO] measuring {len(CASCADE_LADDER)} profiles x {args.repeats} repeat(s) against {args.input} ...")
        result["cost_of_degradation"] = measure_profiles(args.input, args.repeats)
        print_cost_table(result["cost_of_degradation"])

    print(f"\n[INFO] simulating {args.minutes} min of continuous processing per orbit ...")
    result["orbits"] = {name: run_orbit(orbit, args.minutes) for name, orbit in ORBITS.items()}
    print_orbit_results(result["orbits"])

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[INFO] wrote {args.json}")

    print("\n  Reminder for the pitch: the COST table is measured; the THERMAL half is a model.")
    print()


if __name__ == "__main__":
    main()
