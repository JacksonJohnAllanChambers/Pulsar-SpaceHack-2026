"""
Command-Line Interface for the Onboard Edge Satellite Applet.
"""

import sys
import argparse
from applet.config import AppletConfig
from applet.core.exceptions import InvalidManifestError
from applet.core.telemetry import print_telemetry_report
from applet.runner import run_pass
from applet.utils.formatting import format_tactical_table


def parse_args():
    parser = argparse.ArgumentParser(
        prog="applet",
        description="Pulsar SpaceHack 2026: Onboard Satellite Edge Applet (NVIDIA Jetson Orin NX)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    run_parser = subparsers.add_parser("run", help="Run the edge inference pipeline on an input bundle")
    run_parser.add_argument("--input", "-i", required=True, help="Path to input bundle directory")
    run_parser.add_argument("--output", "-o", required=True, help="Path to output artifacts directory")
    run_parser.add_argument("--config", "-c", default=None, help="Path to uplinked mission config YAML")
    run_parser.add_argument(
        "--track",
        "-t",
        default="tactical",
        choices=["tactical", "track1", "track3", "track4", "screening"],
        help="Pipeline track configuration (default: tactical / Track 1)",
    )
    run_parser.add_argument("--no-verifier", action="store_true", help="Physics-only mode (skip the CNN stage)")
    run_parser.add_argument("--arctic", action="store_true",
                            help="In icy scenes, call contacts SHIP / ICEBERG / UNCERTAIN and demote probable ice")

    return parser.parse_args()


def main():
    # Windows consoles default to cp1252; never let a log line crash the pass
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = parse_args()
    if args.command != "run":
        print("Usage: python -m applet run --input <input_dir> --output <output_dir> [--config <file.yaml>]")
        sys.exit(0)

    print("\n" + "=" * 64)
    print("  PULSAR SPACEHACK 2026: ONBOARD EDGE SATELLITE APPLET")
    print("  Target: NVIDIA Jetson Orin NX (16GB) | Offline Mode")
    print("=" * 64)

    config = AppletConfig.load_from_yaml(args.config)
    if args.no_verifier:
        config.verifier.enabled = False
    if args.arctic:
        config.arctic.enabled = True
    print(f"[INFO] Mission: {config.mission.mission_name} | Track: {args.track.upper()}")
    print(f"[INFO] Input bundle: {args.input}")

    try:
        context, telemetry, downlink = run_pass(args.input, args.output, config, args.track, log=print)
    except InvalidManifestError as e:
        print(f"[ERROR] Input bundle rejected: {e}")
        sys.exit(2)

    for scene in context.get("screened_scenes", []):
        q = scene["quality_metrics"]
        verdict = "USABLE" if q["is_usable"] else "REJECTED: " + "; ".join(q["rejection_reasons"])
        print(f"[SCENE] {scene['id']:<12} cloud {q.get('cloud_cover_pct', 0):5.1f}%  "
              f"water {q.get('water_pct', 0):5.1f}%  targets {len(scene.get('detections', [])):3d}  {verdict}")
    for scene in context.get("rejected_scenes", []):
        print(f"[SCENE] {str(scene.get('id')):<12} REJECTED AT INGEST: {scene.get('error')}")

    v = telemetry.get("verifier", {})
    print(f"[VERIFIER] {v.get('status')} | model {v.get('model')} | {v.get('chips_inferred', 0)} chips in "
          f"{v.get('inference_ms', 0)} ms | {v.get('rejected', 0)} rejected")

    print_telemetry_report(telemetry)

    targets = context.get("classified_targets", [])
    if targets:
        print("TACTICAL INTELLIGENCE (downlink queue order):")
        print(format_tactical_table(targets))
        print(
            f"\n[SUMMARY] Dark vessels: {context.get('dark_vessels_count', 0)} | "
            f"Kinematic mismatches: {context.get('spoofing_anomalies_count', 0)} | "
            f"AIS missing in clear water: "
            f"{sum(1 for a in context.get('ais_not_observed', []) if a['reason'] == 'CLEAR_WATER_NO_TARGET')} | "
            f"Tarball: {downlink['downlink_tarball_path']} ({downlink['final_bundle_kb']} KB)\n"
        )
    else:
        print("[INFO] No targets detected or all scenes occluded/cloudy.")

    sys.exit(0)


if __name__ == "__main__":
    main()
