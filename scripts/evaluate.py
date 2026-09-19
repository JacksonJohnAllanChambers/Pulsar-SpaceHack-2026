"""
Scores the applet against bundle ground truth (manifest "ground_truth" blocks).

Reports detection precision / recall / F1, heading, speed and length error, and how
often the AIS classification (dark / mismatch / known) is right. Run it with and without
the CNN stage to quantify what the verifier buys:

    python scripts/evaluate.py --input data/sample_bundle
    python scripts/evaluate.py --input data/sample_bundle --no-verifier
"""

import os
import sys
import json
import math
import argparse
import tempfile
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402

EXPECTED_CLASS = {"on": "CONFIRMED_KNOWN_VESSEL", "dark": "DARK_VESSEL",
                  "spoof_course": "AIS_KINEMATIC_MISMATCH", "spoof_static": "AIS_KINEMATIC_MISMATCH"}


def ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def score_bundle(input_dir: str, config: AppletConfig, verbose: bool = True) -> Dict[str, Any]:
    with open(os.path.join(input_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    # Evaluation runs are independent measurements; persistent unknown-area suppression belongs
    # to repeated operational passes and would otherwise leak state between benchmark runs.
    config.ais_correlation.unknown_memory_enabled = False
    with tempfile.TemporaryDirectory() as out_dir:
        context, telemetry, _ = run_pass(input_dir, out_dir, config)
    return score_context(manifest, context, telemetry, config.mission.gsd_meters, verbose)


def score_context(manifest: Dict[str, Any], context: Dict[str, Any], telemetry: Dict[str, Any],
                  default_gsd: float = 4.75, verbose: bool = False) -> Dict[str, Any]:
    """Scores an already-executed pass (shared with the ground-station GUI)."""
    gsd_default = float(manifest.get("gsd_meters", default_gsd))

    usable = {s["id"] for s in context["screened_scenes"] if s["quality_metrics"]["is_usable"]}
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for t in context["classified_targets"]:
        by_scene.setdefault(t["scene_id"], []).append(t)

    tp = fp = fn = 0
    head_err, speed_err, len_err, class_ok, class_n = [], [], [], 0, 0
    rows = []
    for entry in manifest.get("scenes", []):
        truth = entry.get("ground_truth")
        sid = entry["id"]
        if truth is None or sid not in usable:
            continue
        gsd = float(entry.get("gsd_meters", gsd_default))
        dets = list(by_scene.get(sid, []))
        matched = set()
        s_tp = s_fn = 0
        for v in truth["vessels"]:
            if not v["visible"]:
                continue
            gate_px = max(60.0, v["length_m"]) / gsd
            best, best_d = None, 1e9
            for i, d in enumerate(dets):
                if i in matched:
                    continue
                dist = math.hypot(d["apex_px"][0] - v["x"], d["apex_px"][1] - v["y"])
                if dist < gate_px and dist < best_d:
                    best, best_d = i, dist
            if best is None:
                s_fn += 1
                continue
            matched.add(best)
            s_tp += 1
            d = dets[best]
            if v["speed_knots"] >= 1.5 and not d["heading_ambiguous_180"]:
                head_err.append(ang_diff(d["heading_deg"], v["heading_deg"]))
            if d.get("estimated_speed_knots") is not None and v["speed_knots"] >= 1.5:
                speed_err.append(abs(d["estimated_speed_knots"] - v["speed_knots"]))
            if d.get("hull_resolved", True):
                len_err.append(abs(d["hull_length_m"] - v["length_m"]) / v["length_m"])
            if "nir" in (entry.get("bands") or ["nir"]):
                class_n += 1
                class_ok += int(d["classification"] == EXPECTED_CLASS[v["ais"]])
        s_fp = len(dets) - len(matched)
        tp, fp, fn = tp + s_tp, fp + s_fp, fn + s_fn
        rows.append((sid, s_tp, s_fp, s_fn))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        "heading_mae_deg": round(sum(head_err) / len(head_err), 1) if head_err else None,
        "heading_median_deg": round(sorted(head_err)[len(head_err) // 2], 1) if head_err else None,
        "heading_within_10deg_pct": round(100 * sum(e <= 10 for e in head_err) / len(head_err), 1) if head_err else None,
        "heading_n": len(head_err),
        "speed_mae_knots": round(sum(speed_err) / len(speed_err), 1) if speed_err else None,
        "speed_n": len(speed_err),
        "length_mape_pct": round(100 * sum(len_err) / len(len_err), 1) if len_err else None,
        "ais_class_accuracy": round(class_ok / class_n, 3) if class_n else None,
        "wall_clock_s": telemetry["wall_clock_time_s"], "peak_memory_mb": telemetry["peak_memory_mb"],
        "verifier": telemetry.get("verifier", {}).get("status"),
        "per_scene": [{"scene": sid, "tp": a, "fp": b, "fn": c} for sid, a, b, c in rows],
    }
    if verbose:
        print(f"{'SCENE':<12}{'TP':>4}{'FP':>5}{'FN':>5}")
        for sid, a, b, c in rows:
            print(f"{sid:<12}{a:>4}{b:>5}{c:>5}")
        print(json.dumps({k: v for k, v in result.items() if k != "per_scene"}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="data/sample_bundle")
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--no-verifier", action="store_true")
    args = ap.parse_args()
    config = AppletConfig.load_from_yaml(args.config)
    if args.no_verifier:
        config.verifier.enabled = False
    score_bundle(args.input, config)


if __name__ == "__main__":
    main()
