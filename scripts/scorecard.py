"""
Pooled scorecard over real scenes whose AIS is independent of their geolocation.

`training/sen2ms.py` scores chips whose position match is true by construction: each chip is
pinned to the AIS fix that produced its box, so "the detection is 40 m from the AIS position"
measures the crop, not the detector. This script scores whole Sentinel-2 scenes that were
georeferenced from their own COG transform and paired with the same day's public NOAA AIS.
Nothing about the imagery knows where the ships are, so every number below is evidence:

  recall vs AIS      of the broadcasters the sensor could actually see (clear water, resolvable,
                     not under cloud, outside the shore keep-out), how many did we find?
  position error     distance from each match to its dead-reckoned AIS position
  heading error      observed wake heading vs reported COG, for ships genuinely under way
  contacts w/o AIS   an upper bound on false alarms until the contact sheet is labelled; with
                     --labels it becomes a true precision.

    python scripts/scorecard.py -i data/real/s2_us_bundle -o data/outputs/us_scorecard
    python scripts/scorecard.py -i data/real/s2_us_bundle --labels data/outputs/review/labels.json
"""

import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402

M_PER_NM = 1852.0
MATCHED = ("CONFIRMED_KNOWN_VESSEL", "AIS_KINEMATIC_MISMATCH")


def pct(numerator: float, denominator: float) -> Optional[float]:
    return round(100.0 * numerator / denominator, 1) if denominator else None


def quantiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    s = sorted(values)
    return {
        "n": len(s),
        "median": round(s[len(s) // 2], 1),
        "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
        "max": round(s[-1], 1),
    }


def ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def catalog_index(catalog: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """MMSI -> AIS record. Entries carry source_scene, but a hull only matches inside its own scene."""
    out: Dict[Any, Dict[str, Any]] = {}
    for ship in catalog:
        out.setdefault(ship.get("mmsi"), ship)
    return out


def score(context: Dict[str, Any], telemetry: Dict[str, Any], labels: Optional[Dict[str, str]] = None
          ) -> Dict[str, Any]:
    ships = catalog_index(context.get("ais_catalog", []))
    scenes = {s["id"]: s for s in context.get("screened_scenes", [])}
    targets = context.get("classified_targets", [])
    unobserved = context.get("ais_not_observed", [])

    # Water the detector actually searched: sea pixels, which already exclude land and cloud.
    # Derived from the quality metrics rather than the masks, because run_pass frees the arrays
    # before returning -- and holding 16 scenes of rasters would inflate the RAM we report.
    water_km2 = 0.0
    per_scene: Dict[str, Dict[str, Any]] = {}
    for sid, scene in scenes.items():
        q = scene["quality_metrics"]
        usable = q["is_usable"]
        georef = scene["georef"]
        gsd = georef.gsd_m
        valid_px = (q["valid_data_pct"] / 100.0) * georef.width * georef.height
        km2 = (q["water_pct"] / 100.0) * valid_px * gsd * gsd / 1e6 if usable else 0.0
        water_km2 += km2
        per_scene[sid] = {
            "scene": sid, "usable": usable, "water_km2": round(km2, 1),
            "cloud_pct": q["cloud_cover_pct"],
            "contacts": 0, "matched": 0, "dark": 0, "mismatch": 0, "structure": 0,
            "ais_clear_water_missed": 0, "ais_in_port": 0,
        }

    pos_err_m: List[float] = []
    head_err: List[float] = []
    head_flipped = 0
    len_err_pct: List[float] = []
    speed_err: List[float] = []
    by_class: Dict[str, int] = {}
    contacts_no_ais: List[Dict[str, Any]] = []

    for det in targets:
        sid = det["scene_id"]
        row = per_scene.setdefault(sid, {"scene": sid, "contacts": 0, "matched": 0, "dark": 0,
                                         "mismatch": 0, "structure": 0, "water_km2": 0.0,
                                         "ais_clear_water_missed": 0, "ais_in_port": 0})
        row["contacts"] += 1
        cls = det["classification"]
        by_class[cls] = by_class.get(cls, 0) + 1

        if cls == "KNOWN_STRUCTURE":
            row["structure"] += 1
        if cls == "DARK_VESSEL":
            row["dark"] += 1
            contacts_no_ais.append(det)
        if cls == "AIS_KINEMATIC_MISMATCH":
            row["mismatch"] += 1
        if cls not in MATCHED:
            continue

        row["matched"] += 1
        if det.get("ais_distance_nm") is not None:
            pos_err_m.append(float(det["ais_distance_nm"]) * M_PER_NM)

        ship = ships.get(det.get("matched_vessel"))
        if not ship:
            continue
        sog = float(ship.get("sog_knots") or 0.0)
        # Heading is only meaningful for a ship genuinely under way; COG is noise at rest.
        if sog >= 3.0 and det.get("heading_deg") is not None:
            delta = ang_diff(float(det["heading_deg"]), float(ship.get("cog_deg") or 0.0))
            if delta > 135.0:
                head_flipped += 1
            head_err.append(min(delta, 180.0 - delta) if det.get("heading_ambiguous_180") else delta)
        length = float(ship.get("length_m") or 0.0)
        if length >= 20.0 and det.get("hull_resolved", True) and det.get("hull_length_m"):
            len_err_pct.append(100.0 * abs(float(det["hull_length_m"]) - length) / length)
        if det.get("estimated_speed_knots") is not None and sog >= 3.0:
            speed_err.append(abs(float(det["estimated_speed_knots"]) - sog))

    reasons: Dict[str, int] = {}
    for miss in unobserved:
        reasons[miss["reason"]] = reasons.get(miss["reason"], 0) + 1
        row = per_scene.get(miss["scene_id"])
        if row is None:
            continue
        if miss["reason"] == "CLEAR_WATER_NO_TARGET":
            row["ais_clear_water_missed"] += 1
        elif miss["reason"] == "IN_PORT_OR_SHORE_KEEPOUT":
            row["ais_in_port"] += 1

    n_matched = sum(1 for d in targets if d["classification"] in MATCHED)
    n_missed_visible = reasons.get("CLEAR_WATER_NO_TARGET", 0)
    visible_broadcasters = n_matched + n_missed_visible

    result: Dict[str, Any] = {
        "scenes": len(scenes),
        "usable_scenes": sum(1 for s in scenes.values() if s["quality_metrics"]["is_usable"]),
        "searched_water_km2": round(water_km2, 1),
        "contacts": len(targets),
        "classification_counts": by_class,
        "ais": {
            "broadcasters_in_footprint": len(unobserved) + n_matched,
            "visible_to_sensor": visible_broadcasters,
            "matched": n_matched,
            "missed_in_clear_water": n_missed_visible,
            "recall_vs_visible_ais": round(n_matched / visible_broadcasters, 3) if visible_broadcasters else None,
            "unmatched_reasons": reasons,
        },
        "position_error_m": quantiles(pos_err_m),
        "position_within_200m_pct": pct(sum(e <= 200.0 for e in pos_err_m), len(pos_err_m)),
        "heading_error_deg": quantiles(head_err),
        "heading_within_20deg_pct": pct(sum(e <= 20.0 for e in head_err), len(head_err)),
        "heading_flipped_180_pct": pct(head_flipped, len(head_err)),
        "length_error_pct": quantiles(len_err_pct),
        "speed_error_knots": quantiles(speed_err),
        "contacts_without_ais": len(contacts_no_ais),
        "contacts_without_ais_per_1000km2": round(1000.0 * len(contacts_no_ais) / water_km2, 1) if water_km2 else None,
        "wall_clock_s": telemetry["wall_clock_time_s"],
        "peak_memory_mb": telemetry["peak_memory_mb"],
        "verifier": telemetry.get("verifier", {}).get("status"),
        "per_scene": [per_scene[k] for k in sorted(per_scene)],
    }

    if labels:
        result["labelled"] = score_labels(targets, labels, water_km2, n_matched, visible_broadcasters)
    return result


def score_labels(targets: List[Dict[str, Any]], labels: Dict[str, str], water_km2: float,
                 n_matched: int, visible_broadcasters: int) -> Dict[str, Any]:
    """Folds human adjudication of every contact into a true precision.

    An AIS-matched contact is a vessel by definition, so only contacts without AIS need a verdict.
    Unlabelled contacts are excluded rather than assumed, and the count is reported.
    """
    vessel = notvessel = unlabelled = 0
    for det in targets:
        if det["classification"] in MATCHED:
            vessel += 1
            continue
        verdict = labels.get(det["detection_id"])
        if verdict == "vessel":
            vessel += 1
        elif verdict in ("not_vessel", "structure"):
            notvessel += 1
        else:
            unlabelled += 1
    adjudicated = vessel + notvessel
    return {
        "adjudicated_contacts": adjudicated,
        "unlabelled_contacts": unlabelled,
        "true_vessels": vessel,
        "false_alarms": notvessel,
        "precision": round(vessel / adjudicated, 3) if adjudicated else None,
        "false_alarms_per_1000km2": round(1000.0 * notvessel / water_km2, 1) if water_km2 else None,
        # Recall against everything we know to be a vessel: AIS ships the sensor could see, plus
        # the unlisted craft the reviewer confirmed (which we did find, by definition).
        "recall_vs_all_known_vessels": round(vessel / (vessel + (visible_broadcasters - n_matched)), 3)
        if vessel else None,
    }


def print_report(result: Dict[str, Any]) -> None:
    a = result["ais"]
    print(f"\n{'SCENE':<18}{'water km2':>10}{'cloud%':>8}{'contacts':>10}{'AIS hit':>9}"
          f"{'dark':>6}{'mismatch':>10}{'AIS missed':>12}{'in port':>9}")
    for row in result["per_scene"]:
        if not row.get("usable", True):
            print(f"{row['scene']:<18}{'-- unusable scene --':>50}")
            continue
        print(f"{row['scene']:<18}{row['water_km2']:>10.1f}{row.get('cloud_pct', 0):>8.1f}"
              f"{row['contacts']:>10}{row['matched']:>9}{row['dark']:>6}{row['mismatch']:>10}"
              f"{row['ais_clear_water_missed']:>12}{row['ais_in_port']:>9}")

    print(f"\n{'=' * 78}\nPOOLED OVER {result['usable_scenes']} USABLE SCENES "
          f"({result['searched_water_km2']:,.0f} km2 of searched water)\n{'=' * 78}")
    print(f"  AIS broadcasters in footprint      {a['broadcasters_in_footprint']}")
    print(f"    visible to the sensor            {a['visible_to_sensor']}  "
          f"(clear water, resolvable, not under cloud, outside shore keep-out)")
    print(f"    detected                         {a['matched']}")
    print(f"    missed in clear water            {a['missed_in_clear_water']}")
    print(f"  RECALL vs visible AIS              {a['recall_vs_visible_ais']}")
    print(f"  excluded, with reason              " + ", ".join(
        f"{k.lower()}={v}" for k, v in sorted(a["unmatched_reasons"].items()) if k != "CLEAR_WATER_NO_TARGET"))
    p, h = result["position_error_m"], result["heading_error_deg"]
    print(f"\n  Position error vs dead-reckoned AIS  median {p['median']} m, p90 {p['p90']} m, "
          f"max {p['max']} m  (n={p['n']}, {result['position_within_200m_pct']}% within 200 m)")
    print(f"  Heading error vs reported COG        median {h['median']} deg, "
          f"{result['heading_within_20deg_pct']}% within 20 deg, "
          f"{result['heading_flipped_180_pct']}% flipped 180 (n={h['n']})")
    print(f"  Hull length error                    median {result['length_error_pct']['median']}% "
          f"(n={result['length_error_pct']['n']})")
    print(f"\n  Contacts                             {result['contacts']}  " + ", ".join(
        f"{k.replace('_', ' ').lower()}={v}" for k, v in sorted(result["classification_counts"].items())))
    print(f"  Contacts with no AIS                 {result['contacts_without_ais']}  "
          f"({result['contacts_without_ais_per_1000km2']} per 1000 km2 -- upper bound on false alarms)")
    lab = result.get("labelled")
    if lab:
        print(f"\n  ADJUDICATED ({lab['adjudicated_contacts']} contacts reviewed, "
              f"{lab['unlabelled_contacts']} unlabelled and excluded)")
        print(f"    precision                        {lab['precision']}")
        print(f"    true false alarms                {lab['false_alarms']} "
              f"({lab['false_alarms_per_1000km2']} per 1000 km2)")
        print(f"    recall vs all known vessels      {lab['recall_vs_all_known_vessels']}")
    print(f"\n  {result['wall_clock_s']:.2f} s wall clock, {result['peak_memory_mb']:.0f} MB peak RAM, "
          f"verifier {result['verifier']}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="data/real/s2_us_bundle")
    ap.add_argument("--output", "-o", default="data/outputs/us_scorecard")
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--labels", default=None, help="labels.json exported from the contact sheet")
    args = ap.parse_args()

    config = AppletConfig.load_from_yaml(args.config)
    if args.no_verifier:
        config.verifier.enabled = False

    os.makedirs(args.output, exist_ok=True)
    context, telemetry, _ = run_pass(args.input, args.output, config)

    labels = None
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = json.load(f).get("labels", {})
        print(f"[labels] {len(labels)} human verdicts loaded from {args.labels}")

    result = score(context, telemetry, labels)
    print_report(result)

    name = "scorecard_no_verifier.json" if args.no_verifier else "scorecard.json"
    with open(os.path.join(args.output, name), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    # Everything the contact sheet needs, without re-running the pass.
    contacts = [{k: v for k, v in d.items() if not isinstance(v, (bytes, bytearray))}
                for d in context.get("classified_targets", [])]
    with open(os.path.join(args.output, "contacts.json"), "w", encoding="utf-8") as f:
        json.dump({"bundle": os.path.abspath(args.input), "contacts": contacts,
                   "ais_not_observed": context.get("ais_not_observed", [])}, f, indent=1)
    print(f"[done] {os.path.join(args.output, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
