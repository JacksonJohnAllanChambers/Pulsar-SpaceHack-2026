"""
Carries hand verdicts from one run's contacts to another run's, by position.

Labels are keyed by detection_id, and ids are only stable while the detector sees the same
pixels. Change the imagery -- L2A to L1C, a haze perturbation, another resampling -- and the ids
reshuffle even though most contacts are the same objects at the same place. When two bundles
share a pixel grid, a contact within a few pixels of a labelled one *is* that object, so its
verdict carries over. Anything with no labelled neighbour stays unlabelled: never guessed.

    python scripts/transfer_labels.py --reference data/outputs/us_scorecard/contacts.json \
        --labels data/outputs/review/labels_team.json \
        --contacts data/outputs/us_scorecard_l1c/contacts.json -o data/outputs/review/labels_l1c.json
"""

import sys
import json
import argparse
import collections


def transfer(reference, labels, contacts, radius_px):
    by_scene = collections.defaultdict(list)
    for det in reference:
        if det["detection_id"] in labels:
            by_scene[det["scene_id"]].append(det)

    out, used, stats = {}, set(), collections.Counter()
    # Closest pairs first, each labelled contact claimed at most once
    pairs = []
    for det in contacts:
        x, y = det["apex_px"]
        for ref in by_scene.get(det["scene_id"], []):
            d = ((ref["apex_px"][0] - x) ** 2 + (ref["apex_px"][1] - y) ** 2) ** 0.5
            if d <= radius_px:
                pairs.append((d, det["detection_id"], ref["detection_id"]))
    for _, new_id, ref_id in sorted(pairs):
        if new_id in out or ref_id in used:
            continue
        out[new_id] = labels[ref_id]
        used.add(ref_id)
    stats["contacts"] = len(contacts)
    stats["labels_carried"] = len(out)
    stats["contacts_without_a_labelled_counterpart"] = len(contacts) - len(out)
    lost = [r for rs in by_scene.values() for r in rs if r["detection_id"] not in used]
    stats["labelled_vessels_not_redetected"] = sum(labels[r["detection_id"]] == "vessel" for r in lost)
    stats["labelled_clutter_not_redetected"] = sum(labels[r["detection_id"]] in ("not_vessel", "structure") for r in lost)
    return out, dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="contacts.json of the run the labels were made on")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--contacts", required=True, help="contacts.json of the new run")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--radius-px", type=float, default=4.0)
    args = ap.parse_args()

    with open(args.reference, "r", encoding="utf-8") as f:
        reference = json.load(f)["contacts"]
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    with open(args.contacts, "r", encoding="utf-8") as f:
        contacts = json.load(f)["contacts"]

    out, stats = transfer(reference, labels, contacts, args.radius_px)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"labels": out, "transferred_from": args.labels, "radius_px": args.radius_px, "stats": stats}, f, indent=1)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
