"""
Turns hand-adjudicated contacts into verifier training chips.

The verifier was trained on synthetic scenes and SEN2MS crops, and on full real scenes it is
confidently wrong: it scores genuine vessels 0.998 and false alarms 0.993, so no threshold
separates them. Those false alarms -- sandbars, shoals, jetties, wave texture in turbid bays --
are a distribution it has never seen. This exports them as labelled chips so it can.

Split is BY SCENE, never by chip. Contacts from one scene are highly correlated (same water,
same sun angle, same seabed), so a chip-level split would leak the test set into training and
report an improvement that does not exist.

    python training/adjudicated_chips.py                 # -> data/real/adjudicated/{train,test}_chips.npz
    python training/train_verifier.py --scenes 500 \
        --real-npz data/real/adjudicated/train_chips.npz --extra-npz data/real/sen2ms/train_chips.npz
"""

import os
import sys
import json
import argparse
import collections
from typing import Any, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.pipelines.vessel_detector import VesselDetector  # noqa: E402
from applet.utils.image_io import load_scene_raster, CANONICAL_BANDS  # noqa: E402

CHIP = 64
MATCHED = ("CONFIRMED_KNOWN_VESSEL", "AIS_KINEMATIC_MISMATCH")

# Held out entirely from training so the retrained model can be scored on scenes it has never
# seen. Chosen to span the failure modes rather than to flatter: Tampa is the turbid-bay worst
# case (0.19 precision), Long Beach the busy port with the most AIS confirmations, Puget Sound
# clean deep water, New York a mid-difficulty approach. Miami stays in training, so "generalises
# to a turbid bay it never saw" remains a real question the Tampa result answers.
TEST_SCENES = {"S2_TAMPA", "S2_LONGBEACH", "S2_PUGETSOUND", "S2_NEWYORK"}


def load_scene(bundle: str, scene: Dict[str, Any], manifest: Dict[str, Any]):
    refl, _, status = load_scene_raster(
        os.path.join(bundle, scene["file"]),
        band_names=manifest.get("bands", list(CANONICAL_BANDS)),
        reflectance_scale=manifest.get("reflectance_scale"),
        reflectance_offset=float(scene.get("reflectance_offset", 0.0)),
    )
    return refl, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", "-s", default="data/outputs/us_scorecard")
    ap.add_argument("--labels", "-l", default="data/outputs/review/labels_team.json")
    ap.add_argument("--output", "-o", default="data/real/adjudicated")
    args = ap.parse_args()

    with open(os.path.join(args.scorecard, "contacts.json"), "r", encoding="utf-8") as f:
        payload = json.load(f)
    bundle = payload["bundle"]
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    with open(os.path.join(bundle, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    scenes = {s["id"]: s for s in manifest["scenes"]}

    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for det in payload["contacts"]:
        by_scene[det["scene_id"]].append(det)

    out: Dict[str, Dict[str, list]] = {"train": {"chips": [], "labels": []},
                                       "test": {"chips": [], "labels": []}}
    stats = collections.defaultdict(lambda: collections.Counter())

    for sid in sorted(by_scene):
        scene = scenes.get(sid)
        if scene is None:
            continue
        split = "test" if sid in TEST_SCENES else "train"
        refl, status = load_scene(bundle, scene, manifest)
        if refl is None:
            print(f"  ! {sid}: {status['error']}")
            continue

        kept = 0
        for det in by_scene[sid]:
            verdict = labels.get(det["detection_id"])
            if det["classification"] in MATCHED:
                y = 1          # a live transponder is ground truth, whatever the wake looks like
            elif verdict == "vessel":
                y = 1
            elif verdict in ("not_vessel", "structure"):
                y = 0
            else:
                stats[split]["skipped_unjudged"] += 1
                continue       # 'unsure' and unlabelled are excluded, never guessed
            chip = VesselDetector.crop_chip(refl, det["apex_px"], CHIP)
            if chip.shape != (CHIP, CHIP, 4):
                stats[split]["bad_shape"] += 1
                continue
            out[split]["chips"].append(chip)
            out[split]["labels"].append(y)
            stats[split]["pos" if y else "neg"] += 1
            kept += 1
        print(f"  {sid:<18} {split:<5} {kept:>4} chips")
        del refl

    os.makedirs(args.output, exist_ok=True)
    for split in ("train", "test"):
        chips = np.asarray(out[split]["chips"], dtype=np.float32)
        ys = np.asarray(out[split]["labels"], dtype=np.float32)
        path = os.path.join(args.output, f"{split}_chips.npz")
        np.savez_compressed(path, chips=chips, labels=ys)
        c = stats[split]
        print(f"\n{split}: {len(ys)} chips ({int(ys.sum())} vessel / {len(ys) - int(ys.sum())} not), "
              f"{c['skipped_unjudged']} unjudged skipped -> {path}")
    print(f"\nheld-out scenes: {', '.join(sorted(TEST_SCENES))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
