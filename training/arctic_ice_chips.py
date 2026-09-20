"""
Mines sea-ice hard negatives from the Arctic probe scenes into verifier training chips.

The verifier generalises to turbid bays and busy ports but not to ice. On the four real Arctic
Sentinel-2 scenes it passes 149 contacts over 1,226 km2 of water and every one of them is a floe,
an ice edge or bright speckle in sun-glinted water -- scored at verifier_prob 1.00. Floes are
bright, ship-sized, high-contrast objects on dark water, which is exactly the signature the CFAR
exists to find, so this is a distribution the network has never seen rather than a threshold
problem.

Three kinds of chip are mined, all labelled 0.0:

  contact     the 149 adjudicated false alarms (apex_px from the arctic scorecard) -- the exact
              chips the deployed model is confidently wrong about
  candidate   every candidate the detector proposes with its acceptance thresholds switched off,
              the same convention training/train_verifier.py and training/sen2ms.py mine under
  ice / sea   random positions in the sea mask, half of them biased to the brightest NIR pixels
              (floe fragments, glint speckle), so the network sees ice background as well as the
              ice objects the detector happened to trip on

There are ZERO real vessels in these four scenes, which is what makes every chip here safe to
label 0 -- and also the reason one scene is held out. We have Arctic negatives and no Arctic
positives anywhere in the corpus, so a model trained on all four scenes could learn "Arctic-looking
=> not a ship" and we would have no way to see it. ARC_KOTZEBUE (freeze-up, a different season from
the other three) is written to a separate file and kept out of training, so the ice rejection rate
on it is a generalisation number rather than a memorisation one.

    python training/arctic_ice_chips.py

Writes data/real/arctic_ice/{ice_chips,ice_chips_train,ice_chips_holdout}.npz with the
data/real/adjudicated convention: "chips" float32 (N,64,64,4) reflectance in band order
red,green,blue,nir and "labels" float32, all zero.
"""

import os
import sys
import json
import argparse
import collections
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.config import AppletConfig  # noqa: E402
from applet.pipelines.quality_screener import ImageQualityScreener  # noqa: E402
from applet.pipelines.vessel_detector import VesselDetector  # noqa: E402
from applet.utils.geo import SceneGeoreference  # noqa: E402
from applet.utils.image_io import load_scene_raster, CANONICAL_BANDS  # noqa: E402

CHIP = 64
HOLDOUT_SCENES = {"ARC_KOTZEBUE"}


def spaced_sample(xs: np.ndarray, ys: np.ndarray, want: int, min_sep: int, rng) -> List[Tuple[int, int]]:
    """Draw up to `want` positions at least min_sep px apart, so chips do not simply overlap."""
    order = rng.permutation(len(xs))
    picked: List[Tuple[int, int]] = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all(abs(x - px) >= min_sep or abs(y - py) >= min_sep for px, py in picked):
            picked.append((x, y))
            if len(picked) >= want:
                break
    return picked


def mine_scene(bundle: str, manifest: Dict[str, Any], scene_meta: Dict[str, Any],
               contacts: List[Dict[str, Any]], args, rng) -> Tuple[List[np.ndarray], List[str]]:
    sid = scene_meta["id"]
    refl, nodata, status = load_scene_raster(
        os.path.join(bundle, scene_meta["file"]),
        band_names=manifest.get("bands", list(CANONICAL_BANDS)),
        reflectance_scale=manifest.get("reflectance_scale"),
        reflectance_offset=float(scene_meta.get("reflectance_offset", 0.0)),
    )
    if refl is None:
        print(f"  ! {sid}: {status['error']}")
        return [], []

    h, w = refl.shape[:2]
    gsd = float(scene_meta.get("gsd_meters", manifest.get("gsd_meters", 10.0)))
    cfg = AppletConfig()
    cfg.detection.min_physics_score = 0.0          # keep every candidate: the clutter is the signal
    cfg.detection.clutter_density_per_km2 = 1e9    # never switch to rough-sea suppression here
    screener, detector = ImageQualityScreener(cfg), VesselDetector(cfg)
    scene = {
        "id": sid, "array": refl, "nodata_mask": nodata, "gsd_m": gsd,
        "status": {"missing_bands": status.get("missing_bands", []), "warnings": status.get("warnings", [])},
        "georef": SceneGeoreference(scene_meta, w, h, gsd),
    }
    scene.update(screener.screen_scene(scene))
    sea = np.asarray(scene["sea_mask"]).astype(bool)   # the screener returns uint8 0/1

    positions: List[Tuple[int, int, str]] = []
    seen = set()

    def add(x: int, y: int, tag: str) -> None:
        key = (x // 8, y // 8)   # one chip per 8 px cell: nearer than that is the same object
        if key in seen or not (0 <= x < w and 0 <= y < h):
            return
        seen.add(key)
        positions.append((x, y, tag))

    # 1. the adjudicated false alarms themselves
    for det in contacts:
        add(int(det["apex_px"][0]), int(det["apex_px"][1]), "contact")

    # 2. everything the detector proposes with its gates open
    dets, _ = detector.detect_scene(scene)
    cand = [(int(d["apex_px"][0]), int(d["apex_px"][1])) for d in dets]
    rng.shuffle(cand)
    for x, y in cand[:args.max_candidates]:
        add(x, y, "candidate")

    # 3. bright NIR pixels inside the sea mask: floe fragments and glint speckle the CFAR did not
    #    happen to peak on, but that a 64 px chip centred anywhere in the ice zone contains
    nir = refl[:, :, 3]
    if sea.any():
        cut = float(np.percentile(nir[sea], 97.0))
        ys, xs = np.nonzero(sea & (nir >= cut))
        for x, y in spaced_sample(xs, ys, args.bright_per_scene, 24, rng):
            add(x, y, "bright_ice")

        # 4. plain sea-mask background: ice floes, leads, open water, whatever is there
        ys, xs = np.nonzero(sea)
        idx = rng.choice(len(xs), size=min(len(xs), args.random_per_scene * 40), replace=False)
        for x, y in spaced_sample(xs[idx], ys[idx], args.random_per_scene, 24, rng):
            add(x, y, "background")

    chips, tags = [], []
    for x, y, tag in positions:
        chip = VesselDetector.crop_chip(refl, (x, y), CHIP)
        if chip.shape != (CHIP, CHIP, 4):
            continue
        chips.append(chip)
        tags.append(tag)
    counts = collections.Counter(tags)
    print(f"  {sid:<16} sea {sea.mean() * 100:5.1f}%  {len(chips):>4} chips  "
          f"{dict(sorted(counts.items()))}")
    del refl
    return chips, tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/real/arctic_probe")
    ap.add_argument("--contacts", default="data/outputs/arctic_scorecard/contacts.json",
                    help="scorecard contacts.json; gives apex_px for each of the 149 ice false alarms")
    ap.add_argument("--output", "-o", default="data/real/arctic_ice")
    ap.add_argument("--max-candidates", type=int, default=300, help="detector proposals kept per scene")
    ap.add_argument("--bright-per-scene", type=int, default=200)
    ap.add_argument("--random-per-scene", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    bundle = os.path.join(ROOT, args.bundle) if not os.path.isabs(args.bundle) else args.bundle
    with open(os.path.join(bundle, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.contacts, "r", encoding="utf-8") as f:
        payload = json.load(f)
    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for det in payload["contacts"]:
        by_scene[det["scene_id"]].append(det)
    print(f"{len(payload['contacts'])} adjudicated ice contacts over {len(by_scene)} scenes "
          f"(all false alarms: these scenes contain no vessel)")

    rng = np.random.default_rng(args.seed)
    split: Dict[str, Dict[str, list]] = {s: {"chips": [], "tags": [], "scenes": []}
                                         for s in ("train", "holdout")}
    for scene_meta in manifest["scenes"]:
        sid = scene_meta["id"]
        chips, tags = mine_scene(bundle, manifest, scene_meta, by_scene.get(sid, []), args, rng)
        key = "holdout" if sid in HOLDOUT_SCENES else "train"
        split[key]["chips"] += chips
        split[key]["tags"] += tags
        split[key]["scenes"] += [sid] * len(chips)

    out_dir = os.path.join(ROOT, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(out_dir, exist_ok=True)

    def write(name: str, parts: List[Dict[str, list]]) -> None:
        chips = np.asarray([c for p in parts for c in p["chips"]], dtype=np.float32)
        tags = np.asarray([t for p in parts for t in p["tags"]])
        scenes = np.asarray([s for p in parts for s in p["scenes"]])
        labels = np.zeros(len(chips), dtype=np.float32)   # no vessel exists in any Arctic probe scene
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(path, chips=chips, labels=labels, tags=tags, scene_ids=scenes)
        print(f"  {name:<18} {len(labels):>5} chips (0 positive) -> {path}")

    print("\nwriting:")
    write("ice_chips", [split["train"], split["holdout"]])
    write("ice_chips_train", [split["train"]])
    write("ice_chips_holdout", [split["holdout"]])
    print(f"\nheld out of training entirely: {', '.join(sorted(HOLDOUT_SCENES))} "
          f"-- ice rejection measured on it is generalisation, not recall of a memorised scene")
    return 0


if __name__ == "__main__":
    sys.exit(main())
