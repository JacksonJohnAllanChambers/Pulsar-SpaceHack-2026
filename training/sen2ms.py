"""
SEN2MS Vessel BBoxes (Zenodo 15571607, CC-BY-4.0, Dalhousie University) -> real-data training
chips and a real-data scorecard. Reads the zip in place (the arrays are 3 GB unpacked).

Each sample is a 128x128 px, 10 m, 13-band Sentinel-2 L1C chip; vessel chips carry YOLO boxes
and the matching AIS record (heading, SOG, length), non-vessel chips are verified empty.
Values are stretched to 0-255, not reflectance, so they are mapped to pseudo-reflectance with
one fixed factor (255 -> 0.30); the detector's tests are local-contrast ratios, which survive that.

Scenes (Sentinel-2 products), not chips, are split train/test so nothing leaks.

    python training/sen2ms.py mine       # -> data/real/sen2ms/train_chips.npz (train scenes only)
    python training/train_verifier.py --real-npz data/real/sen2ms/train_chips.npz
    python training/sen2ms.py evaluate   # held-out scenes: physics only vs + CNN, heading vs AIS
"""

import io
import os
import sys
import csv
import json
import math
import zlib
import zipfile
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.config import AppletConfig  # noqa: E402
from applet.pipelines.quality_screener import ImageQualityScreener  # noqa: E402
from applet.pipelines.vessel_detector import VesselDetector  # noqa: E402
from applet.pipelines.chip_verifier import ChipVerifier  # noqa: E402
from applet.utils.geo import SceneGeoreference  # noqa: E402

ZIP_PATH = os.path.join(ROOT, "data", "real", "sen2ms", "SEN2MS_Vessel_BBoxes.zip")
BAND_INDEX = [3, 2, 1, 7]  # B04 red, B03 green, B02 blue, B08 nir
PSEUDO_REFLECTANCE = 0.30 / 255.0
GSD = 10.0


NOMINAL_WATER = np.array([0.020, 0.040, 0.060, 0.012], dtype=np.float32)  # R, G, B, NIR


def harmonise(dn: np.ndarray) -> np.ndarray:
    """
    The published arrays are percentile-stretched per Sentinel-2 product, so the same sea reads
    2 DN in one product and 80 DN in another. The stretch offset is recovered from the chip itself:
    the darkest fifth of NIR pixels is taken as water and every band is shifted so that water sits
    at a nominal open-sea reflectance. Only the offset is corrected; the gain stays at the fixed
    255 -> 0.30 guess. This is dataset repair, done here so the flight thresholds stay physical.
    """
    nir = dn[:, :, 3]
    water = nir <= np.percentile(nir, 20)
    level = np.median(dn[water], axis=0)
    # Only water is darker in NIR than in green. If the darkest pixels are not (an all-land chip:
    # desert, terrain shadow), there is no sea to anchor to and no shift is applied.
    if level[3] > level[1]:
        return np.clip(dn * PSEUDO_REFLECTANCE, 0.0005, 1.2).astype(np.float32)
    return np.clip((dn - level) * PSEUDO_REFLECTANCE + NOMINAL_WATER, 0.0005, 1.2).astype(np.float32)


def is_test_scene(product: str) -> bool:
    return zlib.crc32(product.encode()) % 4 == 0  # ~25 % of Sentinel-2 products held out


def iter_samples(z: zipfile.ZipFile):
    """Yields dict(name, region, vessel, product, refl[H,W,4], boxes[(x0,y0,x1,y1)], ais{})."""
    names = set(z.namelist())
    for region in ("DEN", "USA", "EXT"):
        for kind in ("vessel", "non-vessel"):
            base = f"SEN2MS_Vessel_BBoxes/{region}/{kind}"
            csv_name = f"{base}/{'vessel_AIS_and_misc' if kind == 'vessel' else 'non-vessel_and_misc'}.csv"
            if csv_name not in names:
                continue
            rows = list(csv.DictReader(io.StringIO(z.read(csv_name).decode("utf-8", errors="replace"))))
            by_image = {}
            for r in rows:
                by_image.setdefault(r["Image_name"], []).append(r)
            for image_name, recs in by_image.items():
                stem = image_name.rsplit(".", 1)[0]
                npy = f"{base}/MS/{stem}.npy"
                if npy not in names:
                    continue
                arr = np.load(io.BytesIO(z.read(npy)))
                refl = harmonise(np.transpose(arr[BAND_INDEX], (1, 2, 0)).astype(np.float32))
                h, w = refl.shape[:2]
                boxes = []
                label = f"{base}/BBoxes/{stem}.txt"
                if label in names:
                    for line in z.read(label).decode().split("\n"):
                        p = line.split()
                        if len(p) == 5:
                            cx, cy, bw, bh = (float(v) for v in p[1:])
                            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
                yield {"name": stem, "region": region, "vessel": kind == "vessel", "refl": refl, "boxes": boxes,
                       "product": recs[0].get("Picture_title", stem), "ais": recs if kind == "vessel" else [],
                       "land": int(_float(recs[0].get("Land")) or 0)}  # 0 none, 1 some, 2 mostly land


def make_scene(sample):
    refl = sample["refl"]
    h, w = refl.shape[:2]
    return {"id": sample["name"], "array": refl, "nodata_mask": np.zeros((h, w), dtype=bool), "gsd_m": GSD,
            "status": {"missing_bands": [], "warnings": []},
            "georef": SceneGeoreference({"center_lat": 0.0, "center_lon": 0.0}, w, h, GSD)}


def in_box(x, y, box, slack=3.0):
    return box[0] - slack <= x <= box[2] + slack and box[1] - slack <= y <= box[3] + slack


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mine(args):
    cfg = AppletConfig()
    cfg.detection.min_physics_score = 0.0
    cfg.detection.clutter_density_per_km2 = 1e9
    screener, detector = ImageQualityScreener(cfg), VesselDetector(cfg)
    rng = np.random.default_rng(7)
    chips, labels = [], []
    n_scenes = 0
    with zipfile.ZipFile(args.zip) as z:
        for s in iter_samples(z):
            if is_test_scene(s["product"]):
                continue
            n_scenes += 1
            scene = make_scene(s)
            scene.update(screener.screen_scene(scene))
            dets = detector.detect_scene(scene)[0] if scene["sea_mask"].mean() > 0.02 else []
            hit = [False] * len(s["boxes"])
            for d in dets:
                x, y = d["apex_px"]
                k = next((i for i, b in enumerate(s["boxes"]) if in_box(x, y, b)), None)
                if k is not None:
                    hit[k] = True
                chips.append(detector.crop_chip(s["refl"], (x, y), 64))
                labels.append(1 if k is not None else 0)
            # ships the physics stage missed still teach the network what a real hull looks like
            for k, b in enumerate(s["boxes"]):
                if not hit[k]:
                    chips.append(detector.crop_chip(s["refl"], (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)), 64))
                    labels.append(1)
            if not s["vessel"] and rng.random() < 0.5:  # plain background
                chips.append(detector.crop_chip(s["refl"], (int(rng.integers(32, 96)), int(rng.integers(32, 96))), 64))
                labels.append(0)
            if n_scenes % 500 == 0:
                print(f"  {n_scenes} chips scanned: {sum(labels)} pos / {len(labels) - sum(labels)} neg")
    out = os.path.join(os.path.dirname(args.zip), "train_chips.npz")
    np.savez_compressed(out, chips=np.stack(chips).astype(np.float16), labels=np.array(labels, dtype=np.uint8))
    print(f"[DONE] {sum(labels)} positive / {len(labels) - sum(labels)} negative real chips -> {out}")


def evaluate(args):
    results = {}
    for mode in ("physics_only", "with_cnn"):
        cfg = AppletConfig()
        cfg.verifier.enabled = mode == "with_cnn"
        if args.model:
            cfg.verifier.model_path = args.model
        screener, detector, verifier = ImageQualityScreener(cfg), VesselDetector(cfg), ChipVerifier(cfg)
        tp = fp = fn = fp_on_empty = empty_chips = vessel_chips = 0
        head_err, len_err = [], []
        # The mission is contacts at sea. SEN2MS is dominated by ships alongside quays, which the
        # shoreline keep-out drops by design, so recall is also reported for the strata that matter.
        strata = {"open_water_chips": [0, 0], "underway_sog_ge_3kn": [0, 0], "moored_sog_lt_1kn": [0, 0]}
        with zipfile.ZipFile(args.zip) as z:
            for s in iter_samples(z):
                if not is_test_scene(s["product"]):
                    continue
                scene = make_scene(s)
                scene.update(screener.screen_scene(scene))
                ctx = {"screened_scenes": [scene]}
                ctx = verifier.process(detector.process(ctx))
                dets = ctx["detected_vessels"]
                used = set()
                for d in dets:
                    x, y = d["apex_px"]
                    k = next((i for i, b in enumerate(s["boxes"]) if i not in used and in_box(x, y, b)), None)
                    if k is None:
                        fp += 1
                        fp_on_empty += 0 if s["vessel"] else 1
                        continue
                    used.add(k)
                    tp += 1
                    if len(s["boxes"]) == 1 and len(s["ais"]) == 1:  # unambiguous AIS <-> box pairing
                        a = s["ais"][0]
                        hdg, sog, length = _float(a.get("Heading")), _float(a.get("SOG")), _float(a.get("Length"))
                        if hdg is not None and hdg <= 360 and sog is not None and sog >= 3 and not d["heading_ambiguous_180"]:
                            e = abs(d["heading_deg"] - hdg) % 360
                            head_err.append(min(e, 360 - e))
                        if length and length > 20 and d.get("hull_resolved", True):
                            len_err.append(abs(d["hull_length_m"] - length) / length)
                fn += len(s["boxes"]) - len(used)
                if s["vessel"] and s["land"] == 0:
                    strata["open_water_chips"][0] += len(used)
                    strata["open_water_chips"][1] += len(s["boxes"])
                if len(s["boxes"]) == 1 and len(s["ais"]) == 1:
                    sog = _float(s["ais"][0].get("SOG"))
                    key = "underway_sog_ge_3kn" if (sog is not None and sog >= 3) else (
                        "moored_sog_lt_1kn" if (sog is not None and sog < 1) else None)
                    if key:
                        strata[key][0] += len(used)
                        strata[key][1] += 1
                vessel_chips += int(s["vessel"])
                empty_chips += int(not s["vessel"])
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        head_err.sort()
        results[mode] = {
            "vessel_chips": vessel_chips, "empty_chips": empty_chips, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3),
            "recall_by_stratum": {k: {"found": v[0], "of": v[1], "recall": round(v[0] / max(v[1], 1), 3)}
                                  for k, v in strata.items()},
            "false_alarms_per_1000_km2_of_empty_scene": round(1000 * fp_on_empty / max(empty_chips * 1.28 * 1.28, 1e-9), 1),
            "heading_n": len(head_err),
            "heading_median_deg": round(head_err[len(head_err) // 2], 1) if head_err else None,
            "heading_within_20deg_pct": round(100 * sum(e <= 20 for e in head_err) / len(head_err), 1) if head_err else None,
            "heading_180_flips_pct": round(100 * sum(e >= 160 for e in head_err) / len(head_err), 1) if head_err else None,
            "length_mape_pct": round(100 * sum(len_err) / len(len_err), 1) if len_err else None, "length_n": len(len_err),
            "verifier": verifier.status if mode == "with_cnn" else "DISABLED",
        }
        print(mode, json.dumps(results[mode], indent=2))
    out = os.path.join(os.path.dirname(args.zip), f"scorecard{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["mine", "evaluate"])
    ap.add_argument("--zip", default=ZIP_PATH)
    ap.add_argument("--model", default=None, help="evaluate: verifier ONNX to use instead of the configured one")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"mine": mine, "evaluate": evaluate}[a.command](a)
