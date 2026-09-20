"""
Ship or ice? What separates them on real contacts, and what does not.

Every number quoted in applet/pipelines/arctic_classifier.py comes from here. Four questions, all
answered on the contacts the flight detector actually produces, with hand labels carried BY
POSITION (scripts/transfer_labels.py -- detection ids renumber between runs):

  1. Does the original icebergWatch evidence (centre brightness / flatness / texture) fire?
  2. Which object-level features separate ice from vessels, and do they survive a control?
  3. Fill-invariant spectral slope: fitted on Alaska + US, FROZEN, tested on Svalbard.
  4. Local crowding: same protocol.

Truth: Svalbard = 49 contacts labelled by hand; Alaska arctic_probe = every contact inspected,
zero vessels; US = team labels, plus any AIS-matched contact counted as a vessel.
Svalbard is never used to fit anything in 3. In 4 the neighbour threshold WAS read off the
Svalbard table, and its 5 vessels are all large -- so that result is optimistic, and says so.

    python scripts/iceberg_study.py
"""

import os
import sys
import json
import glob
import tempfile
import collections

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402
from applet.pipelines import chip_verifier  # noqa: E402
from transfer_labels import transfer  # noqa: E402

VERDICT = {"vessel": "vessel", "not_vessel": "clutter", "structure": "clutter", "unsure": "unsure"}
CHIPS = {}


def _keep_chips():
    """The verifier is the last stage that holds the chip; keep a copy of each survivor's."""
    original = chip_verifier.ChipVerifier.process

    def process(self, context):
        held = {id(d): d["chip_tensor"] for d in context.get("detected_vessels", []) if "chip_tensor" in d}
        context = original(self, context)
        for d in context.get("detected_vessels", []):
            if id(d) in held:
                CHIPS[d["detection_id"]] = held[id(d)].astype(np.float32)
        return context

    chip_verifier.ChipVerifier.process = process


def auc(pos, neg):
    """P(pos > neg), ties averaged. 0.5 is a coin."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    both = np.concatenate([pos, neg])
    ranks = np.empty(len(both))
    ranks[both.argsort(kind="mergesort")] = np.arange(1, len(both) + 1)
    for v in np.unique(both):
        ranks[both == v] = ranks[both == v].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---- 1. the original evidence, verbatim from origin/icebergWatch (Megan Neville) -------------------
ORIGINAL = dict(iceberg_min=0.72, brightness=(0.18, 0.55), flatness=(0.55, 0.90), texture=(0.01, 0.12),
                min_wake_m=100.0)


def original_features(chip):
    h, w, _ = chip.shape
    centre = chip[h // 4:3 * h // 4, w // 4:3 * w // 4]
    mean = centre.mean(axis=(0, 1))
    return {
        "centre_brightness": float(mean[:3].mean()),
        "spectral_flatness": float(1.0 - np.std(mean) / max(float(mean.mean()), 1e-6)),
        "texture": float(np.mean(np.abs(np.diff(centre, axis=0))) + np.mean(np.abs(np.diff(centre, axis=1)))),
    }


def original_ice_score(det, chip):
    ramp = lambda v, lo_hi: float(np.clip((v - lo_hi[0]) / (lo_hi[1] - lo_hi[0]), 0.0, 1.0))  # noqa: E731
    f = original_features(chip)
    wake = 1.0 if (det.get("wake_length_m") or 0.0) >= ORIGINAL["min_wake_m"] else 0.0
    ship_p = det.get("verifier_prob", 0.5)
    return (0.35 * ramp(f["centre_brightness"], ORIGINAL["brightness"])
            + 0.25 * ramp(f["spectral_flatness"], ORIGINAL["flatness"])
            + 0.15 * ramp(f["texture"], ORIGINAL["texture"]) + 0.15 * (1.0 - wake) + 0.10 * (1.0 - ship_p)), f


# ---- 2. object-level features ------------------------------------------------------------------------
def object_features(chip):
    h, w, _ = chip.shape
    border = np.ones((h, w), bool)
    border[8:-8, 8:-8] = False
    bg = np.median(chip[border], axis=0)
    mad = 1.4826 * np.median(np.abs(chip[border] - bg), axis=0) + 1e-4
    nir_x = chip[..., 3] - bg[3]
    vis_x = chip[..., :3].mean(-1) - bg[:3].mean()
    mask = (np.maximum(nir_x / mad[3], vis_x / mad[:3].mean()) > 4.0).astype(np.uint8)
    n, lab, _, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    off = np.hypot(cent[1:, 0] - w / 2, cent[1:, 1] - h / 2)
    if off.min() > 10:
        return None
    obj = lab == 1 + int(off.argmin())
    # Summed band excess over the local water. For a sub-pixel object excess = fill * (R_obj - R_water),
    # so a RATIO of excesses cancels the fill fraction: it is the object's own spectral slope.
    r, g, b, nir = (chip[obj] - bg).sum(axis=0)
    vis = (r + g + b) / 3.0
    ys, xs = np.nonzero(obj)
    elong = 1.0
    if len(xs) >= 5:
        (_, _), (a1, a2), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        elong = max(a1, a2) / max(1.0, min(a1, a2))
    return {"area_px": float(obj.sum()), "elongation": float(elong),
            "nir_over_vis_excess": float(nir / vis) if vis > 1e-4 else np.nan,
            "blue_over_red_excess": float(b / r) if r > 1e-4 else np.nan,
            "bg_nir": float(bg[3]), "blobs_in_chip": float(n - 1)}


def run(bundle):
    context, _, _ = run_pass(bundle, tempfile.mkdtemp(prefix="iceberg_study_"), AppletConfig.load_from_yaml(None))
    return list(context.get("classified_targets", []))


def reference_run(label_ids, pattern):
    best = (0, None)
    for path in glob.glob(pattern):
        try:
            ids = {c["detection_id"] for c in json.load(open(path))["contacts"]}
        except Exception:
            continue
        best = max(best, (len(ids & label_ids), path))
    return best[1]


def load(name, bundle, labels_path, reference_glob):
    contacts = run(bundle)
    truth = {}
    if labels_path:
        labels = json.load(open(labels_path))
        labels = labels.get("labels", labels)
        ref = reference_run(set(labels), reference_glob)
        carried, _ = transfer(json.load(open(ref))["contacts"], labels, contacts, 4.0)
        truth = {k: VERDICT.get(v, "unsure") for k, v in carried.items() if not k.startswith("MISS_")}
    rows = []
    for d in contacts:
        chip = CHIPS.get(d["detection_id"])
        if chip is None:
            continue
        t = truth.get(d["detection_id"], "clutter" if labels_path is None else "unlabelled")
        if t == "unlabelled" and d.get("matched_vessel"):
            t = "vessel"  # a transponder under it
        score, orig = original_ice_score(d, chip)
        rows.append({"id": d["detection_id"], "set": name, "scene": d["scene_id"], "truth": t,
                     "wake": float(d.get("wake_length_m") or 0.0), "ais": bool(d.get("matched_vessel")),
                     "regime": bool(d.get("ice_regime")), "original_score": score, "original": orig,
                     "obj": object_features(chip)})
    return rows


def main() -> int:
    _keep_chips()
    rows = (load("svalbard", "data/real/svalbard_poc", "data/labels/svalbard_labels.json",
                 "data/labels/svalbard_reference_contacts.json")
            + load("alaska", "data/real/arctic_probe", None, "")
            + load("us", "data/real/s2_us_bundle", "data/labels/us_labels.json",
                   "data/labels/us_reference_contacts.json"))
    pick = lambda s, t: [r for r in rows if r["set"] == s and r["truth"] == t]  # noqa: E731

    print("\n" + "=" * 78 + "\n1. ORIGINAL icebergWatch evidence (needs score >= 0.72 to say ICEBERG)")
    for s in ("svalbard", "alaska", "us"):
        v = np.array([r["original_score"] for r in rows if r["set"] == s])
        b = np.array([r["original"]["centre_brightness"] for r in rows if r["set"] == s])
        print(f"   {s:9s} n={len(v):3d}  score max {v.max():.2f}  fires on {int((v >= 0.72).sum())}   "
              f"centre brightness median {np.median(b):.3f} p95 {np.percentile(b, 95):.3f}  (ramp starts at 0.18)")

    print("\n" + "=" * 78 + "\n2. OBJECT FEATURES, AUC (clutter scores higher => > 0.5)")
    print("   per-candidate ice_regime flag:  " + ", ".join(
        f"{s} {sum(r['regime'] for r in rows if r['set'] == s)}/{sum(r['set'] == s for r in rows)}"
        for s in ("svalbard", "alaska", "us")))
    ok = [r for r in rows if r["obj"]]
    groups = [("Svalbard clutter vs Svalbard vessels (within-scene; tiny n)", "svalbard", "clutter", "svalbard"),
              ("Svalbard UNSURE  vs Svalbard vessels", "svalbard", "unsure", "svalbard"),
              ("US clutter vs US vessels  (CONTROL: fires here => not an ice feature)", "us", "clutter", "us")]
    for title, cs, ct, vs in groups:
        c = [r for r in ok if r["set"] == cs and r["truth"] == ct]
        v = [r for r in ok if r["set"] == vs and r["truth"] == "vessel"]
        print(f"\n   {title}: n={len(c)} vs {len(v)}")
        for k in ok[0]["obj"]:
            print(f"      {k:22s} AUC {auc([r['obj'][k] for r in c], [r['obj'][k] for r in v]):.3f}   "
                  f"median {np.nanmedian([r['obj'][k] for r in c]):8.3f} vs {np.nanmedian([r['obj'][k] for r in v]):8.3f}")

    fin = [r for r in ok if np.isfinite(r["obj"]["nir_over_vis_excess"]) and np.isfinite(r["obj"]["blue_over_red_excess"])]
    us_v = [r for r in fin if r["set"] == "us" and r["truth"] == "vessel"]
    alaska = [r for r in fin if r["set"] == "alaska"]
    print("\n" + "=" * 78 + "\n3. SPECTRAL SLOPE as a gate: fit on Alaska + US, freeze, test on Svalbard")
    best = None
    for a in (0.4, 0.5, 0.6, 0.7, 0.8):
        for b in (0.0, 1.0, 1.1, 1.2, 1.3):
            hit = lambda r: r["obj"]["nir_over_vis_excess"] < a and r["obj"]["blue_over_red_excess"] > b  # noqa: E731
            lost, got = sum(map(hit, us_v)), sum(map(hit, alaska))
            if lost <= 0.02 * len(us_v) and (best is None or got > best[0]):
                best = (got, lost, a, b)
    got, lost, a, b = best
    print(f"   best rule at <= 2 % US vessel loss: NIR/vis < {a}, blue/red > {b}: "
          f"catches {got}/{len(alaska)} Alaska ice, loses {lost}/{len(us_v)} US vessels")
    hit = lambda r: r["obj"]["nir_over_vis_excess"] < a and r["obj"]["blue_over_red_excess"] > b  # noqa: E731
    for t in ("vessel", "clutter", "unsure"):
        g = [r for r in fin if r["set"] == "svalbard" and r["truth"] == t]
        print(f"   FROZEN on Svalbard  {t:8s} called ice {sum(map(hit, g)):2d}/{len(g)}")

    print("\n" + "=" * 78 + "\n4. LOCAL CROWDING (blobs in the chip >= K), contacts with a wake or AIS match protected")
    print(f"   {'K':>3s} {'US vessels':>12s} {'Alaska':>10s} | Svalbard {'vessel':>7s} {'clutter':>8s} {'unsure':>7s}")
    for k in (4, 6, 8, 10, 12, 16):
        hit = lambda r: not (r["wake"] > 0 or r["ais"]) and r["obj"]["blobs_in_chip"] >= k  # noqa: E731
        cell = lambda g: f"{sum(map(hit, g))}/{len(g)}"  # noqa: E731
        sv = {t: [r for r in ok if r["set"] == "svalbard" and r["truth"] == t] for t in ("vessel", "clutter", "unsure")}
        print(f"   {k:3d} {cell([r for r in ok if r['set'] == 'us' and r['truth'] == 'vessel']):>12s} "
              f"{cell([r for r in ok if r['set'] == 'alaska']):>10s} |          "
              f"{cell(sv['vessel']):>7s} {cell(sv['clutter']):>8s} {cell(sv['unsure']):>7s}")
    print("\n   K=4 flags US vessels at anchor, so crowding is only usable where the SCENE holds ice\n"
          "   (screener ice_cover_pct: US max 0.7 %, Svalbard min 7.9 %). That is arctic.min_scene_ice_pct.")
    print("\n   End to end:  python scripts/scorecard.py -i data/real/svalbard_poc --arctic "
          "--labels data/outputs/svalbard_review/labels.json\n")
    counts = collections.Counter((r["set"], r["truth"]) for r in rows)
    print("   contacts by set and truth:", dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
