"""
Scores one or more verifier ONNX models on held-out adjudicated chips.

The chips come from four scenes that were excluded from training entirely
(see training/adjudicated_chips.py), so this answers the only question that matters:
does the retrained model separate real vessels from real clutter on water it has never seen?

Reports AUC, and -- more usefully than accuracy -- the false-alarm rate that survives at the
operating threshold the applet actually flies (`verifier.reject_below`), together with how many
true vessels that costs.

    python training/eval_verifier.py --models applet/models/verifier_int8.onnx \
        data/outputs/models_before_adjudication/verifier_int8.onnx
"""

import os
import sys
import argparse
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.config import AppletConfig  # noqa: E402
from applet.pipelines.chip_verifier import normalise_chips  # noqa: E402


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run(model_path: str, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(x), 64):
        logit = sess.run(None, {name: x[i:i + 64]})[0].reshape(-1)
        out.append(1.0 / (1.0 + np.exp(-logit)))
    return np.concatenate(out)


def short_name(path: str) -> str:
    """Parent directory included: three of these files are called verifier_int8.onnx."""
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    return "/".join(parts[-2:]).replace(".onnx", "")[-24:]


def ice_report(models: List[str], scores: dict, ice_path: str, thr: float) -> None:
    """How much of the Arctic ice corpus each model still passes at the flight threshold."""
    data = np.load(ice_path)
    tags = data["tags"] if "tags" in data else np.full(len(data["labels"]), "ice")
    scene_ids = data["scene_ids"] if "scene_ids" in data else np.full(len(data["labels"]), "?")
    print(f"\n=== Arctic sea ice: {len(tags)} chips, all negative (no vessel exists in these scenes) ===")
    print("fraction still scoring >= threshold (lower is better)\n")
    groups = [("ALL", np.ones(len(tags), bool))]
    groups += [(f"tag {t}", tags == t) for t in sorted(set(tags.tolist()))]
    groups += [(f"scene {s}", scene_ids == s) for s in sorted(set(scene_ids.tolist()))]
    head = "".join(f"{short_name(m):>26}" for m in models)
    print(f"{'group':<26}{'n':>6}{head}")
    for name, mask in groups:
        cells = ""
        for m in models:
            p = scores[m][mask]
            cells += f"{f'{(p >= thr).mean():.3f}  (med {np.median(p):.2f})':>26}"
        print(f"{name:<26}{int(mask.sum()):>6}{cells}")


def background_stats(chips: np.ndarray) -> tuple:
    """Per chip: NIR background level, and background roughness with the centre object removed."""
    nir = chips[:, :, :, 3]
    level = np.median(nir.reshape(len(nir), -1), axis=1)
    ring = nir.copy()
    ring[:, 20:44, 20:44] = np.nan          # the object the detector centred on is not background
    flat = ring.reshape(len(ring), -1)
    rough = np.nanpercentile(flat, 90, axis=1) - np.nanpercentile(flat, 10, axis=1)
    return level, rough


def composite(hull_chips: np.ndarray, bg_chips: np.ndarray, rng) -> np.ndarray:
    """
    Paste the radiometric signature of a real vessel into a real background chip.

    We have Arctic negatives and no Arctic positives anywhere, so "would it still call a ship a
    ship in ice?" cannot be answered from data. This answers it synthetically: the hull's own
    anomaly (chip minus its local background median, windowed down to the object so the wake and
    the surrounding water do not come with it) is added to another chip's background. It is a
    proxy, not evidence -- but it is the SAME proxy for both models, so a model that scores these
    far lower than its predecessor has learned the background rather than the object.
    """
    h, w = hull_chips.shape[1:3]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    win = np.clip((18.0 - r) / 6.0, 0.0, 1.0)[None, :, :, None]   # soft disc, radius 18 px
    med = np.median(hull_chips.reshape(len(hull_chips), -1, 4), axis=1)[:, None, None, :]
    signal = (hull_chips - med) * win
    pick = rng.integers(0, len(bg_chips), size=len(hull_chips))
    return np.clip(bg_chips[pick] + signal, 0.0, 1.2).astype(np.float32)


def shortcut_report(models: List[str], test_path: str, ice_path: str, hull_path: str, thr: float) -> None:
    """
    The failure this whole exercise risks: learning "Arctic-looking scene => not a ship".

    Two independent probes. (1) Real held-out vessels split by how bright and how textured their
    background is -- if the new model's losses sit in the bright/rough half, it is rejecting on
    background. (2) Real vessel signatures composited into real ice and into real Arctic open
    water; an honest model keeps calling those vessels.
    """
    rng = np.random.default_rng(11)
    test = np.load(test_path)
    ty, tc = test["labels"].astype(np.float32), test["chips"].astype(np.float32)
    pos = tc[ty == 1]
    level, rough = background_stats(pos)
    strata = [("held-out vessels, dark bg", level <= np.median(level)),
              ("held-out vessels, bright bg", level > np.median(level)),
              ("held-out vessels, smooth bg", rough <= np.median(rough)),
              ("held-out vessels, rough bg", rough > np.median(rough))]

    ice = np.load(ice_path)
    ic = ice["chips"].astype(np.float32)
    ilevel, _ = background_stats(ic)
    bright_bg = ic[ilevel >= np.percentile(ilevel, 67)]
    dark_bg = ic[ilevel <= np.percentile(ilevel, 33)]

    hull = np.load(hull_path)
    hulls = hull["chips"].astype(np.float32)[hull["labels"].astype(np.float32) == 1]
    reps = max(1, 400 // max(len(hulls), 1))
    hulls = np.repeat(hulls, reps, axis=0)
    cases = [("vessel signature on ice floe bg", composite(hulls, bright_bg, rng)),
             ("vessel signature on Arctic water", composite(hulls, dark_bg, rng)),
             ("vessel signature on own bg (ctl)", composite(hulls, hulls, rng))]

    print(f"\n=== Shortcut check: did it learn 'Arctic => not a ship'? ===")
    print(f"vessel-side metrics, so HIGHER is better; composites are synthetic "
          f"({len(hulls)} from {os.path.basename(hull_path)})\n")
    head = "".join(f"{short_name(m):>26}" for m in models)
    print(f"{'probe':<36}{'n':>6}{head}")
    rows = [(name, pos[mask]) for name, mask in strata] + cases
    for name, chips in rows:
        if len(chips) == 0:
            continue
        x = normalise_chips(chips)
        cells = ""
        for m in models:
            p = run(m, x)
            cells += f"{f'kept {(p >= thr).mean():.3f}  med {np.median(p):.2f}':>26}"
        print(f"{name:<36}{len(chips):>6}{cells}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="data/real/adjudicated/test_chips.npz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="defaults to the flight config's verifier.reject_below")
    ap.add_argument("--ice-chips", default=None,
                    help="Arctic ice negatives (training/arctic_ice_chips.py); adds a rejection table")
    ap.add_argument("--shortcut", action="store_true",
                    help="probe for 'Arctic-looking => not a ship' on held-out vessels and composites")
    ap.add_argument("--hull-chips", default="data/real/adjudicated/train_chips.npz",
                    help="--shortcut: where the composited vessel signatures come from. Training "
                         "chips by default, so the probe does not consume the held-out set")
    args = ap.parse_args()

    thr = args.threshold if args.threshold is not None else AppletConfig().verifier.reject_below
    data = np.load(args.chips)
    y = data["labels"].astype(np.float32)
    x = normalise_chips(data["chips"].astype(np.float32))
    print(f"{len(y)} held-out chips: {int(y.sum())} vessel / {int(len(y) - y.sum())} not a vessel")
    print(f"operating threshold (reject_below) = {thr}\n")

    print(f"{'model':<46}{'AUC':>7}{'kept':>7}{'FA kept':>9}{'vessels lost':>14}{'precision':>11}")
    for path in args.models:
        if not os.path.exists(path):
            print(f"{os.path.relpath(path):<46}  missing")
            continue
        p = run(path, x)
        keep = p >= thr
        tp = int(((y == 1) & keep).sum())
        fp = int(((y == 0) & keep).sum())
        lost = int(((y == 1) & ~keep).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        print(f"{os.path.relpath(path):<46}{auc_score(y, p):>7.3f}{tp + fp:>7}{fp:>9}"
              f"{lost:>14}{prec:>11.3f}")

        # Where the scores actually sit: a model that cannot separate puts both classes at ~1.0
        pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
        print(f"{'':<46}  vessel  median {np.median(pos):.3f}  p10 {pos[int(.1 * len(pos))]:.3f}")
        print(f"{'':<46}  clutter median {np.median(neg):.3f}  p90 {neg[int(.9 * len(neg))]:.3f}")

    present = [m for m in args.models if os.path.exists(m)]
    if args.ice_chips:
        ice = np.load(args.ice_chips)
        xi = normalise_chips(ice["chips"].astype(np.float32))
        ice_report(present, {m: run(m, xi) for m in present}, args.ice_chips, thr)
    if args.shortcut:
        shortcut_report(present, args.chips, args.ice_chips or "data/real/arctic_ice/ice_chips.npz",
                        args.hull_chips, thr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
