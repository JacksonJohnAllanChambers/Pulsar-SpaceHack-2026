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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="data/real/adjudicated/test_chips.npz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="defaults to the flight config's verifier.reject_below")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
