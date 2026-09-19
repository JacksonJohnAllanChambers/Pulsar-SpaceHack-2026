"""
Trains, exports and quantises the chip verifier (ground-side; needs torch + onnx).

Training data is *mined*, not hand-drawn: random scenes are rendered, the real onboard
screener + detector is run over them with its acceptance thresholds switched off, and
every candidate it proposes is labelled from ground truth. The network therefore learns
exactly the distribution it will see in flight -- the detector's own mistakes (cloud
puffs, islets, whitecaps, foam patches in wakes) are its negatives.

Real chips can be mixed in with --real-npz (arrays "chips" [N,64,64,4] reflectance,
"labels" [N]) once a labelled set for the target sensor exists.

    python training/train_verifier.py --scenes 500 --epochs 14

Outputs applet/models/verifier_fp32.onnx, verifier_int8.onnx and model_card.json
(size / latency / accuracy before and after quantisation).
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.config import AppletConfig  # noqa: E402
from applet.pipelines.quality_screener import ImageQualityScreener  # noqa: E402
from applet.pipelines.vessel_detector import VesselDetector  # noqa: E402
from applet.pipelines.chip_verifier import normalise_chips  # noqa: E402
from applet.utils.geo import SceneGeoreference  # noqa: E402
from simulation.scene_synth import render_scene, random_spec  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "applet", "models")
CHIP = 64


def mine_chips(n_scenes: int, size: int, seed: int, max_neg_per_scene: int = 30):
    cfg = AppletConfig()
    cfg.detection.min_physics_score = 0.0
    cfg.detection.clutter_density_per_km2 = 1e9  # keep the clutter: it is the training signal
    screener, detector = ImageQualityScreener(cfg), VesselDetector(cfg)
    rng = np.random.default_rng(seed)

    chips, labels, groups = [], [], []
    t0 = time.time()
    for i in range(n_scenes):
        gsd = float(rng.choice([4.75, 4.75, 4.75, 10.0]))
        spec = random_spec(rng, size, gsd)
        refl, truth = render_scene(spec)
        scene = {
            "id": f"S{i}", "array": refl, "nodata_mask": np.zeros(refl.shape[:2], bool), "gsd_m": gsd,
            "status": {"missing_bands": [], "warnings": []},
            "georef": SceneGeoreference({"center_lat": 44.0, "center_lon": -63.0}, size, size, gsd),
        }
        scene.update(screener.screen_scene(scene))
        if scene["sea_mask"].mean() < 0.05:
            continue
        dets, _ = detector.detect_scene(scene)

        negs = 0
        for d in dets:
            x, y = d["apex_px"]
            positive = any(
                v["visible"] and math.hypot(x - v["x"], y - v["y"]) <= max(40.0, v["length_m"] / 2) / gsd
                for v in truth["vessels"]
            )
            if not positive:
                if negs >= max_neg_per_scene:
                    continue
                negs += 1
            chips.append(detector.crop_chip(refl, (x, y), CHIP))
            labels.append(1 if positive else 0)
            groups.append(i)

        # empty-sea / random-location negatives keep the network honest about "nothing there"
        for _ in range(2):
            x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
            if all(math.hypot(x - v["x"], y - v["y"]) > 40 for v in truth["vessels"]):
                chips.append(detector.crop_chip(refl, (x, y), CHIP))
                labels.append(0)
                groups.append(i)

        if (i + 1) % 50 == 0:
            print(f"  mined {i + 1}/{n_scenes} scenes: {sum(labels)} pos / {len(labels) - sum(labels)} neg "
                  f"({time.time() - t0:.0f}s)")
    return np.stack(chips), np.array(labels, dtype=np.float32), np.array(groups)


def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    return nn.Sequential(
        block(4, 16), block(16, 32), block(32, 48), block(48, 64),  # 64 -> 4
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1),
    )


def augment(x, rng):
    """x: torch tensor (N,4,H,W). Dihedral flips + gain jitter; all label-preserving."""
    import torch

    k = int(rng.integers(0, 4))
    x = torch.rot90(x, k, dims=(2, 3))
    if rng.random() < 0.5:
        x = torch.flip(x, dims=(3,))
    gain = torch.empty(x.shape[0], 1, 1, 1).uniform_(0.75, 1.3)
    return x * gain + torch.randn_like(x) * 0.02


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=500)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--real-npz", nargs="*", default=None,
                    help="npz files of real labelled chips (arrays 'chips', 'labels'). Append *N to "
                         "oversample one of them, e.g. adjudicated.npz*8 -- a few hundred "
                         "hand-adjudicated chips are otherwise swamped by thousands of mined ones")
    args = ap.parse_args()

    import torch
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

    torch.manual_seed(args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[1/4] Mining candidate chips from {args.scenes} synthetic scenes...")
    chips, labels, groups = mine_chips(args.scenes, args.size, args.seed)
    source = {"synthetic_chips": int(len(labels)), "real_chips": 0}
    for spec in (args.real_npz or []):
        path, _, mult = spec.partition("*")
        repeat = int(mult) if mult else 1
        real = np.load(path)
        rc = real["chips"].astype(np.float32)
        rl = real["labels"].astype(np.float32)
        if repeat > 1:
            rc = np.repeat(rc, repeat, axis=0)
            rl = np.repeat(rl, repeat, axis=0)
        rg = np.full(len(rl), -1) - np.arange(len(rl)) % 5  # spread over 5 pseudo-groups
        chips = np.concatenate([chips, rc])
        labels = np.concatenate([labels, rl])
        groups = np.concatenate([groups, rg])
        source["real_chips"] += int(len(rl))
        source.setdefault("real_sources", []).append({"file": os.path.basename(path),
                                                      "chips": int(len(real["labels"])),
                                                      "repeat": repeat})
        print(f"      + {len(rl)} real chips from {os.path.basename(path)} "
              f"({int(rl.sum())} pos){' x' + str(repeat) if repeat > 1 else ''}")

    x_all = normalise_chips(chips)
    val_mask = (groups % 5) == 0  # split by scene, never by chip
    x_tr, y_tr = torch.from_numpy(x_all[~val_mask]), torch.from_numpy(labels[~val_mask])
    x_va, y_va = x_all[val_mask], labels[val_mask]
    print(f"      train {len(y_tr)} ({int(y_tr.sum())} pos) | val {len(y_va)} ({int(y_va.sum())} pos)")

    print(f"[2/4] Training for {args.epochs} epochs on CPU...")
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * math.ceil(len(y_tr) / 64))
    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(y_tr))
        total = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(augment(x_tr[idx], rng)).squeeze(1), y_tr[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(x_va)).squeeze(1)).numpy()
        acc = float(((p > 0.5) == (y_va > 0.5)).mean())
        print(f"      epoch {epoch + 1:2d}  loss {total / len(perm):.4f}  val acc {acc:.3f}  AUC {auc_score(y_va, p):.4f}")

    print("[3/4] Exporting ONNX and quantising to INT8...")
    fp32_path = os.path.join(MODEL_DIR, "verifier_fp32.onnx")
    int8_path = os.path.join(MODEL_DIR, "verifier_int8.onnx")
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 4, CHIP, CHIP), fp32_path, input_names=["chips"], output_names=["logit"],
                      dynamic_axes={"chips": {0: "n"}, "logit": {0: "n"}}, opset_version=17, dynamo=False)

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.it = iter([{"chips": data[i:i + 32]} for i in range(0, min(len(data), 512), 32)])

        def get_next(self):
            return next(self.it, None)

    quantize_static(fp32_path, int8_path, Reader(x_all[~val_mask]), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)

    print("[4/4] Benchmarking FP32 vs INT8 on the ONNX Runtime CPU provider...")
    card = {"architecture": "4x[conv3x3-BN-ReLU-pool] -> GAP -> FC", "parameters": int(n_params),
            "input": "N x 4 x 64 x 64 (R,G,B,NIR), median-subtracted reflectance / 0.10", "training_data": source,
            "validation_chips": int(len(y_va)), "variants": {}}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        logits = np.concatenate([sess.run(None, {"chips": x_va[i:i + 64]})[0].reshape(-1) for i in range(0, len(x_va), 64)])
        p = 1 / (1 + np.exp(-logits))
        batch = x_va[:32] if len(x_va) >= 32 else x_va
        sess.run(None, {"chips": batch})
        t0 = time.perf_counter()
        for _ in range(20):
            sess.run(None, {"chips": batch})
        ms = (time.perf_counter() - t0) / 20 * 1000
        card["variants"][name] = {
            "file": os.path.basename(path), "size_kb": round(os.path.getsize(path) / 1024, 1),
            "val_accuracy": round(float(((p > 0.5) == (y_va > 0.5)).mean()), 4), "val_auc": round(auc_score(y_va, p), 4),
            "latency_ms_per_32_chips": round(ms, 2), "latency_ms_per_chip": round(ms / len(batch), 3),
        }
        print(f"      {name}: {card['variants'][name]}")
    with open(os.path.join(MODEL_DIR, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"[DONE] Models + model_card.json written to {MODEL_DIR}")


if __name__ == "__main__":
    main()
