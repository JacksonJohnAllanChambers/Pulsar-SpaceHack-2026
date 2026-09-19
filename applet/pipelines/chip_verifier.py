"""
Micro-CNN chip verifier (Stage 3): local inference, only where physics says "look here".

The network never sees the scene -- only the 64x64 four-band chips the detector proposes,
typically a few dozen per 19 km swath. That keeps inference in the millisecond range on
the Orin's CPU cores alone, and leaves the GPU/DLA free (or powered down).

Runs through ONNX Runtime so the same graph executes on the CPU provider in the emulated
container and on the TensorRT/CUDA providers on real hardware, with no code change.
If the runtime or the model file is absent the stage degrades to physics-only scoring
instead of failing the pass.
"""

import os
import time
import numpy as np
from typing import Dict, Any, List, Optional

from applet.core.base import BasePipeline

CHIP_SCALE = 0.10  # reflectance units mapped to 1.0 at the network input


def normalise_chips(chips: np.ndarray) -> np.ndarray:
    """(N, H, W, 4) reflectance -> (N, 4, H, W) background-subtracted, fixed-scale tensor."""
    med = np.median(chips.reshape(chips.shape[0], -1, chips.shape[-1]), axis=1)
    x = (chips - med[:, None, None, :]) / CHIP_SCALE
    np.clip(x, -2.0, 5.0, out=x)
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)), dtype=np.float32)


class ChipVerifier(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.session = None
        self.model_path: Optional[str] = None
        self.status = "DISABLED"
        self.providers: List[str] = []
        if config.verifier.enabled:
            self._load()

    def _load(self) -> None:
        cfg = self.config.verifier
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = []
        for p in (cfg.model_path, cfg.fallback_model_path):
            candidates += [p, os.path.join(root, p)]
        path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if path is None:
            self.status = "MODEL_NOT_FOUND"
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = cfg.intra_op_threads
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            available = [p for p in wanted if p in ort.get_available_providers()]
            self.session = ort.InferenceSession(path, sess_options=opts, providers=available)
            self.providers = self.session.get_providers()
            self.model_path = path
            self.status = "READY"
        except Exception as e:
            self.status = f"RUNTIME_UNAVAILABLE ({type(e).__name__})"

    def predict(self, chips: np.ndarray) -> np.ndarray:
        x = normalise_chips(chips)
        name = self.session.get_inputs()[0].name
        out = []
        for i in range(0, len(x), 64):
            logits = self.session.run(None, {name: x[i:i + 64]})[0].reshape(-1)
            out.append(1.0 / (1.0 + np.exp(-logits.astype(np.float64))))
        return np.concatenate(out)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.verifier
        detections = context.get("detected_vessels", [])
        info = {
            "status": self.status,
            "model": os.path.basename(self.model_path) if self.model_path else None,
            "model_bytes": os.path.getsize(self.model_path) if self.model_path else 0,
            "providers": self.providers,
            "chips_inferred": 0,
            "inference_ms": 0.0,
            "rejected": 0,
        }

        if self.session is not None and detections:
            chips = np.stack([d["chip_tensor"] for d in detections])
            t0 = time.perf_counter()
            try:
                probs = self.predict(chips)
            except Exception as e:
                probs = None
                info["status"] = f"INFERENCE_FAULT ({type(e).__name__})"
            info["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            if probs is not None:
                info["chips_inferred"] = len(detections)
                for det, p in zip(detections, probs):
                    p = float(p)
                    det["verifier_prob"] = round(p, 3)
                    det["confidence"] = round(0.5 * det["physics_score"] + 0.5 * p, 3)
                    det["verifier_rejected"] = bool(
                        p < cfg.reject_below and det["physics_score"] < cfg.physics_override_score
                    )

        kept = [d for d in detections if not d.get("verifier_rejected")]
        rejected = [d for d in detections if d.get("verifier_rejected")]
        info["rejected"] = len(rejected)

        for scene in context.get("screened_scenes", []):
            dets = scene.get("detections", [])
            scene["rejected_candidates"] = [d for d in dets if d.get("verifier_rejected")]
            scene["detections"] = [d for d in dets if not d.get("verifier_rejected")]

        for d in detections:
            d.pop("chip_tensor", None)

        context["detected_vessels"] = kept
        context["verifier_rejected"] = rejected
        context["total_vessels_detected"] = len(kept)
        context["verifier_info"] = info
        context.setdefault("detection_funnel", {})["verified"] = len(kept)
        return context
