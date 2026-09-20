"""Conservative Arctic ship-versus-iceberg evidence fusion.

This is deliberately a three-way classifier. Iceberg classification requires positive
ice evidence and low ship evidence; otherwise the target remains UNCERTAIN instead of
being forced into a ship/ice decision.
"""

from typing import Any, Dict, Optional

import numpy as np


class ArcticClassifier:
    def __init__(self, config):
        self.config = config.arctic

    def classify(self, detection: Dict[str, Any], chip: np.ndarray, ship_probability: Optional[float]) -> None:
        if not self.config.enabled:
            return

        features = self._features(chip)
        wake_evidence = 1.0 if detection.get("wake_length_m", 0.0) >= self.config.min_wake_length_m else 0.0
        ship_probability = float(ship_probability) if ship_probability is not None else 0.5

        brightness = self._ramp(features["center_brightness"], self.config.ice_brightness_low,
                                self.config.ice_brightness_high)
        spectral = self._ramp(features["spectral_flatness"], self.config.spectral_flatness_low,
                               self.config.spectral_flatness_high)
        texture = self._ramp(features["texture"], self.config.texture_low, self.config.texture_high)
        ice_score = (0.35 * brightness + 0.25 * spectral + 0.15 * texture +
                     0.15 * (1.0 - wake_evidence) + 0.10 * (1.0 - ship_probability))
        ship_score = min(1.0, 0.65 * ship_probability + 0.35 * wake_evidence)

        detection["arctic_features"] = {key: round(value, 4) for key, value in features.items()}
        detection["iceberg_probability"] = round(float(ice_score), 3)
        detection["ship_probability"] = round(float(ship_score), 3)

        margin = self.config.decision_margin
        if ice_score >= self.config.iceberg_min_probability and ice_score - ship_score >= margin:
            label = "ICEBERG"
        elif ship_score >= self.config.ship_min_probability and ship_score - ice_score >= margin:
            label = "SHIP"
        else:
            label = "UNCERTAIN"
        detection["arctic_classification"] = label

    @staticmethod
    def _ramp(value: float, low: float, high: float) -> float:
        if high <= low:
            return float(value >= high)
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    @staticmethod
    def _features(chip: np.ndarray) -> Dict[str, float]:
        """Extract explainable chip features; input is HWC reflectance in RGBNIR order."""
        h, w, _ = chip.shape
        y0, y1 = h // 4, max(h // 4 + 1, 3 * h // 4)
        x0, x1 = w // 4, max(w // 4 + 1, 3 * w // 4)
        center = chip[y0:y1, x0:x1]
        border = np.concatenate((chip[:y0], chip[y1:], chip[y0:y1, :x0], chip[y0:y1, x1:]), axis=None)
        center_mean = center.mean(axis=(0, 1))
        center_brightness = float(center_mean[:3].mean())
        spectral_mean = max(float(center_mean.mean()), 1e-6)
        spectral_flatness = float(1.0 - np.std(center_mean) / spectral_mean)
        texture = float(np.mean(np.abs(np.diff(center, axis=0))) + np.mean(np.abs(np.diff(center, axis=1))))
        border_level = float(np.median(border)) if border.size else 0.0
        return {
            "center_brightness": center_brightness,
            "spectral_flatness": spectral_flatness,
            "texture": texture,
            "local_contrast": center_brightness - border_level,
        }