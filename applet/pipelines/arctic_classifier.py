"""Conservative Arctic ship / iceberg / uncertain call for contacts with no AIS.

Megan Neville's design (branch icebergWatch), kept: three classes rather than two, an ICEBERG is
demoted in the downlink queue rather than deleted, and a close call stays UNCERTAIN for a human.
What changed is the evidence. Her brightness / flatness / texture ramps were set against a
synthetic chip and returned ICEBERG on 0 of 539 real contacts -- a real contact is ~12 pixels, so
the mean of the chip's central 32x32 is the water around it. Measured on the hand-adjudicated
Svalbard contacts, what separates calved glacier ice from a ship is not the object at all:

  * the scene holds ice (the screener's ice class), and
  * the contact is one of several bright objects crowded into the same few hundred metres --
    bergy bits and growlers arrive as a field, vessels at sea do not -- and
  * nothing says "under way": no wake, no lead through ice, no AIS.

The per-candidate ice regime in the detector cannot see this case. Glacier ice in a fjord floats
in open dark water, so the annulus around each piece really is water (0 of 49 Svalbard contacts
are in the ice regime); that module is for pack ice. Crowding alone is not usable either -- ships
at anchor crowd too (35 of 220 US vessels) -- which is why it only counts inside an icy scene.

Measured, Svalbard, 2 scenes, labels by hand: 21 of 26 clutter, 9 of 17 "cannot tell by eye" and
0 of 5 vessels called ICEBERG. Those 5 vessels are all large, and the neighbour threshold was read
off this same table, so treat it as optimistic. A small stationary hull inside a growler field
WILL be called ice: at 10 m that is the payload's information limit, and it is the reason the
contact is demoted and still downlinked, never dropped.

SHIP is only ever said by a transponder. A wake or a lead keeps a contact from being called ice,
but it does not make it a ship: asserted from the image alone, SHIP was wrong on 21 of 21 Alaska
pack-ice contacts and 3 of 5 adjudicated Svalbard ones, because leads and brash streaks are
linear too. Image evidence here can demote; it cannot confirm.

The object's spectral slope (ratio of band excess over the local water, which cancels sub-pixel
fill) is reported because an operator can read it -- glacier ice is blue and NIR-dark, 0.46
against 0.77-1.0 for hulls -- but it carries no vote: fitted on Alaska + US and frozen, it caught
1 of 104 Alaska ice contacts, and white superstructure scores like ice.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np


class ArcticClassifier:
    def __init__(self, config):
        self.config = config.arctic

    def applies(self, scene: Dict[str, Any]) -> bool:
        ice_pct = float(scene.get("quality_metrics", {}).get("ice_cover_pct") or 0.0)
        return self.config.enabled and ice_pct >= self.config.min_scene_ice_pct

    def evidence(self, chip: np.ndarray, gsd_m: float) -> Optional[Dict[str, float]]:
        """Chip-level evidence; input is HWC reflectance in R,G,B,NIR order."""
        h, w, _ = chip.shape
        rim = max(1, min(h, w) // 8)
        border = np.ones((h, w), bool)
        border[rim:-rim, rim:-rim] = False
        background = np.median(chip[border], axis=0)
        mad = 1.4826 * np.median(np.abs(chip[border] - background), axis=0) + 1e-4

        nir_excess = chip[..., 3] - background[3]
        vis_excess = chip[..., :3].mean(axis=-1) - background[:3].mean()
        z = np.maximum(nir_excess / mad[3], vis_excess / mad[:3].mean())
        bright = (z > self.config.blob_sigma).astype(np.uint8)
        count, labels, _, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if count <= 1:
            return None
        offsets = np.hypot(centroids[1:, 0] - w / 2.0, centroids[1:, 1] - h / 2.0)
        nearest = int(np.argmin(offsets))
        if offsets[nearest] > 10.0:  # nothing segmentable under the contact itself
            return None

        neighbours = count - 2
        area_km2 = (h * gsd_m) * (w * gsd_m) / 1e6
        excess = (chip[labels == nearest + 1] - background).sum(axis=0)
        visible = float(excess[:3].mean())
        return {
            "neighbours": int(neighbours),
            "neighbours_per_km2": round(neighbours / area_km2, 2),
            "nir_vis_slope": round(float(excess[3]) / visible, 3) if visible > 1e-4 else None,
        }

    def decide(self, detection: Dict[str, Any], scene_ice_pct: float, ais_matched: bool) -> str:
        cfg = self.config
        evidence = detection.get("arctic_evidence")
        wake_m = float(detection.get("wake_length_m") or 0.0)
        under_way = wake_m > 0.0 or bool(detection.get("lead_found")) or bool(detection.get("kelvin_arms_detected"))

        if ais_matched:
            label = "SHIP"
        elif (evidence is not None and not under_way
              and evidence["neighbours"] >= cfg.min_neighbours
              and evidence["neighbours_per_km2"] >= cfg.min_neighbours_per_km2):
            label = "ICEBERG"
        else:
            label = "UNCERTAIN"
        if evidence is not None:
            evidence["scene_ice_pct"] = round(float(scene_ice_pct), 2)
        detection["arctic_classification"] = label
        return label
