"""
Physics-informed vessel and wake detector (Stage 2).

Why this design, at 4.75 m GSD:

  * Open water reflects ~1-3 % in NIR; steel, paint and foam reflect 10-40 %. A local
    CFAR (constant false alarm rate) test on NIR therefore finds a 10-pixel hull that a
    full-scene CNN would have to tile gigabytes of empty ocean to see.
  * The part of a wake that survives at this resolution is the turbulent centreline
    streak -- a line, not a "V". We recover it with a ray transform centred on each
    hull: integrate contrast along 360 rays and keep the direction that stands out.
    Kelvin cusp arms (+-19.47 deg) and the transverse wavelength (lambda = 2*pi*V^2/g)
    are measured on the same rays when the ship is fast and large enough to show them,
    and reported as absent when it is not. Nothing is assumed.

Everything heavy is a whole-image OpenCV call; per-candidate work touches only a few
thousand pixels, so cost scales with the number of ships, not the size of the ocean.
"""

import math
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from applet.core.base import BasePipeline
from applet.pipelines.quality_screener import meters_to_px, area_to_px, disk, map_scenes
from applet.utils.geo import speed_from_kelvin_wavelength

_N_ANGLES = 360
_SPEED_RAY_OFFSET_DEG = 8


class _RayTable:
    """Integer pixel offsets for every (bearing, radius); built once per search radius."""

    _cache: Dict[int, "_RayTable"] = {}

    def __init__(self, radius_px: int):
        theta = np.radians(np.arange(_N_ANGLES, dtype=np.float64))[:, None]
        r = np.arange(radius_px, dtype=np.float64)[None, :]
        # image-frame bearing: 0 = up (-y), 90 = right (+x)
        self.dx = np.rint(np.sin(theta) * r).astype(np.int32)
        self.dy = np.rint(-np.cos(theta) * r).astype(np.int32)
        self.radius_px = radius_px

    @classmethod
    def get(cls, radius_px: int) -> "_RayTable":
        if radius_px not in cls._cache:
            cls._cache[radius_px] = cls(radius_px)
        return cls._cache[radius_px]


class VesselDetector(BasePipeline):
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        all_detections: List[Dict[str, Any]] = []
        funnel = {"pixels_screened": 0, "cfar_pixels": 0, "candidates": 0, "physics_accepted": 0}

        scenes = context.get("screened_scenes", [])
        usable = [s for s in scenes if s["quality_metrics"]["is_usable"]]
        for scene in scenes:
            scene["detections"] = []

        def work(scene: Dict[str, Any]):
            try:
                detections, stats = self.detect_scene(scene)
            except Exception as e:
                scene["quality_metrics"].setdefault("warnings", []).append(f"DETECTOR_FAULT ({type(e).__name__})")
                return None
            memory = context.get("unknown_memory")
            if memory is not None:
                detections = [
                    detection for detection in detections
                    if not memory.is_suppressed(
                        detection["world_coordinates"]["latitude"], detection["world_coordinates"]["longitude"],
                        context.get("ais_protected_locations", {}).get(scene["id"], []),
                    )
                ]
            for det in detections:
                chip = self.crop_chip(scene["array"], det["apex_px"], self.config.detection.chip_crop_size_px)
                det["chip_tensor"] = chip
                det["chip_jpeg"] = self.encode_chip_jpeg(chip, self.config.downlink.chip_jpeg_quality)
            return detections, stats

        # Scenes are independent and OpenCV/NumPy release the GIL, so they fan out across the
        # Orin's cores. Results are gathered in scene order: output does not depend on timing.
        for scene, result in zip(usable, map_scenes(work, usable, self.config.runtime.scene_workers)):
            if result is None:
                continue
            detections, stats = result
            scene["detections"] = detections
            scene["detector_stats"] = stats
            all_detections.extend(detections)
            for key in funnel:
                funnel[key] += stats.get(key, 0)

        context["detected_vessels"] = all_detections
        context["total_vessels_detected"] = len(all_detections)
        context["detection_funnel"] = funnel
        return context

    # ------------------------------------------------------------------ CFAR

    def compute_cfar(
        self, nir: np.ndarray, sea_mask: np.ndarray, gsd: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ring-background CFAR. Background mean/sigma come from an annulus (background box
        minus guard box) computed on a 4x-decimated grid in float64: the sea background is
        smooth, so this is ~16x cheaper and avoids float32 cancellation in E[x^2]-E[x]^2.
        A second pass removes first-pass detections from the background so a ship cannot
        raise its own threshold. Returns (z_score, contrast, detection_mask).
        """
        cfg = self.config.detection
        h, w = nir.shape
        ds = 4 if min(h, w) >= 512 else 1
        small = (max(w // ds, 1), max(h // ds, 1))

        guard = 2 * meters_to_px(cfg.cfar_guard_m, gsd * ds, minimum=1) + 1
        back = max(2 * meters_to_px(cfg.cfar_background_m, gsd * ds, minimum=3) + 1, guard + 4)

        valid = sea_mask.astype(np.float32)
        level = float(np.median(nir[::8, ::8][sea_mask[::8, ::8] > 0])) if sea_mask[::8, ::8].any() else 0.0
        centred = (nir - level) * valid

        def ring_stats(valid_f: np.ndarray, centred_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            v = cv2.resize(valid_f, small, interpolation=cv2.INTER_AREA).astype(np.float64)
            s1 = cv2.resize(centred_f, small, interpolation=cv2.INTER_AREA).astype(np.float64)
            s2 = cv2.resize(centred_f * centred_f, small, interpolation=cv2.INTER_AREA).astype(np.float64)

            def ring(a: np.ndarray) -> np.ndarray:
                big = cv2.boxFilter(a, -1, (back, back), normalize=False, borderType=cv2.BORDER_REFLECT)
                inner = cv2.boxFilter(a, -1, (guard, guard), normalize=False, borderType=cv2.BORDER_REFLECT)
                return big - inner

            n = np.maximum(ring(v), 1e-6)
            mean = ring(s1) / n
            var = np.maximum(ring(s2) / n - mean * mean, 0.0)
            sigma = np.sqrt(var)
            starved = n < 0.05 * (back * back - guard * guard)  # too little sea nearby to trust
            sigma[starved] = np.inf
            mu_full = cv2.resize(mean.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) + level
            sg_full = cv2.resize(
                np.where(np.isinf(sigma), 1e6, sigma).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
            return mu_full, np.maximum(sg_full, cfg.cfar_sigma_floor)

        # Pass 0 -- censoring. With several ships in view (an anchorage) or a bright shore nearby,
        # the targets themselves inflate the ring sigma and mask each other: on real Sentinel-2 a
        # hull with 0.20 contrast scored z = 3. A classic two-pass CFAR cannot recover, because its
        # first pass finds nothing to exclude. So compact outliers against the *global* robust sea
        # statistics (median / MAD) are censored from the background first. Large bright regions
        # (glint, haze) are background structure, not targets, and stay in.
        grow = disk(meters_to_px(30.0, gsd, minimum=2))
        censored = np.zeros_like(sea_mask)
        sample = nir[::4, ::4][sea_mask[::4, ::4] > 0] if min(h, w) >= 512 else nir[sea_mask > 0]
        if sample.size > 64:
            mad = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
            outlier = ((nir - float(np.median(sample))) > 6.0 * max(mad, cfg.cfar_sigma_floor)) & (sea_mask > 0)
            n_out, lab_out, st_out, _ = cv2.connectedComponentsWithStats(outlier.astype(np.uint8), connectivity=8)
            if n_out > 1:
                compact = st_out[:, cv2.CC_STAT_AREA] <= area_to_px(cfg.max_target_area_m2, gsd)
                compact[0] = False
                censored = cv2.dilate(compact[lab_out].astype(np.uint8), grow)

        valid0 = valid * (1 - censored)
        mu, sigma = ring_stats(valid0, (nir - level) * valid0)
        contrast = (nir - mu) * valid
        z = contrast / sigma
        det = (z > cfg.cfar_k_sigma) & (contrast > cfg.cfar_min_contrast)

        if det.any():
            exclude = cv2.dilate(det.astype(np.uint8), grow) | censored
            valid2 = valid * (1 - exclude)
            mu, sigma = ring_stats(valid2, (nir - level) * valid2)
            contrast = (nir - mu) * valid
            z = contrast / sigma
            det = (z > cfg.cfar_k_sigma) & (contrast > cfg.cfar_min_contrast)

        return z, contrast, det.astype(np.uint8)

    # ------------------------------------------------------------- scene level

    def detect_scene(self, scene: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        cfg = self.config.detection
        gsd = scene["gsd_m"]
        nir = scene["array"][:, :, 3]
        sea_mask = scene["sea_mask"]
        h, w = nir.shape

        z, contrast, det_mask = self.compute_cfar(nir, sea_mask, gsd)
        scene["zmap"] = z
        scene["det_mask"] = det_mask

        link_px = meters_to_px(cfg.link_gap_m, gsd, minimum=1)
        linked = cv2.morphologyEx(det_mask, cv2.MORPH_CLOSE, disk(link_px))
        n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)

        areas = cc_stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0, dtype=np.int32)
        min_area_px = max(3, area_to_px(cfg.min_target_area_m2, gsd))
        max_area_px = area_to_px(cfg.max_target_area_m2, gsd)

        keep = np.nonzero((areas >= min_area_px) & (areas <= max_area_px))[0] + 1
        # Largest first, then raster order: deterministic regardless of labelling internals
        keep = sorted(keep.tolist(), key=lambda i: (-int(cc_stats[i, cv2.CC_STAT_AREA]),
                                                    int(cc_stats[i, cv2.CC_STAT_TOP]),
                                                    int(cc_stats[i, cv2.CC_STAT_LEFT])))
        keep = keep[: cfg.max_candidates_per_scene]

        # Shoreline keep-out. On real imagery the residual false alarms are surf, shoals, piers and
        # breakwaters hugging the land mask; a blob that reaches into this strip is "in port", not a
        # contact at sea, and is dropped before any analysis.
        if scene["land_mask"].any() and cfg.shore_exclusion_m > 0:
            k = 2 * meters_to_px(cfg.shore_exclusion_m, gsd, minimum=1) + 1
            shore = cv2.dilate(scene["land_mask"], np.ones((k, k), np.uint8))  # square kernel: separable, fast
            touching = np.unique(labels[(shore > 0) & (labels > 0)])
            if touching.size:
                near_shore = set(touching.tolist())
                keep = [i for i in keep if i not in near_shore]

        z_smooth = cv2.blur(np.clip(z, -3.0, 12.0), (3, 3))
        cloud_near = None
        if scene["cloud_mask"].any():
            cloud_near = cv2.dilate(scene["cloud_mask"], disk(meters_to_px(60.0, gsd, minimum=3)))

        # Object-level CFAR. In a whitecap field the candidates themselves are the clutter
        # population: a blob must then be an outlier in integrated contrast (ships are bigger
        # AND brighter than breaking waves) or own a long wake to survive. Integrated contrast
        # for every blob is one bincount, so the gate is known before any per-candidate work
        # and the expensive ray transform only runs on blobs that could still pass.
        sea_km2 = max(float(np.count_nonzero(sea_mask)) * gsd * gsd / 1e6, 1e-6)
        clutter_density = len(keep) / sea_km2
        rough_sea = clutter_density > cfg.clutter_density_per_km2 and len(keep) >= 8  # 8: enough for a median/MAD
        clutter_gate = 0.0
        to_analyse = keep
        if rough_sea:
            ic_all = np.bincount(labels.ravel(), weights=contrast.ravel(), minlength=n)
            logs = np.log(np.maximum(ic_all[keep], 1e-6))
            med = float(np.median(logs))
            mad = float(np.median(np.abs(logs - med)))
            clutter_gate = math.exp(med + cfg.clutter_outlier_mads * max(1.4826 * mad, 0.15))
            # a faint wake shows up in the ray transform long before it survives thresholding,
            # so anything even slightly elongated still gets the full analysis
            min_wake_px = max(5.0, 0.4 * cfg.wake_min_length_m / gsd)
            to_analyse = [
                i for i in keep
                if ic_all[i] >= clutter_gate
                or max(cc_stats[i, cv2.CC_STAT_WIDTH], cc_stats[i, cv2.CC_STAT_HEIGHT]) >= min_wake_px
            ]

        analysed: List[Dict[str, Any]] = []
        for label_id in to_analyse:
            det = self._analyse_candidate(label_id, labels, cc_stats, z, z_smooth, contrast, cloud_near, scene)
            if det is not None:
                analysed.append(det)

        detections: List[Dict[str, Any]] = []
        for det in analysed:
            if det["physics_score"] < cfg.min_physics_score:
                continue
            strong_wake = det["target_type"] == "VESSEL_UNDERWAY" and det["wake_snr"] >= 8.0 \
                and det["wake_length_m"] >= 100.0
            # breaking waves are a few unresolved pixels; a resolved, slender 30 m+ outline is not
            ship_shaped = det["hull_resolved"] and det["hull_length_m"] >= 30.0 \
                and det["hull_length_m"] >= 2.5 * det["hull_width_m"]
            if rough_sea and det["integrated_contrast"] < clutter_gate and not (strong_wake or ship_shaped):
                continue
            detections.append(det)

        detections = self._suppress_wake_debris(detections, gsd)
        detections.sort(key=lambda d: (d["apex_px"][1], d["apex_px"][0]))
        for i, det in enumerate(detections):
            det["detection_id"] = f"{scene['id']}_T{i + 1:03d}"

        stats = {
            "pixels_screened": int(h * w),
            "cfar_pixels": int(np.count_nonzero(det_mask)),
            "candidates": len(keep),
            "physics_accepted": len(detections),
            "clutter_density_per_km2": round(clutter_density, 2),
            "rough_sea_mode": bool(rough_sea),
            "clutter_gate": round(clutter_gate, 4),
        }
        return detections, stats

    @staticmethod
    def _suppress_wake_debris(detections: List[Dict[str, Any]], gsd: float) -> List[Dict[str, Any]]:
        """
        Foam patches and whitecaps riding in a ship's wake are detected as their own blobs,
        and their rays happily follow the parent wake. Anything much weaker than a vessel
        and sitting inside that vessel's (slowly widening) wake corridor is dropped.
        """
        parents = [d for d in detections if d["wake_end_px"] is not None]
        parents.sort(key=lambda d: -d["integrated_contrast"])
        dropped = set()
        for parent in parents:
            if id(parent) in dropped:
                continue
            ax, ay = parent["apex_px"]
            bx, by = parent["wake_end_px"]
            vx, vy = bx - ax, by - ay
            seg_len = math.hypot(vx, vy)
            if seg_len < 1.0:
                continue
            ux, uy = vx / seg_len, vy / seg_len
            for child in detections:
                if child is parent or id(child) in dropped:
                    continue
                if child["integrated_contrast"] > 0.5 * parent["integrated_contrast"]:
                    continue
                # debris borrows the parent's wake, so its "heading" lies along the parent's
                # track; a vessel crossing the wedge on its own course is left alone
                if child["wake_end_px"] is not None:
                    d = abs(child["heading_deg"] - parent["heading_deg"]) % 180.0
                    if min(d, 180.0 - d) > 25.0:
                        continue
                px, py = child["apex_px"][0] - ax, child["apex_px"][1] - ay
                along = px * ux + py * uy
                across = abs(-px * uy + py * ux)
                # corridor = the Kelvin wedge (19.47 deg) plus a little slack
                if 0.0 < along <= 1.25 * seg_len and across <= (20.0 / gsd) + 0.41 * along:
                    dropped.add(id(child))
        return [d for d in detections if id(d) not in dropped]

    # --------------------------------------------------------- candidate level

    def _analyse_candidate(
        self,
        label_id: int,
        labels: np.ndarray,
        cc_stats: np.ndarray,
        z: np.ndarray,
        z_smooth: np.ndarray,
        contrast: np.ndarray,
        cloud_near: Optional[np.ndarray],
        scene: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        cfg = self.config.detection
        gsd = scene["gsd_m"]
        georef = scene["georef"]

        x0, y0 = int(cc_stats[label_id, cv2.CC_STAT_LEFT]), int(cc_stats[label_id, cv2.CC_STAT_TOP])
        bw, bh = int(cc_stats[label_id, cv2.CC_STAT_WIDTH]), int(cc_stats[label_id, cv2.CC_STAT_HEIGHT])
        window = labels[y0:y0 + bh, x0:x0 + bw] == label_id
        local_contrast = np.where(window, contrast[y0:y0 + bh, x0:x0 + bw], 0.0)
        if local_contrast.max() <= 0:
            return None

        ys, xs = np.nonzero(window)
        cvals = local_contrast[ys, xs].astype(np.float64)
        blob_len, blob_wid, blob_axis = self._principal_axes(xs, ys, np.ones(len(xs)))

        # Which end is the ship? A wake decays away from the stern, so an elongated blob whose
        # ends differ in energy has the vessel at the strong end -- even when foam outshines a
        # dark hull, or a whitecap sits in the wake. A symmetric blob is just a (moored) hull.
        search = np.ones(len(xs), dtype=bool)
        occluded, far_end = False, None
        if blob_len >= 3.0 * max(blob_wid, 1.0) and blob_len * gsd >= cfg.wake_min_length_m:
            ax_x, ax_y = math.sin(math.radians(blob_axis)), -math.cos(math.radians(blob_axis))
            t = (xs - xs.mean()) * ax_x + (ys - ys.mean()) * ax_y
            t_lo, t_hi = float(t.min()), float(t.max())
            span = max(t_hi - t_lo, 1.0)
            e_lo = float(cvals[t <= t_lo + span / 3.0].mean())
            e_hi = float(cvals[t >= t_hi - span / 3.0].mean())
            ship_at_hi = e_hi >= e_lo
            # A blob contains a wake if it is slimmer than any hull (L/B > 10), or if one end is
            # several times dimmer than the other (deck detail only varies a hull by tens of %)
            asymmetric = blob_len >= 10.0 * max(blob_wid, 1.0) or                 max(e_lo, e_hi) > 3.0 * max(min(e_lo, e_hi), 1e-9)

            if cloud_near is not None:
                touch = cloud_near[ys + y0, xs + x0] > 0
                lo_touch = bool(touch[t <= t_lo + 0.15 * span].any())
                hi_touch = bool(touch[t >= t_hi - 0.15 * span].any())
                if lo_touch != hi_touch:
                    # a streak that runs into a cloud edge: the ship is under the cloud unless
                    # the open-water end is overwhelmingly the stronger one
                    cloud_is_hi = hi_touch
                    e_cloud, e_open = (e_hi, e_lo) if cloud_is_hi else (e_lo, e_hi)
                    if e_cloud >= 0.4 * e_open:
                        occluded, ship_at_hi, asymmetric = True, cloud_is_hi, True

            if asymmetric:
                search = (t >= t_hi - 0.3 * span) if ship_at_hi else (t <= t_lo + 0.3 * span)
                k_far = int(np.argmin(t)) if ship_at_hi else int(np.argmax(t))
                far_end = (float(xs[k_far] + x0), float(ys[k_far] + y0))
                if occluded:
                    k_near = int(np.argmax(t)) if ship_at_hi else int(np.argmin(t))

        # Hull = bright core around the peak of the ship-end region
        k_peak = int(np.flatnonzero(search)[np.argmax(cvals[search])])
        py, px = int(ys[k_peak]), int(xs[k_peak])
        region = np.zeros(window.shape, dtype=bool)
        region[ys[search], xs[search]] = True
        core = (region & (local_contrast >= 0.4 * local_contrast[py, px])).astype(np.uint8)
        _, core_labels = cv2.connectedComponents(core, connectivity=8)
        hull = core_labels == core_labels[py, px]
        hys, hxs = np.nonzero(hull)
        hw = local_contrast[hys, hxs].astype(np.float64)
        cx = float((hxs * hw).sum() / hw.sum()) + x0
        cy = float((hys * hw).sum() / hw.sum()) + y0
        # extent from the unweighted footprint: deck brightness varies, the hull outline does not
        hull_len, hull_wid, hull_axis = self._principal_axes(hxs, hys, np.ones(len(hxs)))
        peak_z = float(z[y0 + py, x0 + px])
        integrated = float(cvals.sum())

        # Small fast craft: the hull is a couple of mixed pixels and the brightest thing in the
        # blob is foam well astern. If the "hull" core is itself wake-thin, trust geometry
        # instead: the vessel is at the tip of the ship end, and its size is unresolved.
        hull_resolved = True
        foam_core = False
        if far_end is not None and not occluded:
            k_tip = int(np.argmax(t)) if ship_at_hi else int(np.argmin(t))
            tip_dist = math.hypot(cx - (xs[k_tip] + x0), cy - (ys[k_tip] + y0))
            # a real hull sits at the head of its blob; a bright core further back than its own
            # length is foam (a widening wake gathers more foam per pixel some way astern)
            foam_core = hull_len >= 10.0 * max(hull_wid, 1.0) or tip_dist > max(hull_len, 6.0)
        if foam_core:
            inward = -1.0 if ship_at_hi else 1.0
            cx = float(xs[k_tip] + x0) + inward * ax_x * 1.5
            cy = float(ys[k_tip] + y0) + inward * ax_y * 1.5
            hull_resolved = False
            hull_len, hull_wid = 3.0 * blob_wid, blob_wid
            hull = np.zeros_like(hull)
            hull[int(np.clip(round(cy) - y0, 0, bh - 1)), int(np.clip(round(cx) - x0, 0, bw - 1))] = True

        if occluded:
            cx, cy = float(xs[k_near] + x0), float(ys[k_near] + y0)
            bearing = math.degrees(math.atan2(far_end[0] - cx, -(far_end[1] - cy))) % 360.0
            wake = {"found": True, "bearing_deg": bearing, "length_px": float(blob_len), "snr": 0.0,
                    "kelvin_found": False, "kelvin_half_angle_deg": None, "speed_knots": None}
        else:
            # Hull footprint (+2 px of optical bleed) is blanked so the ship cannot be its own wake
            pad = 3
            hull_pad = cv2.dilate(
                cv2.copyMakeBorder(hull.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT), disk(2)
            )
            wake = self._ray_transform(z_smooth, contrast, cx, cy, gsd, hull_pad, (x0 - pad, y0 - pad))

        if wake["found"]:
            heading_img = (wake["bearing_deg"] + 180.0) % 360.0
            heading_ambiguous = False
        else:
            heading_img = hull_axis if hull_len >= 1.5 * hull_wid else blob_axis
            heading_ambiguous = True
        heading_true = georef.image_bearing_to_true(
            math.sin(math.radians(heading_img)), -math.cos(math.radians(heading_img))
        )

        hull_len_m, hull_wid_m = hull_len * gsd, hull_wid * gsd
        physics = self._physics_score(peak_z, hull_len_m, hull_wid_m, wake, occluded)

        # Band-parallax test. A pushbroom imager records its bands a fraction of a second apart, so
        # anything much faster than a ship (aircraft, mostly) lands in a different place in each band
        # and shows up as separated red / green / blue dots. A 30-knot vessel moves ~1 px in that time.
        parallax_px = self._band_parallax(scene["array"], cx, cy) if hull_len <= 8.0 and not wake["found"] else 0.0
        airborne = parallax_px >= cfg.parallax_reject_px
        if airborne:
            physics = min(physics, 0.2)

        if airborne:
            target_type = "AIRBORNE_OR_FAST_MOVER"
        elif occluded:
            target_type = "WAKE_ONLY_CLOUD_OCCLUDED"
        elif wake["found"]:
            target_type = "VESSEL_UNDERWAY"
        else:
            target_type = "VESSEL_STATIONARY_OR_SLOW"

        lon, lat = georef.pixel_to_lonlat(cx, cy)
        wake_end = None
        if wake["found"]:
            rad = math.radians(wake["bearing_deg"])
            wake_end = [round(cx + math.sin(rad) * wake["length_px"], 1),
                        round(cy - math.cos(rad) * wake["length_px"], 1)]

        return {
            "detection_id": "",
            "scene_id": scene["id"],
            "target_type": target_type,
            "apex_px": (int(round(cx)), int(round(cy))),
            "bbox_px": [x0, y0, bw, bh],
            "world_coordinates": {"latitude": round(lat, 6), "longitude": round(lon, 6)},
            "heading_deg": round(heading_true, 1),
            "heading_image_deg": round(heading_img % 360.0, 1),
            "heading_ambiguous_180": heading_ambiguous,
            "hull_length_m": round(hull_len_m, 1),
            "hull_width_m": round(hull_wid_m, 1),
            "hull_resolved": hull_resolved,
            "size_class": self._size_class(hull_len_m),
            "wake_length_m": round(wake["length_px"] * gsd, 1) if wake["found"] else 0.0,
            "wake_snr": round(wake["snr"], 2),
            "wake_end_px": wake_end,
            "kelvin_arms_detected": wake["kelvin_found"],
            "kelvin_half_angle_deg": wake["kelvin_half_angle_deg"],
            "estimated_speed_knots": wake["speed_knots"],
            "speed_method": "KELVIN_TRANSVERSE_WAVELENGTH" if wake["speed_knots"] is not None else None,
            "band_parallax_px": round(parallax_px, 2),
            "peak_z": round(peak_z, 2),
            "integrated_contrast": round(integrated, 4),
            "physics_score": round(physics, 3),
            "verifier_prob": None,
            "confidence": round(physics, 3),
            "ais_status": "PENDING_CORRELATION",
        }

    @staticmethod
    def _band_parallax(reflectance: np.ndarray, cx: float, cy: float, half: int = 10) -> float:
        """Largest distance (px) between the per-band bright centroids around a compact target."""
        h, w = reflectance.shape[:2]
        x0, x1 = max(int(cx) - half, 0), min(int(cx) + half + 1, w)
        y0, y1 = max(int(cy) - half, 0), min(int(cy) + half + 1, h)
        win = reflectance[y0:y1, x0:x1]
        if win.shape[0] < 5 or win.shape[1] < 5:
            return 0.0
        yy, xx = np.mgrid[0:win.shape[0], 0:win.shape[1]]
        centroids = []
        for b in range(win.shape[2]):
            band = win[:, :, b]
            excess = band - np.median(band)
            peak = float(excess.max())
            if peak < 0.02:
                continue
            wgt = np.clip(excess - 0.5 * peak, 0.0, None)  # only the bright core of this band
            centroids.append((float((xx * wgt).sum() / wgt.sum()), float((yy * wgt).sum() / wgt.sum())))
        if len(centroids) < 3:
            return 0.0
        return max(math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(centroids) for b in centroids[i + 1:])

    @staticmethod
    def _principal_axes(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
        """Length, width (px, uniform-bar equivalent) and image bearing of the major axis."""
        if len(xs) < 2:
            return 1.0, 1.0, 0.0
        w = weights / weights.sum()
        mx, my = float((xs * w).sum()), float((ys * w).sum())
        dx, dy = xs - mx, ys - my
        cxx, cyy, cxy = float((w * dx * dx).sum()), float((w * dy * dy).sum()), float((w * dx * dy).sum())
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
        l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 0.0)
        # +1/12: a single pixel has the variance of a unit square, so a 1-px-wide bar reads 1 px
        length = math.sqrt(12.0 * (l1 + 1.0 / 12.0))
        width = math.sqrt(12.0 * (l2 + 1.0 / 12.0))
        angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)  # direction of major axis in (x, y)
        bearing = math.degrees(math.atan2(math.cos(angle), -math.sin(angle))) % 360.0
        return length, width, bearing

    def _ray_transform(
        self, z_smooth: np.ndarray, contrast: np.ndarray, cx: float, cy: float, gsd: float,
        hull_mask: np.ndarray, hull_origin: Tuple[int, int],
    ) -> Dict[str, Any]:
        cfg = self.config.detection
        h, w = z_smooth.shape
        radius = meters_to_px(cfg.wake_search_radius_m, gsd, minimum=16)
        table = _RayTable.get(radius)
        result: Dict[str, Any] = {
            "found": False, "bearing_deg": 0.0, "length_px": 0.0, "snr": 0.0,
            "kelvin_found": False, "kelvin_half_angle_deg": None, "speed_knots": None,
        }

        ix, iy = int(round(cx)), int(round(cy))
        xs = table.dx + ix
        ys = table.dy + iy
        inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        prof = np.where(inb, z_smooth[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)], 0.0)

        # Each ray starts where it leaves the hull footprint
        mh, mw = hull_mask.shape
        lx, ly = xs - hull_origin[0], ys - hull_origin[1]
        in_box = (lx >= 0) & (lx < mw) & (ly >= 0) & (ly < mh)
        on_hull = in_box & (hull_mask[np.clip(ly, 0, mh - 1), np.clip(lx, 0, mw - 1)] > 0)
        n_r = prof.shape[1]
        start = np.where(on_hull.any(axis=1), n_r - np.argmax(on_hull[:, ::-1], axis=1), 0)
        start = np.minimum(start, n_r - 1)
        prof[on_hull] = 0.0

        # A wake is contiguous from the stern: walk outward until fewer than half of the next
        # 9 samples are elevated. Occupancy, not mean -- one whitecap at z=50 must not be able
        # to carry a ray across open water.
        win = 9
        csum = np.cumsum(np.pad(prof, ((0, 0), (1, 0))), axis=1)
        occ = np.cumsum(np.pad((prof > 1.2).astype(np.float64), ((0, 0), (1, 0))), axis=1)
        idx = np.arange(n_r)
        hi = np.minimum(idx + win, n_r)
        running = (occ[:, hi] - occ[:, idx]) / (hi - idx)
        dead = (running < 0.5) & (idx[None, :] >= start[:, None])
        end = np.where(dead.any(axis=1), dead.argmax(axis=1), n_r)
        length = end - start
        rows = np.arange(_N_ANGLES)
        score = csum[rows, end] - csum[rows, start]

        best = int(np.argmax(score))
        best_len, best_start = int(length[best]), int(start[best])
        if best_len < 2:
            return result

        sep = np.minimum((rows - best) % 360, (best - rows) % 360)
        others = score[sep > 30]
        med = float(np.median(others))
        mad = float(np.median(np.abs(others - med)))
        # noise floor: a sum of best_len smoothed unit-variance samples, never less than that
        snr = (float(score[best]) - med) / max(1.4826 * mad, 0.6 * math.sqrt(best_len))
        mean_z = float(score[best]) / best_len

        result.update(bearing_deg=float(best), length_px=float(end[best]), snr=snr)
        min_len_px = max(cfg.wake_min_length_m / gsd, 6.0)
        if best_len < min_len_px or snr < cfg.wake_min_snr or mean_z < 1.5:
            return result
        result["found"] = True

        # Kelvin cusp arms: paired ridges either side of the track. The wedge opens from the
        # BOW, so the arm rays are cast from a point half a hull length ahead of the centroid;
        # seen from the centroid the arms are curves, not rays, and the test would never fire.
        rad = math.radians(best)
        bx = int(round(cx - math.sin(rad) * best_start))
        by = int(round(cy + math.cos(rad) * best_start))
        r_lo = min(2 * best_start + 4, n_r - 2)
        r_hi = min(r_lo + max(best_len, 12), n_r)
        alphas = np.arange(10, 31)
        arm_rows = np.concatenate([(best + alphas) % 360, (best - alphas) % 360])
        axs = np.clip(table.dx[arm_rows, r_lo:r_hi] + bx, 0, w - 1)
        ays = np.clip(table.dy[arm_rows, r_lo:r_hi] + by, 0, h - 1)
        arm_mean = z_smooth[ays, axs].mean(axis=1)
        paired = np.minimum(arm_mean[:len(alphas)], arm_mean[len(alphas):])
        k = int(np.argmax(paired))
        if paired[k] >= 1.0 and paired[k] - float(np.median(paired)) >= 0.5:
            half = float(alphas[k])
            if abs(half - cfg.kelvin_half_angle_deg) <= cfg.kelvin_tolerance_deg:
                result["kelvin_found"] = True
                result["kelvin_half_angle_deg"] = half

        result["speed_knots"] = self._speed_from_transverse_waves(
            contrast, table, ix, iy, best, best_start, best_len, gsd
        )
        return result

    @staticmethod
    def _speed_from_transverse_waves(
        contrast: np.ndarray, table: _RayTable, ix: int, iy: int, bearing: int, r0: int, length: int, gsd: float
    ) -> Optional[float]:
        """
        Transverse Kelvin waves travel with the ship, so their wavelength encodes speed:
        lambda = 2*pi*V^2/g. Sampled just off the foam centreline, where they are not swamped.

        Ocean swell is periodic too. A ray astern and the ray dead ahead lie on the same line,
        so swell projects to the same wavelength on both -- but only the astern ray crosses
        the wake. A spectral line is accepted only if it is absent ahead of the ship.
        Returns None unless that holds; most small or slow vessels have no measurable line.
        """
        n = min(length, table.radius_px - r0)
        if n < 32:
            return None
        h, w = contrast.shape

        radii = np.arange(r0, r0 + n, dtype=np.float64)

        def spectrum(base_bearing: int) -> np.ndarray:
            # bilinear sampling: nearest-pixel stepping along a diagonal aliases a short
            # wave into a long one, which reads as a ship going twice as fast
            profile = np.zeros(n, dtype=np.float64)
            for off in (-_SPEED_RAY_OFFSET_DEG, _SPEED_RAY_OFFSET_DEG):
                a = math.radians((base_bearing + off) % 360)
                fx = np.clip(ix + np.sin(a) * radii, 0, w - 1.001)
                fy = np.clip(iy - np.cos(a) * radii, 0, h - 1.001)
                x0, y0 = fx.astype(np.intp), fy.astype(np.intp)
                tx, ty = fx - x0, fy - y0
                profile += (contrast[y0, x0] * (1 - tx) * (1 - ty) + contrast[y0, x0 + 1] * tx * (1 - ty)
                            + contrast[y0 + 1, x0] * (1 - tx) * ty + contrast[y0 + 1, x0 + 1] * tx * ty)
            k = max(n // 4, 5) | 1
            trend = np.convolve(np.pad(profile, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
            return np.abs(np.fft.rfft((profile - trend) * np.hanning(n))) ** 2

        # the reference ray must itself be open water, otherwise swell cannot be ruled out
        a_ref = (bearing + 180) % 360
        rx = table.dx[a_ref, r0:r0 + n] + ix
        ry = table.dy[a_ref, r0:r0 + n] + iy
        ok = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
        if ok.mean() < 0.9 or (contrast[np.clip(ry, 0, h - 1), np.clip(rx, 0, w - 1)] != 0).mean() < 0.9:
            return None

        spec = spectrum(bearing)
        ahead = spectrum(a_ref)
        freqs = np.arange(len(spec))
        wavelength_px = n / np.maximum(freqs, 1)
        # 3 px is the sampling limit; 110 m is a 26-knot ship, beyond which nothing is plausible
        # 4 px: sampling limit. n/8: a line needs >= 8 cycles on the ray to be credible -- the
        # foam's exponential decay leaves low-frequency residue that otherwise reads as a fast ship.
        # 110 m is a 26-knot vessel, beyond which nothing is plausible.
        band = (freqs > 0) & (wavelength_px >= 4.0) & (wavelength_px <= min(n / 8.0, 110.0 / gsd))
        if band.sum() < 4:
            return None
        peak = int(np.argmax(np.where(band, spec, 0.0)))
        floor = float(np.median(spec[band])) + 1e-18
        lo, hi = max(peak - 1, 0), min(peak + 2, len(spec))
        if spec[peak] / floor < 12.0 or spec[peak] < 6.0 * float(ahead[lo:hi].max()):
            return None
        rivals = np.where(band, spec, 0.0)
        rivals[max(peak - 2, 0):peak + 3] = 0.0
        if spec[peak] < 2.5 * float(rivals.max()):  # a line, not a hump of detrending residue
            return None
        # a genuine fundamental has no stronger content at its own harmonics
        for m in (2, 3, 4):
            k = peak * m
            if k + 1 < len(spec) and float(spec[k - 1:k + 2].max()) > 0.5 * spec[peak]:
                return None
        shift = 0.0
        if 0 < peak < len(spec) - 1:
            a, b, c = spec[peak - 1], spec[peak], spec[peak + 1]
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-18 else 0.0
            shift = min(max(shift, -0.5), 0.5)  # a true peak never interpolates past its neighbours
        lam_m = (n / (peak + shift)) * gsd * math.cos(math.radians(_SPEED_RAY_OFFSET_DEG))
        knots = speed_from_kelvin_wavelength(lam_m)
        return round(knots, 1) if 3.0 <= knots <= 45.0 else None

    @staticmethod
    def _physics_score(peak_z: float, hull_len_m: float, hull_wid_m: float, wake: Dict[str, Any], occluded: bool) -> float:
        """
        Evidence fusion, 0..1. Contrast says "something solid", shape says "ship-like",
        the wake says "and it is moving". Things no vessel can be (wider than 65 m, or a
        large blob that is round) are capped below the acceptance threshold outright.
        """
        if occluded:
            return 0.55
        s_contrast = min(max((peak_z - 5.0) / 15.0, 0.0), 1.0)
        elong = hull_len_m / max(hull_wid_m, 1e-3)
        if hull_len_m < 30.0:
            s_shape = 0.5  # resolution-limited: a few mixed pixels carry no shape information
        else:
            s_shape = min(max((elong - 1.2) / 1.8, 0.0), 1.0)
        s_wake = min(wake["snr"] / 10.0, 1.0) if wake["found"] else 0.0
        s_kelvin = 1.0 if wake["kelvin_found"] else 0.0
        score = 0.35 * s_contrast + 0.25 * s_shape + 0.30 * s_wake + 0.10 * s_kelvin

        impossible = hull_wid_m > 65.0 or hull_len_m > 460.0 or hull_len_m < 6.0
        round_and_big = hull_len_m >= 30.0 and elong < 1.5 and not wake["found"]
        return min(score, 0.30) if (impossible or round_and_big) else score

    @staticmethod
    def _size_class(length_m: float) -> str:
        if length_m < 25:
            return "SMALL_CRAFT"
        if length_m < 100:
            return "MEDIUM_VESSEL"
        if length_m < 250:
            return "LARGE_VESSEL"
        return "VERY_LARGE_VESSEL"

    # ------------------------------------------------------------------ chips

    @staticmethod
    def crop_chip(reflectance: np.ndarray, apex_px: Tuple[int, int], size: int) -> np.ndarray:
        """size x size x 4 float32 chip centred on the target; scene edges are replicated."""
        x, y = apex_px
        half = size // 2
        h, w = reflectance.shape[:2]
        x1, y1 = x - half, y - half
        x2, y2 = x1 + size, y1 + size
        crop = reflectance[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
        pad = (max(-y1, 0), max(y2 - h, 0), max(-x1, 0), max(x2 - w, 0))
        if any(pad):
            crop = cv2.copyMakeBorder(crop, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(crop, dtype=np.float32)

    @staticmethod
    def encode_chip_jpeg(chip: np.ndarray, quality: int) -> bytes:
        """Fixed (scene-independent) stretch so repeated runs give byte-identical chips."""
        rgb = np.clip(chip[:, :, :3] / 0.35, 0.0, 1.0) ** (1.0 / 2.2)
        bgr = (rgb[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else b""
