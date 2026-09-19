"""
Image Quality and Usability Screener (Track 4 + Stage 1 pre-filter).

Builds the three masks every later stage depends on -- no-data, cloud, land -- from
band physics rather than learned weights, and decides whether a scene is worth any
further compute at all.

  water : absorbs NIR almost completely  -> NDWI = (G - NIR)/(G + NIR) > 0, NIR low
  cloud : bright AND spectrally flat     -> high mean reflectance, low (max-min)/mean
  ship / foam share the cloud signature, so size is the discriminator: only blobs
  bigger than any vessel are allowed into the cloud and land masks.
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple
from applet.core.base import BasePipeline
from applet.utils.image_io import split_bands


def meters_to_px(meters: float, gsd_m: float, minimum: int = 1) -> int:
    return max(minimum, int(round(meters / gsd_m)))


def area_to_px(area_m2: float, gsd_m: float) -> int:
    return max(1, int(round(area_m2 / (gsd_m * gsd_m))))


def keep_large_components(mask_u8: np.ndarray, min_area_px: int) -> np.ndarray:
    """Returns a uint8 {0,1} mask of connected components with area >= min_area_px."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area_px
    keep[0] = False
    return keep[labels].astype(np.uint8)


def map_scenes(fn, scenes, workers: int):
    """Ordered map over scenes on a small thread pool (1 worker = plain loop)."""
    if workers <= 1 or len(scenes) <= 1:
        return [fn(s) for s in scenes]
    with ThreadPoolExecutor(max_workers=min(workers, len(scenes))) as pool:
        return list(pool.map(fn, scenes))


def disk(radius_px: int) -> np.ndarray:
    size = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class ImageQualityScreener(BasePipeline):
    """
    Fast pre-screener running vectorised OpenCV checks to assess image usability.
    Drops bad/cloudy scenes early so no detector or model time is spent on them.
    """

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        def work(scene: Dict[str, Any]) -> Dict[str, Any]:
            result = dict(scene)
            try:
                result.update(self.screen_scene(scene))
            except Exception as e:  # never let one bad scene take down the pass
                result["quality_metrics"] = {
                    "is_usable": False,
                    "rejection_reasons": [f"SCREENER_FAULT ({type(e).__name__})"],
                }
            return result

        screened = map_scenes(work, context.get("valid_scenes", []), self.config.runtime.scene_workers)

        context["screened_scenes"] = screened
        context["usable_scenes_count"] = sum(1 for s in screened if s["quality_metrics"]["is_usable"])
        return context

    def screen_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.screening
        gsd = scene["gsd_m"]
        bands = split_bands(scene["array"])
        nodata = scene["nodata_mask"]
        missing = scene["status"].get("missing_bands", [])

        valid_data_pct = 100.0 * (1.0 - float(nodata.mean()))

        cloud_mask = self.build_cloud_mask(bands, gsd)
        land_mask, ndwi = self.build_land_mask(bands, cloud_mask, nodata, gsd, nir_missing="nir" in missing)

        sea_mask = ((cloud_mask == 0) & (land_mask == 0) & ~nodata).astype(np.uint8)

        n_valid = max(int((~nodata).sum()), 1)
        cloud_pct = 100.0 * float(np.count_nonzero(cloud_mask & ~nodata)) / n_valid
        water_pct = 100.0 * float(np.count_nonzero(sea_mask)) / n_valid

        blur_score = float(cv2.Laplacian(bands["green"], cv2.CV_32F).var())
        clutter_sigma, sea_nir_median = self.sea_state(bands["nir"], sea_mask, gsd)

        reasons, warnings = [], list(scene["status"].get("warnings", []))
        if valid_data_pct < cfg.min_valid_data_pct:
            reasons.append(f"INSUFFICIENT_VALID_DATA ({valid_data_pct:.1f}% < {cfg.min_valid_data_pct}%)")
        if cloud_pct > cfg.cloud_cover_max_pct:
            reasons.append(f"CLOUD_COVER_EXCEEDED ({cloud_pct:.1f}% > {cfg.cloud_cover_max_pct}%)")
        if blur_score < cfg.blur_min_laplacian_var:
            reasons.append(f"NO_HIGH_FREQUENCY_CONTENT (Laplacian var {blur_score:.2e})")
        if water_pct < cfg.min_water_pct and not reasons:
            reasons.append(f"NO_OPEN_WATER ({water_pct:.1f}% < {cfg.min_water_pct}%)")
        if missing:
            warnings.append("MISSING_BANDS_" + "_".join(b.upper() for b in missing))
        if sea_nir_median > 0.06:
            warnings.append("SUNGLINT_OR_HAZE_LIKELY")

        return {
            "quality_metrics": {
                "cloud_cover_pct": round(cloud_pct, 2),
                "water_pct": round(water_pct, 2),
                "valid_data_pct": round(valid_data_pct, 2),
                "blur_score": float(f"{blur_score:.3e}"),
                "sea_clutter_sigma": round(clutter_sigma, 5),
                "sea_nir_median": round(sea_nir_median, 4),
                "is_usable": not reasons,
                "rejection_reasons": reasons,
                "warnings": warnings,
            },
            "cloud_mask": cloud_mask,
            "land_mask": land_mask,
            "sea_mask": sea_mask,
            "ndwi": ndwi,
        }

    def build_cloud_mask(self, bands: Dict[str, np.ndarray], gsd: float) -> np.ndarray:
        cfg = self.config.screening
        r, g, b, nir = bands["red"], bands["green"], bands["blue"], bands["nir"]

        brightness = (r + g + b + nir) * 0.25
        spread = cv2.max(cv2.max(r, g), cv2.max(b, nir)) - cv2.min(cv2.min(r, g), cv2.min(b, nir))
        whiteness = spread / (brightness + 1e-4)

        core = ((brightness > cfg.cloud_brightness_min) & (whiteness < cfg.cloud_whiteness_max)).astype(np.uint8)
        core = keep_large_components(core, area_to_px(cfg.cloud_min_area_m2, gsd))
        if not core.any():
            return core

        # Hysteresis: thin cloud fringe attached to a confirmed core
        fringe = ((brightness > 0.5 * cfg.cloud_brightness_min) & (whiteness < 1.5 * cfg.cloud_whiteness_max))
        buffer_px = meters_to_px(cfg.cloud_buffer_m, gsd, minimum=2)
        reach = cv2.dilate(core, disk(3 * buffer_px))
        cloud = core | (fringe.astype(np.uint8) & reach)
        return cv2.dilate(cloud, disk(buffer_px))

    def build_land_mask(
        self,
        bands: Dict[str, np.ndarray],
        cloud_mask: np.ndarray,
        nodata: np.ndarray,
        gsd: float,
        nir_missing: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config.screening
        g, nir = bands["green"], bands["nir"]
        ndwi = (g - nir) / (g + nir + 1e-4)

        if nir_missing:
            # Without NIR the water test is meaningless; fall back to "dark = water"
            water = (bands["red"] < cfg.nir_water_max)
        else:
            # NDWI is a ratio, so it is undefined where both bands are ~0: atmospherically corrected
            # clear deep water reads 0-1 DN in green AND NIR. Anything that dark is water, not land.
            water = (nir < cfg.nir_water_max) & ((ndwi > cfg.ndwi_water_min) | (nir < cfg.nir_deep_water_max))

        not_water = (~water & (cloud_mask == 0) & ~nodata).astype(np.uint8)
        land = keep_large_components(not_water, area_to_px(cfg.land_min_area_m2, gsd))
        if land.any():
            land = cv2.dilate(land, disk(meters_to_px(cfg.land_buffer_m, gsd, minimum=2)))
        return land, ndwi

    @staticmethod
    def sea_state(nir: np.ndarray, sea_mask: np.ndarray, gsd: float) -> Tuple[float, float]:
        """Robust (MAD) high-pass NIR sigma over open water, and the median NIR level."""
        if not sea_mask.any():
            return 0.0, 0.0
        step = max(1, int(min(nir.shape) // 1024))  # subsample: statistics, not imaging
        sub = nir[::step, ::step]
        sel = sea_mask[::step, ::step] > 0
        if sel.sum() < 64:
            return 0.0, 0.0
        k = 2 * meters_to_px(100.0, gsd * step, minimum=2) + 1
        highpass = sub - cv2.blur(sub, (k, k))
        vals = highpass[sel]
        mad = float(np.median(np.abs(vals - np.median(vals))))
        return 1.4826 * mad, float(np.median(sub[sel]))
