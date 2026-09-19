"""
Physics detector: measured against rendered truth, not against its own assumptions.
"""

import math
import numpy as np

from applet.config import AppletConfig
from applet.pipelines.quality_screener import ImageQualityScreener
from applet.pipelines.vessel_detector import VesselDetector
from applet.utils.geo import SceneGeoreference
from simulation.scene_synth import SceneSpec, Vessel, render_scene


def _run(spec):
    cfg = AppletConfig()
    refl, truth = render_scene(spec)
    scene = {
        "id": "T", "array": refl, "nodata_mask": np.zeros(refl.shape[:2], bool), "gsd_m": spec.gsd_m,
        "status": {"missing_bands": [], "warnings": []},
        "georef": SceneGeoreference({"center_lat": 44.0, "center_lon": -63.0}, spec.width, spec.height, spec.gsd_m),
    }
    scene.update(ImageQualityScreener(cfg).screen_scene(scene))
    dets, stats = VesselDetector(cfg).detect_scene(scene)
    return scene, dets, stats, truth


def _nearest(dets, x, y):
    return min(dets, key=lambda d: math.hypot(d["apex_px"][0] - x, d["apex_px"][1] - y))


def test_empty_ocean_yields_nothing():
    """Otsu-style thresholds always find 'something'; a CFAR detector must not."""
    _, dets, stats, _ = _run(SceneSpec(512, 512, 4.75, seed=1, wind=0.3))
    assert dets == []
    assert stats["cfar_pixels"] < 20


def test_moving_ship_position_heading_and_speed():
    ship = Vessel(260, 250, heading_deg=315, speed_knots=18, length_m=190, beam_m=30, palette=4)
    _, dets, _, _ = _run(SceneSpec(640, 640, 4.75, seed=2, wind=0.25, vessels=[ship]))
    assert len(dets) == 1
    d = dets[0]
    assert math.hypot(d["apex_px"][0] - 260, d["apex_px"][1] - 250) < 190 / 4.75 / 2
    assert d["target_type"] == "VESSEL_UNDERWAY" and not d["heading_ambiguous_180"]
    assert abs(d["heading_deg"] - 315) <= 3
    assert 0.6 * 190 <= d["hull_length_m"] <= 1.3 * 190  # the sub-pixel bow taper reads short
    # transverse-wave speed, when reported, must be close; None is acceptable, a wrong number is not
    assert d["estimated_speed_knots"] is None or abs(d["estimated_speed_knots"] - 18) <= 3


def test_moored_ship_is_static_with_ambiguous_heading():
    ship = Vessel(256, 256, heading_deg=140, speed_knots=0, length_m=240, beam_m=40, palette=1)
    _, dets, _, _ = _run(SceneSpec(512, 512, 4.75, seed=3, wind=0.2, vessels=[ship]))
    assert len(dets) == 1
    d = dets[0]
    assert d["target_type"] == "VESSEL_STATIONARY_OR_SLOW" and d["heading_ambiguous_180"]
    assert min(abs(d["heading_deg"] - 140) % 180, 180 - abs(d["heading_deg"] - 140) % 180) <= 5
    assert d["estimated_speed_knots"] is None


def test_small_fast_boat_is_located_at_the_head_of_its_wake():
    boat = Vessel(300, 300, heading_deg=20, speed_knots=24, length_m=16, beam_m=4.5, palette=0)
    _, dets, _, _ = _run(SceneSpec(640, 640, 4.75, seed=4, wind=0.25, vessels=[boat]))
    assert dets
    d = _nearest(dets, 300, 300)
    assert math.hypot(d["apex_px"][0] - 300, d["apex_px"][1] - 300) < 60 / 4.75
    assert abs(d["heading_deg"] - 20) <= 5


def test_gale_whitecaps_do_not_bury_the_ship():
    ship = Vessel(300, 300, heading_deg=250, speed_knots=13, length_m=140, beam_m=22, palette=1)
    _, dets, stats, _ = _run(SceneSpec(640, 640, 4.75, seed=5, wind=0.95, vessels=[ship]))
    assert stats["rough_sea_mode"] and stats["candidates"] > 50
    assert len(dets) <= 4
    d = _nearest(dets, 300, 300)
    assert math.hypot(d["apex_px"][0] - 300, d["apex_px"][1] - 300) < 140 / 4.75


def test_land_and_cloud_are_masked_not_detected():
    spec = SceneSpec(640, 640, 4.75, seed=6, wind=0.3, coast=True, cloud_cover=0.25)
    scene, dets, _, truth = _run(spec)
    assert truth["land_fraction"] > 0.05 and scene["land_mask"].mean() > 0.05
    assert scene["cloud_mask"].mean() > 0.1
    for d in dets:
        x, y = d["apex_px"]
        assert scene["land_mask"][y, x] == 0 and scene["cloud_mask"][y, x] == 0


def test_works_at_sentinel2_resolution():
    ship = Vessel(200, 200, heading_deg=90, speed_knots=16, length_m=220, beam_m=32, palette=2)
    _, dets, _, _ = _run(SceneSpec(400, 400, 10.0, seed=7, wind=0.3, vessels=[ship]))
    assert dets
    d = _nearest(dets, 200, 200)
    assert math.hypot(d["apex_px"][0] - 200, d["apex_px"][1] - 200) < 15
    assert abs(d["heading_deg"] - 90) <= 5


def test_zero_reflectance_deep_water_is_not_land():
    """Regression from real Sentinel-2 L2A: clear deep water is ~0 DN in green and NIR, NDWI is 0/0."""
    refl = np.zeros((400, 400, 4), dtype=np.float32)
    refl[:, :, 2] = 0.004  # a trace of blue, nothing else
    refl[200:203, 190:212, :] = 0.25  # a 220 m ship
    scene = {
        "id": "T", "array": refl, "nodata_mask": np.zeros((400, 400), bool), "gsd_m": 10.0,
        "status": {"missing_bands": [], "warnings": []},
        "georef": SceneGeoreference({"center_lat": 33.6, "center_lon": -118.2}, 400, 400, 10.0),
    }
    cfg = AppletConfig()
    scene.update(ImageQualityScreener(cfg).screen_scene(scene))
    assert scene["land_mask"].mean() < 0.01 and scene["quality_metrics"]["water_pct"] > 95
    dets, _ = VesselDetector(cfg).detect_scene(scene)
    assert len(dets) == 1 and abs(dets[0]["apex_px"][0] - 200) < 12
