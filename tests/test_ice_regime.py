"""
Sea ice: the fourth surface class, and the detector's behaviour inside it.

These are built from synthetic reflectance rather than the real Arctic scenes, because a test
has to assert what SHOULD happen -- including for a vessel in pack ice, which no imagery we
hold contains. The radiometry used here is the radiometry measured on four real Sentinel-2
Arctic scenes (ice: green ~0.36, NIR ~0.29, so NDWI ~ +0.11; cloud: flat at ~0.27 across all
four bands; open water: NIR ~0.01).
"""

import numpy as np

from applet.config import AppletConfig
from applet.pipelines.quality_screener import ImageQualityScreener
from applet.pipelines.vessel_detector import VesselDetector
from applet.utils.geo import SceneGeoreference

GSD = 10.0


def _blank(h=512, w=512):
    """Open water: near-zero NIR, a little green. Bands are ordered red, green, blue, nir."""
    a = np.zeros((h, w, 4), np.float32)
    a[..., 0], a[..., 1], a[..., 2], a[..., 3] = 0.015, 0.055, 0.070, 0.012
    return a


def _paint_ice(a, sl_y, sl_x):
    a[sl_y, sl_x, 0] = 0.34
    a[sl_y, sl_x, 1] = 0.36
    a[sl_y, sl_x, 2] = 0.38
    a[sl_y, sl_x, 3] = 0.29     # absorbs toward 865 nm -> NDWI stays positive


def _paint_cloud(a, sl_y, sl_x):
    a[sl_y, sl_x, :] = 0.27     # spectrally flat -> NDWI ~ 0


def _screen(a):
    cfg = AppletConfig()
    scene = {
        "id": "ICE", "array": a, "nodata_mask": np.zeros(a.shape[:2], bool), "gsd_m": GSD,
        "status": {"missing_bands": [], "warnings": []},
        "georef": SceneGeoreference({"center_lat": 71.0, "center_lon": -156.0},
                                    a.shape[1], a.shape[0], GSD),
    }
    scene.update(ImageQualityScreener(cfg).screen_scene(scene))
    return cfg, scene


def test_ice_is_not_called_cloud():
    """
    The bug this exists for: a floe field is bright and, across VNIR only, flat enough to pass
    the whiteness test, so Prudhoe Bay reported 50.4 % cloud on an acquisition with 0.0 % cloud
    and the scene was thrown away. Ice must land in the ice mask and nowhere else.
    """
    a = _blank()
    _paint_ice(a, slice(0, 256), slice(None))
    _, scene = _screen(a)
    assert scene["quality_metrics"]["ice_cover_pct"] > 40.0
    assert scene["quality_metrics"]["cloud_cover_pct"] < 1.0
    assert scene["ice_mask"][10, 10] == 1
    assert scene["cloud_mask"][10, 10] == 0


def test_cloud_is_still_called_cloud():
    """The ice class must not be bought by giving up on cloud."""
    a = _blank()
    _paint_cloud(a, slice(0, 256), slice(None))
    _, scene = _screen(a)
    assert scene["quality_metrics"]["cloud_cover_pct"] > 40.0
    assert scene["quality_metrics"]["ice_cover_pct"] < 1.0


def test_ice_is_not_called_land_and_stays_searchable():
    """
    Ice fails the water test outright (NIR 0.29 against a 0.12 limit), so without an explicit
    ice class a floe field becomes a 50 km2 'island' and the 200 m shoreline keep-out erases
    every contact in the pack -- including the vessel we went there to find.
    """
    a = _blank()
    _paint_ice(a, slice(0, 256), slice(None))
    _, scene = _screen(a)
    assert scene["land_mask"][10, 10] == 0
    assert scene["sea_mask"][10, 10] == 1, "ice must remain inside the searchable sea"


def test_floe_in_pack_ice_is_rejected_but_the_same_floe_in_open_water_is_not():
    """
    The whole point of the regime. An identical bright blob is a candidate in open water and
    must not be one when it is surrounded by ice, because in ice that signature is just a floe.
    """
    def run(with_pack):
        a = _blank()
        if with_pack:
            rng = np.random.default_rng(7)
            for _ in range(140):  # scattered floes: enough to own the neighbourhood
                y, x = rng.integers(40, 470, 2)
                _paint_ice(a, slice(y, y + 12), slice(x, x + 16))
        _paint_ice(a, slice(250, 256), slice(250, 262))  # the candidate itself
        cfg, scene = _screen(a)
        dets, _ = VesselDetector(cfg).detect_scene(scene)
        return [d for d in dets
                if abs(d["apex_px"][0] - 256) < 20 and abs(d["apex_px"][1] - 253) < 20]

    assert run(with_pack=False), "a bright blob in open water is a legitimate candidate"
    assert not run(with_pack=True), "the same blob inside pack ice must be treated as a floe"


def test_lead_transform_finds_an_open_water_channel_astern():
    """
    A vessel working through pack ice leaves a lead: open water where the floes were. At 865 nm
    that channel is ~1 % reflectance against ~29 % for the ice, so it is a far stronger signal
    than the wake the same ship would raise in open sea -- and unlike a bright wake, a drifting
    floe cannot counterfeit it.
    """
    a = _blank()
    _paint_ice(a, slice(None), slice(None))          # solid pack, edge to edge
    a[250:258, 120:400, :] = _blank(8, 280)          # the channel: open water through the ice
    _paint_ice(a, slice(248, 260), slice(398, 416))  # the vessel at the head of it

    cfg, scene = _screen(a)
    det = VesselDetector(cfg)
    z, contrast, det_mask = det.compute_cfar(scene["array"][:, :, 3], scene["sea_mask"], GSD)
    import cv2
    z_smooth = cv2.blur(np.clip(z, -3.0, 12.0), (3, 3))
    lead = det._lead_transform(z_smooth, 405.0, 254.0, GSD)

    assert lead["found"], "the channel astern should be found"
    assert lead["length_px"] * GSD >= cfg.detection.lead_min_length_m
    # the channel runs away to the west; image bearing 270 deg, allow the ray quantisation
    assert min(abs(lead["bearing_deg"] - 270.0), 360.0 - abs(lead["bearing_deg"] - 270.0)) < 25.0


def test_open_water_has_no_lead():
    """The lead test must not fire on ordinary sea, or every contact would carry a fake channel."""
    a = _blank()
    _paint_ice(a, slice(250, 258), slice(390, 410))
    cfg, scene = _screen(a)
    det = VesselDetector(cfg)
    z, _, _ = det.compute_cfar(scene["array"][:, :, 3], scene["sea_mask"], GSD)
    import cv2
    lead = det._lead_transform(cv2.blur(np.clip(z, -3.0, 12.0), (3, 3)), 400.0, 254.0, GSD)
    assert not lead["found"]


def test_no_ice_scene_behaves_exactly_as_before():
    """
    The Arctic work must be free in temperate water: on the 16 real US scenes the ice mask fires
    on under 0.7 % of pixels and no candidate ever enters the regime. Assert the cheap version
    of that here -- an ice-free scene must produce an empty ice mask and no regime flag.
    """
    a = _blank()
    _paint_ice(a, slice(250, 256), slice(250, 262))  # one bright blob, no surrounding ice
    cfg, scene = _screen(a)
    assert scene["quality_metrics"]["ice_cover_pct"] < 1.0
    dets, _ = VesselDetector(cfg).detect_scene(scene)
    assert dets, "the blob is still a candidate"
    assert not any(d["ice_regime"] for d in dets)
