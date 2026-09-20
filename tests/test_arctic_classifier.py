"""
Arctic ship / iceberg / uncertain call.

The three-way design and the first three cases are Megan Neville's (branch icebergWatch). The
evidence behind them was replaced after scoring on real contacts, and the invariants below are
the ones that scoring turned up: off unless asked, never in an ice-free scene, a transponder
outranks the image, the image alone never says SHIP, and ice is demoted -- never dropped.
"""

import numpy as np

from applet.cli import parse_args
from applet.config import AppletConfig
from applet.pipelines.ais_correlator import AISKinematicCorrelator
from applet.pipelines.arctic_classifier import ArcticClassifier
from applet.pipelines.chip_verifier import ChipVerifier
from applet.utils.geo import SceneGeoreference

LAT, LON = 78.93, 12.30  # Kongsfjorden
GSD = 10.0


def _config(enabled=True):
    config = AppletConfig()
    config.arctic.enabled = enabled
    config.verifier.enabled = False
    return config


def _chip(neighbours=0, target=True):
    """Dark water, one bright object under the contact, and `neighbours` more scattered around it."""
    chip = np.full((64, 64, 4), 0.02, dtype=np.float32)
    chip[..., 3] = 0.005
    if target:
        chip[30:34, 30:34] = (0.30, 0.32, 0.36, 0.15)
    spots = [(12, 12), (12, 50), (50, 12), (50, 50), (12, 31), (50, 31)]
    for y, x in spots[:neighbours]:
        chip[y:y + 3, x:x + 3] = (0.30, 0.32, 0.36, 0.15)
    return chip


def _scene(ice_pct=15.0):
    georef = SceneGeoreference({"center_lat": LAT, "center_lon": LON}, 1024, 1024, GSD)
    zeros = np.zeros((1024, 1024), np.uint8)
    return {"id": "S1", "gsd_m": GSD, "shutter_time": "2024-06-22T12:00:00Z", "georef": georef,
            "quality_metrics": {"is_usable": True, "ice_cover_pct": ice_pct},
            "cloud_mask": zeros, "land_mask": zeros, "detections": []}


def _det(det_id="S1_T001", wake=0.0, chip=None, lat=LAT):
    return {"detection_id": det_id, "scene_id": "S1", "world_coordinates": {"latitude": lat, "longitude": LON},
            "heading_deg": 0.0, "heading_ambiguous_180": True, "estimated_speed_knots": None,
            "target_type": "VESSEL_STATIONARY_OR_SLOW", "size_class": "SMALL_CRAFT", "confidence": 0.8,
            "physics_score": 0.6, "wake_length_m": wake, "wake_snr": 0.0, "lead_found": False,
            "kelvin_arms_detected": False, "chip_tensor": _chip() if chip is None else chip}


def _run(dets, config, ice_pct=15.0, ships=()):
    ctx = {"detected_vessels": dets, "ais_catalog": list(ships), "screened_scenes": [_scene(ice_pct)]}
    ctx = ChipVerifier(config).process(ctx)
    return AISKinematicCorrelator(config).process(ctx)


def test_evidence_counts_the_objects_around_the_contact_not_the_contact():
    clf = ArcticClassifier(_config())
    assert clf.evidence(_chip(neighbours=0), GSD)["neighbours"] == 0
    crowded = clf.evidence(_chip(neighbours=4), GSD)
    assert crowded["neighbours"] == 4
    assert crowded["neighbours_per_km2"] == round(4 / 0.4096, 2)
    assert crowded["nir_vis_slope"] < 0.6  # the synthetic object is blue and NIR-dark, like glacier ice


def test_nothing_under_the_contact_is_no_evidence():
    assert ArcticClassifier(_config()).evidence(_chip(neighbours=4, target=False), GSD) is None


def test_one_of_a_field_with_no_wake_is_an_iceberg_candidate():
    t = _run([_det(chip=_chip(neighbours=4))], _config())["classified_targets"][0]
    assert t["arctic_classification"] == "ICEBERG" and t["classification"] == "ICEBERG"
    assert "Probable ice" in t["intelligence_notes"]


def test_a_wake_protects_a_contact_but_does_not_make_it_a_ship():
    """Asserted from the image alone, SHIP was wrong on 21 of 21 Alaska pack-ice contacts."""
    t = _run([_det(wake=240.0, chip=_chip(neighbours=4))], _config())["classified_targets"][0]
    assert t["arctic_classification"] == "UNCERTAIN" and t["classification"] == "DARK_VESSEL"


def test_a_lone_bright_object_stays_uncertain():
    t = _run([_det(chip=_chip(neighbours=1))], _config())["classified_targets"][0]
    assert t["arctic_classification"] == "UNCERTAIN" and t["classification"] == "DARK_VESSEL"


def test_a_transponder_outranks_the_image():
    ship = {"mmsi": 257000001, "name": "BESET", "latitude": LAT, "longitude": LON, "sog_knots": 0.0,
            "cog_deg": 0.0, "timestamp": "2024-06-22T12:00:00Z"}
    t = _run([_det(chip=_chip(neighbours=4))], _config(), ships=[ship])["classified_targets"][0]
    assert t["arctic_classification"] == "SHIP" and t["classification"] == "CONFIRMED_KNOWN_VESSEL"


def test_ice_is_demoted_to_the_back_of_the_queue_and_never_dropped():
    dets = [_det("S1_T001", chip=_chip(neighbours=4)), _det("S1_T002", chip=_chip(neighbours=0), lat=LAT + 0.05)]
    out = _run(dets, _config())
    targets = out["classified_targets"]
    assert [t["detection_id"] for t in targets] == ["S1_T002", "S1_T001"]
    assert targets[1]["downlink_priority"] < 0.05  # under a charted structure
    assert out["icebergs_count"] == 1 and out["dark_vessels_count"] == 1 and out["arctic_uncertain_count"] == 1


def test_the_question_is_not_asked_in_an_ice_free_scene():
    """Ships at anchor crowd too (35 of 220 US vessels), so crowding only counts where there is ice."""
    t = _run([_det(chip=_chip(neighbours=4))], _config(), ice_pct=0.7)["classified_targets"][0]
    assert t["classification"] == "DARK_VESSEL"
    assert "arctic_classification" not in t and "arctic_evidence" not in t


def test_off_by_default_and_then_invisible():
    assert AppletConfig().arctic.enabled is False
    t = _run([_det(chip=_chip(neighbours=4))], _config(enabled=False))["classified_targets"][0]
    assert t["classification"] == "DARK_VESSEL"
    assert "arctic_classification" not in t and "arctic_evidence" not in t


def test_arctic_flag_is_available(monkeypatch):
    monkeypatch.setattr("sys.argv", ["applet", "run", "-i", "input", "-o", "output", "--arctic"])
    assert parse_args().arctic is True
