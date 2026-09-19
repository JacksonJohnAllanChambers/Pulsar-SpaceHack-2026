"""The scorecard produces the numbers we publish, so its arithmetic is tested like flight code.

These build contexts by hand rather than running a pass: the point is to pin down what each
metric counts, especially the denominators, which are where an honest scorecard usually goes wrong.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.scorecard import ang_diff, quantiles, score, score_labels  # noqa: E402


class FakeGeoref:
    width = 1000
    height = 1000
    gsd_m = 10.0


def make_scene(sid="S1", usable=True, water_pct=50.0):
    return {
        "id": sid,
        "georef": FakeGeoref(),
        "quality_metrics": {
            "is_usable": usable, "water_pct": water_pct, "valid_data_pct": 100.0,
            "cloud_cover_pct": 1.0,
        },
    }


def make_target(did, cls, scene="S1", **kw):
    det = {
        "detection_id": did, "scene_id": scene, "classification": cls,
        "heading_deg": 90.0, "heading_ambiguous_180": False, "hull_length_m": 100.0,
        "hull_resolved": True, "world_coordinates": {"latitude": 0.0, "longitude": 0.0},
        "apex_px": [10, 10], "confidence": 0.9,
    }
    det.update(kw)
    return det


def make_context(targets, unobserved=(), catalog=(), scenes=None):
    return {
        "screened_scenes": scenes if scenes is not None else [make_scene()],
        "classified_targets": list(targets),
        "ais_not_observed": list(unobserved),
        "ais_catalog": list(catalog),
    }


TELEMETRY = {"wall_clock_time_s": 1.0, "peak_memory_mb": 100.0, "verifier": {"status": "READY"}}


def test_quantiles_and_angles():
    assert quantiles([]) == {"n": 0, "median": None, "p90": None, "max": None}
    q = quantiles([1.0, 2.0, 3.0, 100.0])
    assert q["n"] == 4 and q["median"] == 3.0 and q["max"] == 100.0
    assert ang_diff(350.0, 10.0) == pytest.approx(20.0)
    assert ang_diff(10.0, 350.0) == pytest.approx(20.0)


def test_recall_denominator_excludes_ships_the_sensor_cannot_see():
    """The headline recall must be against visible broadcasters only.

    A ship alongside a quay, under cloud, or smaller than a pixel is not a detection the
    design claims to make, and counting it would understate a number we then have to defend.
    """
    targets = [make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1, ais_distance_nm=0.05)]
    unobserved = [
        {"scene_id": "S1", "mmsi": 2, "reason": "CLEAR_WATER_NO_TARGET"},
        {"scene_id": "S1", "mmsi": 3, "reason": "IN_PORT_OR_SHORE_KEEPOUT"},
        {"scene_id": "S1", "mmsi": 4, "reason": "UNDER_CLOUD"},
        {"scene_id": "S1", "mmsi": 5, "reason": "BELOW_SENSOR_RESOLUTION"},
    ]
    result = score(make_context(targets, unobserved), TELEMETRY)
    ais = result["ais"]
    assert ais["broadcasters_in_footprint"] == 5
    assert ais["visible_to_sensor"] == 2        # one matched + one genuinely missed
    assert ais["recall_vs_visible_ais"] == 0.5
    assert ais["unmatched_reasons"]["IN_PORT_OR_SHORE_KEEPOUT"] == 1


def test_mismatch_counts_as_a_detection_not_a_miss():
    """An AIS_KINEMATIC_MISMATCH still found the ship; only its behaviour is disputed."""
    targets = [
        make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1, ais_distance_nm=0.01),
        make_target("D2", "AIS_KINEMATIC_MISMATCH", matched_vessel=2, ais_distance_nm=0.02),
    ]
    result = score(make_context(targets), TELEMETRY)
    assert result["ais"]["matched"] == 2
    assert result["ais"]["recall_vs_visible_ais"] == 1.0
    assert result["position_error_m"]["n"] == 2


def test_water_area_drives_the_false_alarm_rate():
    scene = make_scene(water_pct=50.0)  # 1000x1000 px at 10 m = 100 km2, half water
    targets = [make_target(f"D{i}", "DARK_VESSEL") for i in range(5)]
    result = score(make_context(targets, scenes=[scene]), TELEMETRY)
    assert result["searched_water_km2"] == pytest.approx(50.0)
    assert result["contacts_without_ais"] == 5
    assert result["contacts_without_ais_per_1000km2"] == pytest.approx(100.0)


def test_unusable_scene_contributes_no_water():
    result = score(make_context([], scenes=[make_scene(usable=False)]), TELEMETRY)
    assert result["searched_water_km2"] == 0.0
    assert result["usable_scenes"] == 0


def test_heading_only_scored_for_ships_under_way():
    catalog = [
        {"mmsi": 1, "sog_knots": 0.2, "cog_deg": 270.0},   # at anchor: COG is noise
        {"mmsi": 2, "sog_knots": 12.0, "cog_deg": 95.0},
    ]
    targets = [
        make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1, ais_distance_nm=0.01),
        make_target("D2", "CONFIRMED_KNOWN_VESSEL", matched_vessel=2, ais_distance_nm=0.01),
    ]
    result = score(make_context(targets, catalog=catalog), TELEMETRY)
    assert result["heading_error_deg"]["n"] == 1          # the anchored ship is excluded
    assert result["heading_error_deg"]["median"] == pytest.approx(5.0)


def test_180_flip_is_counted_but_folded_when_the_detector_declared_it_ambiguous():
    catalog = [{"mmsi": 1, "sog_knots": 10.0, "cog_deg": 270.0}]
    flipped = make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1,
                          ais_distance_nm=0.01, heading_deg=90.0)
    result = score(make_context([flipped], catalog=catalog), TELEMETRY)
    assert result["heading_flipped_180_pct"] == 100.0
    assert result["heading_error_deg"]["median"] == pytest.approx(180.0)

    ambiguous = make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1,
                            ais_distance_nm=0.01, heading_deg=90.0, heading_ambiguous_180=True)
    folded = score(make_context([ambiguous], catalog=catalog), TELEMETRY)
    assert folded["heading_error_deg"]["median"] == pytest.approx(0.0)


def test_labels_give_precision_and_never_assume_the_unlabelled():
    targets = [
        make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1, ais_distance_nm=0.01),
        make_target("D2", "DARK_VESSEL"),
        make_target("D3", "DARK_VESSEL"),
        make_target("D4", "DARK_VESSEL"),   # deliberately left unlabelled
    ]
    labels = {"D2": "vessel", "D3": "not_vessel"}
    out = score_labels(targets, labels, water_km2=100.0, n_matched=1, visible_broadcasters=1)
    assert out["adjudicated_contacts"] == 3     # D1 by AIS, D2 and D3 by hand
    assert out["unlabelled_contacts"] == 1
    assert out["true_vessels"] == 2 and out["false_alarms"] == 1
    assert out["precision"] == pytest.approx(2 / 3, abs=5e-4)  # the scorecard rounds to 3 dp
    assert out["false_alarms_per_1000km2"] == pytest.approx(10.0)


def test_labelled_recall_adds_confirmed_unlisted_craft_to_the_truth_set():
    """A reviewer-confirmed vessel with no transponder is a real vessel we did find."""
    targets = [
        make_target("D1", "CONFIRMED_KNOWN_VESSEL", matched_vessel=1, ais_distance_nm=0.01),
        make_target("D2", "DARK_VESSEL"),
    ]
    # 1 matched, 2 visible broadcasters -> 1 AIS ship missed
    out = score_labels(targets, {"D2": "vessel"}, water_km2=100.0,
                       n_matched=1, visible_broadcasters=2)
    assert out["true_vessels"] == 2
    assert out["recall_vs_all_known_vessels"] == pytest.approx(2 / 3, abs=5e-4)  # 2 found of 3 known
