"""
AIS kinematic correlation: dead reckoning, gating and anomaly classes.
"""

import numpy as np

from applet.config import AppletConfig
from applet.pipelines.ais_correlator import AISKinematicCorrelator
from applet.utils.geo import (
    SceneGeoreference, haversine_distance_nm, project_dead_reckoning, speed_from_kelvin_wavelength,
)

LAT, LON = 44.6488, -63.5752


def _scene():
    georef = SceneGeoreference({"center_lat": LAT, "center_lon": LON}, 1024, 1024, 4.75)
    zeros = np.zeros((1024, 1024), np.uint8)
    return {"id": "S1", "shutter_time": "2026-09-19T12:00:00Z", "georef": georef,
            "quality_metrics": {"is_usable": True}, "cloud_mask": zeros, "land_mask": zeros}


def _det(heading=315.0, speed=16.0, target_type="VESSEL_UNDERWAY", lat=LAT, lon=LON):
    return {"detection_id": "S1_T001", "scene_id": "S1", "world_coordinates": {"latitude": lat, "longitude": lon},
            "heading_deg": heading, "heading_ambiguous_180": False, "estimated_speed_knots": speed,
            "target_type": target_type, "size_class": "MEDIUM_VESSEL", "confidence": 0.9}


def _run(det, ships):
    ctx = {"detected_vessels": [det], "ais_catalog": ships, "screened_scenes": [_scene()]}
    return AISKinematicCorrelator(AppletConfig()).process(ctx)


def test_haversine_one_minute_of_latitude_is_one_nautical_mile():
    assert round(haversine_distance_nm(LAT, LON, LAT + 1 / 60, LON), 2) == 1.0


def test_dead_reckoning_round_trip():
    lat, lon = project_dead_reckoning(LAT, LON, 12.0, 90.0, 0.5)  # 6 NM due east
    assert round(haversine_distance_nm(LAT, LON, lat, lon), 2) == 6.0
    assert lon > LON and abs(lat - LAT) < 0.01


def test_kelvin_wavelength_to_speed():
    # lambda = 2*pi*V^2/g ; 10 m/s -> 64.07 m -> 19.44 kn
    assert abs(speed_from_kelvin_wavelength(64.07) - 19.44) < 0.05


def test_no_ais_nearby_is_a_dark_vessel():
    far = {"mmsi": 1, "name": "FAR", "latitude": 45.5, "longitude": -62.0, "sog_knots": 12, "cog_deg": 315,
           "timestamp": "2026-09-19T12:00:00Z"}
    out = _run(_det(), [far])
    t = out["classified_targets"][0]
    assert t["classification"] == "DARK_VESSEL" and t["downlink_priority"] > 0.9
    assert out["dark_vessels_count"] == 1


def test_consistent_broadcaster_is_confirmed_and_deprioritised():
    # fix taken 6 min before shutter, 1.6 NM back along the track
    fix_lat, fix_lon = project_dead_reckoning(LAT, LON, 16.0, 135.0, 0.1)
    ship = {"mmsi": 2, "name": "HONEST", "latitude": fix_lat, "longitude": fix_lon, "sog_knots": 16.0,
            "cog_deg": 315.0, "timestamp": "2026-09-19T11:54:00Z"}
    t = _run(_det(), [ship])["classified_targets"][0]
    assert t["classification"] == "CONFIRMED_KNOWN_VESSEL" and t["downlink_priority"] < 0.2
    assert t["matched_vessel"] == 2


def test_wake_contradicting_reported_course_is_flagged():
    ship = {"mmsi": 3, "name": "LIAR", "latitude": LAT, "longitude": LON, "sog_knots": 16.0, "cog_deg": 90.0,
            "timestamp": "2026-09-19T12:00:00Z"}
    t = _run(_det(heading=315.0), [ship])["classified_targets"][0]
    assert t["classification"] == "AIS_KINEMATIC_MISMATCH"


def test_wake_while_reporting_stationary_is_flagged():
    ship = {"mmsi": 4, "name": "ANCHORED", "latitude": LAT, "longitude": LON, "sog_knots": 0.0, "cog_deg": 0.0,
            "timestamp": "2026-09-19T12:00:00Z"}
    t = _run(_det(speed=None), [ship])["classified_targets"][0]
    assert t["classification"] == "AIS_KINEMATIC_MISMATCH"


def test_broadcaster_with_no_optical_target_is_reported():
    ghost_lat, ghost_lon = project_dead_reckoning(LAT, LON, 1.0, 45.0, 1.0)  # 1 NM away, inside the frame
    ghost = {"mmsi": 5, "name": "GHOST", "latitude": ghost_lat, "longitude": ghost_lon, "sog_knots": 0.0,
             "cog_deg": 0.0, "timestamp": "2026-09-19T12:00:00Z"}
    out = _run(_det(), [ghost])
    assert out["classified_targets"][0]["classification"] == "DARK_VESSEL"
    assert [g["mmsi"] for g in out["ais_not_observed"]] == [5]
    assert out["ais_not_observed"][0]["reason"] == "CLEAR_WATER_NO_TARGET"
