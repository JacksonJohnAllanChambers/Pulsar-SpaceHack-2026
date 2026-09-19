"""
AIS Kinematic Correlator and Anomaly Classifier (Stage 4).

The uplinked AIS catalogue is dead-reckoned to each scene's shutter time, then matched
one-to-one against optical detections. What falls out of the match is the intelligence:

  DARK_VESSEL              seen optically, nobody broadcasting nearby
  AIS_KINEMATIC_MISMATCH   a broadcaster is nearby but its course/speed contradict the wake
  CONFIRMED_KNOWN_VESSEL   position and kinematics agree (lowest downlink priority)
  AIS_NOT_OBSERVED         AIS claims a ship in clear open water and nothing is there
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from applet.core.base import BasePipeline
from applet.utils.geo import project_dead_reckoning, haversine_distance_nm, angular_difference_deg


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


class AISKinematicCorrelator(BasePipeline):
    # Kept as static aliases: other tools and tests call these through the class
    project_dead_reckoning = staticmethod(project_dead_reckoning)
    haversine_distance_nm = staticmethod(haversine_distance_nm)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.ais_correlation
        detections: List[Dict[str, Any]] = context.get("detected_vessels", [])
        catalog: List[Dict[str, Any]] = context.get("ais_catalog", [])
        scenes = {s["id"]: s for s in context.get("screened_scenes", [])}

        # Dead-reckon per scene, because shutter times differ along the pass
        predictions: Dict[str, List[Tuple[float, float]]] = {}
        for scene_id, scene in scenes.items():
            shutter = _parse_iso(scene.get("shutter_time"))
            predictions[scene_id] = [self._predict(ship, shutter) for ship in catalog]

        # Tier 1: tight gate that grows with the age of the fix (position uncertainty).
        # Tier 2: whatever is left may still pair inside the wide gate, but a broadcaster
        # that far from its own dead-reckoned position is by definition inconsistent.
        ages = {sid: [self._fix_age_hours(ship, _parse_iso(scenes[sid].get("shutter_time"))) for ship in catalog]
                for sid in scenes}
        pairs = []
        for di, det in enumerate(detections):
            lat, lon = det["world_coordinates"]["latitude"], det["world_coordinates"]["longitude"]
            for si, (plat, plon) in enumerate(predictions.get(det["scene_id"], [])):
                # A stale fix is not an identity: found on real data, where yesterday's broadcast from
                # the same berth "confirmed" a ship that was silent today.
                if abs(ages[det["scene_id"]][si]) > cfg.max_fix_age_hours:
                    continue
                dist = haversine_distance_nm(lat, lon, plat, plon)
                run_nm = float(catalog[si].get("sog_knots", 0.0)) * abs(ages[det["scene_id"]][si])
                tight = cfg.tight_gate_nm + cfg.gate_growth_fraction * run_nm
                # a false course can displace the prediction by at most twice the distance run
                wide = min(cfg.spatial_gating_radius_nm, cfg.tight_gate_nm + 2.0 * run_nm)
                if dist <= tight:
                    pairs.append((0, dist, di, si))
                elif dist <= wide:
                    ship = catalog[si]
                    from_fix = haversine_distance_nm(lat, lon, float(ship["latitude"]), float(ship["longitude"]))
                    if abs(from_fix - run_nm) <= tight:
                        pairs.append((1, dist, di, si))
        pairs.sort()
        det_match: Dict[int, Tuple[int, float, int]] = {}
        used_ships = set()
        for tier, dist, di, si in pairs:
            if di in det_match or (detections[di]["scene_id"], si) in used_ships:
                continue
            det_match[di] = (si, dist, tier)
            used_ships.add((detections[di]["scene_id"], si))

        structures = context.get("known_structures", [])
        for di, det in enumerate(detections):
            if di in det_match:
                si, dist, tier = det_match[di]
                self._classify_matched(det, catalog[si], dist, off_track=tier == 1)
                continue
            structure = self._nearest_structure(det, structures)
            if structure is not None:
                self._classify_structure(det, structure)
            else:
                self._classify_dark(det)

        detections.sort(key=lambda d: (-d["downlink_priority"], d["detection_id"]))

        context["classified_targets"] = detections
        context["ais_not_observed"] = self._unobserved_broadcasters(
            catalog, predictions, scenes, used_ships, cfg.max_fix_age_hours * 3600.0,
            self.config.detection.shore_exclusion_m)
        context["ais_predictions"] = {
            sid: [
                {"mmsi": ship.get("mmsi"), "name": ship.get("name"), "latitude": round(p[0], 6),
                 "longitude": round(p[1], 6), "cog_deg": ship.get("cog_deg"), "sog_knots": ship.get("sog_knots")}
                for ship, p in zip(catalog, preds)
            ]
            for sid, preds in predictions.items()
        }
        for key, label in (("dark_vessels_count", "DARK_VESSEL"),
                           ("spoofing_anomalies_count", "AIS_KINEMATIC_MISMATCH"),
                           ("confirmed_known_count", "CONFIRMED_KNOWN_VESSEL"),
                           ("known_structures_count", "KNOWN_STRUCTURE")):
            context[key] = sum(1 for t in detections if t["classification"] == label)
        return context

    def _fix_age_hours(self, ship: Dict[str, Any], shutter: Optional[datetime]) -> float:
        fix_time = _parse_iso(ship.get("timestamp"))
        if shutter is not None and fix_time is not None:
            delta_h = (shutter - fix_time).total_seconds() / 3600.0
        else:
            delta_h = float(ship.get("delta_hours_to_shutter", self.config.ais_correlation.default_delta_hours))
        return delta_h

    def _predict(self, ship: Dict[str, Any], shutter: Optional[datetime]) -> Tuple[float, float]:
        # extrapolation is capped; fixes older than max_fix_age_hours are ignored by the matcher anyway
        limit = self.config.ais_correlation.max_fix_age_hours
        delta_h = min(max(self._fix_age_hours(ship, shutter), -limit), limit)
        return project_dead_reckoning(
            float(ship["latitude"]), float(ship["longitude"]),
            float(ship.get("sog_knots", 0.0)), float(ship.get("cog_deg", 0.0)), delta_h,
        )

    def _classify_matched(self, det: Dict[str, Any], ship: Dict[str, Any], dist_nm: float, off_track: bool = False) -> None:
        cfg = self.config.ais_correlation
        cog, sog = float(ship.get("cog_deg", 0.0)), float(ship.get("sog_knots", 0.0))
        problems = []
        if off_track:
            problems.append(f"found {dist_nm:.2f} NM from its dead-reckoned position")

        # A wake proves motion; heading from hull shape alone is ambiguous by 180 degrees
        if det["target_type"] != "VESSEL_STATIONARY_OR_SLOW" and sog >= 2.0:
            delta = angular_difference_deg(det["heading_deg"], cog)
            if det["heading_ambiguous_180"]:
                delta = min(delta, 180.0 - delta)
            if delta > cfg.max_heading_delta_deg:
                problems.append(f"observed heading {det['heading_deg']:.0f} deg vs reported COG {cog:.0f} deg")

        # The transverse-wave speed estimate is the least mature measurement in the chain, so by
        # default a speed disagreement is reported in the notes but cannot, alone, raise an anomaly
        speed = det.get("estimated_speed_knots")
        speed_note = ""
        if speed is not None and abs(speed - sog) > cfg.speed_tolerance_knots:
            text = f"wake-derived speed {speed:.1f} kn vs reported SOG {sog:.1f} kn"
            if cfg.speed_can_raise_anomaly:
                problems.append(text)
            else:
                speed_note = f" Advisory: {text}."
        # Real berthed ships lie along piers, and a pier reads as a short "wake". Only a long, strong
        # wake contradicts a transponder that says the ship is stopped.
        if det["target_type"] == "VESSEL_UNDERWAY" and sog < 0.5 \
                and det.get("wake_length_m", 0.0) >= cfg.static_mismatch_min_wake_m \
                and det.get("wake_snr", 0.0) >= cfg.static_mismatch_min_wake_snr:
            problems.append(f"{det['wake_length_m']:.0f} m wake but AIS reports {sog:.1f} kn")

        det["matched_vessel"] = ship.get("mmsi")
        det["matched_vessel_name"] = ship.get("name")
        det["ais_distance_nm"] = round(dist_nm, 3)
        if problems:
            det["classification"] = "AIS_KINEMATIC_MISMATCH"
            det["downlink_priority"] = round(0.60 + 0.25 * det["confidence"], 3)
            det["intelligence_notes"] = (
                f"Co-located with MMSI {ship.get('mmsi')} ({dist_nm:.2f} NM) but " + "; ".join(problems) + "."
            )
        else:
            det["classification"] = "CONFIRMED_KNOWN_VESSEL"
            det["downlink_priority"] = 0.10
            det["intelligence_notes"] = (
                f"Matches {ship.get('name', 'UNKNOWN')} (MMSI {ship.get('mmsi')}) at {dist_nm:.2f} NM; "
                f"position and course consistent.{speed_note}"
            )
        det["ais_status"] = "CORRELATED"

    @staticmethod
    def _nearest_structure(det: Dict[str, Any], structures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        c = det["world_coordinates"]
        for st in structures:
            radius_nm = float(st.get("radius_m", 150.0)) / 1852.0
            if haversine_distance_nm(c["latitude"], c["longitude"], float(st["latitude"]), float(st["longitude"])) <= radius_nm:
                return st
        return None

    @staticmethod
    def _classify_structure(det: Dict[str, Any], structure: Dict[str, Any]) -> None:
        """Platforms, artificial islands, moored buoys: bright, ship-sized, never on AIS, never moving."""
        det.update(matched_vessel=None, matched_vessel_name=structure.get("name"), ais_distance_nm=None,
                   classification="KNOWN_STRUCTURE", downlink_priority=0.05, ais_status="CHARTED",
                   intelligence_notes=f"Charted fixed structure: {structure.get('name', 'unnamed')}.")

    def _classify_dark(self, det: Dict[str, Any]) -> None:
        cfg = self.config.ais_correlation
        c = det["world_coordinates"]
        det["matched_vessel"] = None
        det["matched_vessel_name"] = None
        det["ais_distance_nm"] = None
        det["classification"] = "DARK_VESSEL"
        det["downlink_priority"] = round(0.70 + 0.30 * det["confidence"], 3)
        det["intelligence_notes"] = (
            f"{det['size_class'].replace('_', ' ').title()} at [{c['latitude']:.4f}, {c['longitude']:.4f}] "
            f"with no AIS broadcast within {cfg.spatial_gating_radius_nm} NM."
        )
        det["ais_status"] = "NO_AIS"

    @staticmethod
    def _unobserved_broadcasters(catalog, predictions, scenes, used_ships, max_age_s: float = 3 * 3600.0,
                                 shore_exclusion_m: float = 0.0) -> List[Dict[str, Any]]:
        import cv2
        import numpy as np

        out = []
        for scene_id, preds in predictions.items():
            scene = scenes[scene_id]
            if not scene["quality_metrics"]["is_usable"]:
                continue
            georef = scene["georef"]
            shutter = _parse_iso(scene.get("shutter_time"))
            keepout = scene["land_mask"]
            if shore_exclusion_m > 0 and keepout.any():
                k = 2 * max(1, int(round(shore_exclusion_m / georef.gsd_m))) + 1
                keepout = cv2.dilate(keepout, np.ones((k, k), np.uint8))
            for si, (plat, plon) in enumerate(preds):
                if (scene_id, si) in used_ships:
                    continue
                fix = _parse_iso(catalog[si].get("timestamp"))
                if shutter is not None and fix is not None and abs((shutter - fix).total_seconds()) > max_age_s:
                    continue
                x, y = georef.lonlat_to_pixel(plon, plat)
                if not (0 <= x < georef.width and 0 <= y < georef.height):
                    continue
                xi, yi = int(x), int(y)
                if scene["cloud_mask"][yi, xi]:
                    reason = "UNDER_CLOUD"
                elif keepout[yi, xi]:
                    reason = "IN_PORT_OR_SHORE_KEEPOUT"  # the detector does not look here, so silence means nothing
                elif 0 < float(catalog[si].get("length_m") or 0) < 2.5 * georef.gsd_m:
                    reason = "BELOW_SENSOR_RESOLUTION"  # a 12 m yacht is one mixed pixel at 10 m GSD
                else:
                    reason = "CLEAR_WATER_NO_TARGET"  # the only reason that suggests a ghost transponder
                out.append({
                    "scene_id": scene_id, "mmsi": catalog[si].get("mmsi"), "name": catalog[si].get("name"),
                    "predicted_latitude": round(plat, 6), "predicted_longitude": round(plon, 6),
                    "classification": "AIS_NOT_OBSERVED", "reason": reason,
                })
        return out
