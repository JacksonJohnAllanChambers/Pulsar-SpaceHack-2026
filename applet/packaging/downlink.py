"""
Downlink Package Generator for Orbital Pass Transmission.

The science product (GeoJSON + scene report + chips) is packed into a byte-reproducible
tarball: fixed mtimes, fixed ordering, compact JSON. Identical input -> identical bytes,
which is what lets the ground segment deduplicate and checksum re-transmissions.
Housekeeping telemetry (timings, RAM) is inherently run-dependent, so it rides next to
the tarball rather than inside it; both are counted against the byte budget.
"""

import io
import os
import gzip
import json
import shutil
import tarfile
from typing import Dict, Any, List
from applet.config import AppletConfig

_PROPERTY_KEYS = (
    "detection_id", "scene_id", "classification", "downlink_priority", "target_type", "size_class",
    "confidence", "physics_score", "verifier_prob", "heading_deg", "heading_ambiguous_180",
    "estimated_speed_knots", "speed_method", "hull_length_m", "hull_width_m", "wake_length_m",
    "kelvin_arms_detected", "kelvin_half_angle_deg", "matched_vessel", "ais_distance_nm", "intelligence_notes",
)


def _dump(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


class DownlinkPackager:
    def __init__(self, config: AppletConfig, output_dir: str):
        self.config = config
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def package(self, context: Dict[str, Any], telemetry_summary: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.downlink
        targets = context.get("classified_targets", [])
        max_bytes = cfg.max_downlink_budget_kb * 1024

        files: Dict[str, bytes] = {
            "tactical_intelligence.geojson": _dump(self._build_geojson(targets, context)),
            "scene_report.json": _dump(self._build_scene_report(context)),
        }

        # Chips are the expensive part: spend the remaining budget on the highest priority targets
        included_chips: List[str] = []
        budget_used = sum(len(b) for b in files.values()) + 1024
        if cfg.include_target_chips:
            for tgt in targets:  # already sorted by priority
                chip = tgt.get("chip_jpeg")
                if not chip or tgt["downlink_priority"] < cfg.chip_min_priority:
                    continue
                if budget_used + len(chip) > max_bytes:
                    break
                name = f"chips/{tgt['detection_id']}.jpg"
                files[name] = chip
                included_chips.append(name)
                budget_used += len(chip)

        files["manifest.json"] = _dump({
            "mission": self.config.mission.model_dump(),
            "targets_detected": len(targets),
            "dark_vessels": context.get("dark_vessels_count", 0),
            "kinematic_mismatches": context.get("spoofing_anomalies_count", 0),
            "confirmed_known": context.get("confirmed_known_count", 0),
            "chips_included": included_chips,
            "budget_kb": cfg.max_downlink_budget_kb,
        })

        bundle_dir = os.path.join(self.output_dir, "downlink_bundle")
        shutil.rmtree(bundle_dir, ignore_errors=True)
        for name, data in files.items():
            path = os.path.join(bundle_dir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)

        tar_path = os.path.join(self.output_dir, f"downlink_{self.config.mission.orbital_pass_id}.tar.gz")
        self._write_reproducible_tar(tar_path, files)

        telemetry_path = os.path.join(self.output_dir, "edge_telemetry.json")
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_summary, f, indent=2)

        tar_bytes = os.path.getsize(tar_path)
        return {
            "downlink_tarball_path": tar_path,
            "downlink_bundle_dir": bundle_dir,
            "telemetry_path": telemetry_path,
            "final_bundle_bytes": tar_bytes,
            "final_bundle_kb": round(tar_bytes / 1024, 2),
            "geojson_path": os.path.join(bundle_dir, "tactical_intelligence.geojson"),
            "included_chips_count": len(included_chips),
            "within_budget": tar_bytes <= max_bytes,
        }

    @staticmethod
    def write_telemetry(path: str, telemetry_summary: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(telemetry_summary, f, indent=2)

    @staticmethod
    def _write_reproducible_tar(tar_path: str, files: Dict[str, bytes]) -> None:
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(files):
                info = tarfile.TarInfo(name=f"downlink/{name}")
                info.size = len(files[name])
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(files[name]))
        with open(tar_path, "wb") as f:
            with gzip.GzipFile(filename="", fileobj=f, mode="wb", compresslevel=9, mtime=0) as gz:
                gz.write(raw.getvalue())

    def _build_geojson(self, targets: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        features = []
        for tgt in targets:
            c = tgt["world_coordinates"]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c["longitude"], c["latitude"]]},
                "properties": {k: tgt.get(k) for k in _PROPERTY_KEYS},
            })
        # Only a resolvable broadcaster missing from clear open water is intelligence (a possible ghost
        # transponder). Ships in port or too small to resolve are merely counted in the metadata.
        not_observed = context.get("ais_not_observed", [])
        reason_counts: Dict[str, int] = {}
        for ais in not_observed:
            reason_counts[ais["reason"]] = reason_counts.get(ais["reason"], 0) + 1
        for ais in not_observed:
            if ais["reason"] != "CLEAR_WATER_NO_TARGET":
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [ais["predicted_longitude"], ais["predicted_latitude"]]},
                "properties": {k: ais[k] for k in ("scene_id", "mmsi", "name", "classification", "reason")},
            })
        return {
            "type": "FeatureCollection",
            "metadata": {
                "satellite_id": self.config.mission.satellite_id,
                "pass_id": self.config.mission.orbital_pass_id,
                "total_targets": len(targets),
                "dark_vessels": context.get("dark_vessels_count", 0),
                "ais_not_observed_by_reason": reason_counts,
            },
            "features": features,
        }

    @staticmethod
    def _build_scene_report(context: Dict[str, Any]) -> Dict[str, Any]:
        scenes = []
        for s in context.get("screened_scenes", []):
            scenes.append({
                "id": s["id"],
                "quality": s["quality_metrics"],
                "targets": len(s.get("detections", [])),
                "detector": s.get("detector_stats"),
            })
        for s in context.get("rejected_scenes", []):
            scenes.append({"id": s.get("id"), "quality": {"is_usable": False, "rejection_reasons": [s.get("error")]}})
        verifier = dict(context.get("verifier_info", {}))
        verifier.pop("inference_ms", None)  # timing is telemetry, not science
        return {"scenes": scenes, "funnel": context.get("detection_funnel", {}), "verifier": verifier}
