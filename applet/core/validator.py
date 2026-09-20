"""
Input Bundle Ingestion and Corruption Validator.
Guarantees satellite applet fails gracefully on bad input bundles or corrupted images.
"""

import os
import json
from typing import Dict, Any, List
from applet.utils.image_io import load_scene_raster
from applet.utils.geo import SceneGeoreference
from applet.core.exceptions import InvalidManifestError

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")

# Keys the ground segment may ship for scoring; the onboard pipeline must never see them.
_GROUND_ONLY_KEYS = ("ground_truth",)


class InputBundleValidator:
    """
    Validates and ingests the raw input bundle from satellite storage.
    Filters out corrupted images and checks manifest integrity.
    """

    def __init__(self, input_dir: str, default_gsd_m: float = 4.75):
        self.input_dir = os.path.abspath(input_dir)
        self.manifest_path = os.path.join(self.input_dir, "manifest.json")
        self.ais_catalog_path = os.path.join(self.input_dir, "ais_catalog.json")
        self.default_gsd_m = default_gsd_m
        self.manifest: Dict[str, Any] = {}
        self.valid_scenes: List[Dict[str, Any]] = []
        self.rejected_scenes: List[Dict[str, Any]] = []
        self.ais_catalog: List[Dict[str, Any]] = []
        self.known_structures: List[Dict[str, Any]] = []
        self.total_file_bytes: int = 0
        self.total_raw_bytes: int = 0

    def validate(self) -> Dict[str, Any]:
        if not os.path.isdir(self.input_dir):
            raise InvalidManifestError(f"Input directory does not exist: {self.input_dir}")

        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                if not isinstance(self.manifest, dict):
                    raise ValueError("manifest root must be an object")
            except Exception as e:
                raise InvalidManifestError(f"Failed to parse manifest.json: {str(e)}")

        self.ais_catalog = self._load_ais_catalog()
        self.known_structures = self._load_known_structures()

        image_entries = self.manifest.get("scenes") or []
        if not image_entries:
            for fname in sorted(os.listdir(self.input_dir)):
                if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                    image_entries.append({"id": os.path.splitext(fname)[0], "file": fname})

        for entry in image_entries:
            if not isinstance(entry, dict):
                continue
            self._ingest_scene(entry)

        return {
            "total_scenes_found": len(image_entries),
            "valid_scenes_count": len(self.valid_scenes),
            "rejected_scenes_count": len(self.rejected_scenes),
            "total_file_bytes": self.total_file_bytes,
            "total_raw_bytes": self.total_raw_bytes,
            "ais_vessels_in_catalog": len(self.ais_catalog),
        }

    def _load_ais_catalog(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.ais_catalog_path):
            return []
        try:
            with open(self.ais_catalog_path, "r", encoding="utf-8") as f:
                vessels = json.load(f).get("vessels", [])
            return [v for v in vessels if isinstance(v, dict) and "latitude" in v and "longitude" in v]
        except Exception:
            return []

    def _load_known_structures(self) -> List[Dict[str, Any]]:
        """Optional known_structures.json: charted platforms / islands / buoys uplinked with the AIS picture."""
        path = os.path.join(self.input_dir, "known_structures.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f).get("structures", [])
            return [s for s in items if isinstance(s, dict) and "latitude" in s and "longitude" in s]
        except Exception:
            return []

    def _ingest_scene(self, entry: Dict[str, Any]) -> None:
        info = {k: v for k, v in entry.items() if k not in _GROUND_ONLY_KEYS}
        fname = str(entry.get("file", ""))
        info.setdefault("id", os.path.splitext(fname)[0] or "scene")
        fpath = os.path.join(self.input_dir, fname)

        # Refuse path traversal out of the bundle
        if not os.path.abspath(fpath).startswith(self.input_dir):
            info["error"] = "PATH_OUTSIDE_BUNDLE"
            self.rejected_scenes.append(info)
            return

        if os.path.isfile(fpath):
            self.total_file_bytes += os.path.getsize(fpath)

        refl, nodata, status = load_scene_raster(
            fpath,
            band_names=entry.get("bands") or self.manifest.get("bands"),
            reflectance_scale=entry.get("reflectance_scale", self.manifest.get("reflectance_scale")),
            reflectance_offset=float(
                entry.get("reflectance_offset", self.manifest.get("reflectance_offset", 0.0))
            ),
        )
        if not status["is_valid"]:
            info["error"] = status["error"]
            self.rejected_scenes.append(info)
            return

        gsd = float(entry.get("gsd_meters") or self.manifest.get("gsd_meters") or self.default_gsd_m)
        info["array"] = refl
        info["nodata_mask"] = nodata
        info["status"] = status
        info["gsd_m"] = gsd
        info["shutter_time"] = entry.get("shutter_time", self.manifest.get("shutter_time"))
        info["georef"] = SceneGeoreference(entry, status["width"], status["height"], gsd)
        self.total_raw_bytes += status["raw_sensor_bytes"]
        self.valid_scenes.append(info)
