"""Persistent spatial memory for repeated dark-vessel false alarms."""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from applet.utils.geo import haversine_distance_nm


class UnknownContactMemory:
    def __init__(self, path: str, radius_nm: float, required_passes: int):
        self.path = os.path.abspath(path)
        self.radius_nm = radius_nm
        self.required_passes = required_passes
        self.entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            entries = data.get("entries", []) if isinstance(data, dict) else []
            return [entry for entry in entries if self._valid(entry)]
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def _valid(entry: Any) -> bool:
        return isinstance(entry, dict) and all(key in entry for key in ("latitude", "longitude", "passes"))

    def suppression_regions(self) -> List[Dict[str, float]]:
        return [
            {"latitude": float(entry["latitude"]), "longitude": float(entry["longitude"]), "passes": int(entry["passes"])}
            for entry in self.entries
            if int(entry["passes"]) >= self.required_passes
        ]

    def is_suppressed(self, latitude: float, longitude: float, protected_locations: Optional[List[Dict[str, float]]] = None) -> bool:
        if protected_locations and any(
            haversine_distance_nm(latitude, longitude, location["latitude"], location["longitude"]) <= self.radius_nm
            for location in protected_locations
        ):
            return False
        return any(
            haversine_distance_nm(latitude, longitude, region["latitude"], region["longitude"]) <= self.radius_nm
            for region in self.suppression_regions()
        )

    def record(self, latitude: float, longitude: float) -> None:
        nearest: Optional[Dict[str, Any]] = None
        nearest_distance = float("inf")
        for entry in self.entries:
            distance = haversine_distance_nm(latitude, longitude, float(entry["latitude"]), float(entry["longitude"]))
            if distance <= self.radius_nm and distance < nearest_distance:
                nearest, nearest_distance = entry, distance
        if nearest is None:
            self.entries.append({"latitude": round(latitude, 6), "longitude": round(longitude, 6), "passes": 1})
        else:
            nearest["latitude"] = round((float(nearest["latitude"]) + latitude) / 2.0, 6)
            nearest["longitude"] = round((float(nearest["longitude"]) + longitude) / 2.0, 6)
            nearest["passes"] = int(nearest["passes"]) + 1

    def record_pass(self, observations: List[Dict[str, float]]) -> None:
        """Record at most one observation per remembered area for a single pass."""
        recorded = []
        for observation in observations:
            latitude, longitude = float(observation["latitude"]), float(observation["longitude"])
            if any(haversine_distance_nm(latitude, longitude, lat, lon) <= self.radius_nm
                   for lat, lon in recorded):
                continue
            self.record(latitude, longitude)
            recorded.append((latitude, longitude))

    def save(self) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="unknown-memory-", suffix=".json", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"entries": self.entries}, stream, indent=2)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)