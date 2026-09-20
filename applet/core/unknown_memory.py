"""Persistent spatial memory for repeated dark-vessel false alarms.

An area counts once per distinct acquisition DATE, never once per run: re-processing the same
bundle must not change what the applet reports, or the downlink stops being reproducible and a
fourth demo run silently erases every contact. Entries that stop recurring expire."""

import json
import os
import tempfile
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from applet.utils.geo import haversine_distance_nm


class UnknownContactMemory:
    def __init__(self, path: str, radius_nm: float, required_passes: int, max_age_days: float = 180.0):
        self.path = os.path.abspath(path)
        self.radius_nm = radius_nm
        self.required_passes = required_passes
        self.max_age_days = max_age_days
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

    def record(self, latitude: float, longitude: float, observed_on: str) -> None:
        nearest: Optional[Dict[str, Any]] = None
        nearest_distance = float("inf")
        for entry in self.entries:
            distance = haversine_distance_nm(latitude, longitude, float(entry["latitude"]), float(entry["longitude"]))
            if distance <= self.radius_nm and distance < nearest_distance:
                nearest, nearest_distance = entry, distance
        if nearest is None:
            self.entries.append({"latitude": round(latitude, 6), "longitude": round(longitude, 6),
                                 "passes": 1, "dates": [observed_on]})
        elif observed_on not in nearest.setdefault("dates", []):
            nearest["latitude"] = round((float(nearest["latitude"]) + latitude) / 2.0, 6)
            nearest["longitude"] = round((float(nearest["longitude"]) + longitude) / 2.0, 6)
            nearest["dates"] = sorted(nearest["dates"] + [observed_on])
            nearest["passes"] = len(nearest["dates"])

    def record_pass(self, observations: List[Dict[str, float]], observed_on: Optional[str]) -> None:
        """Record at most one observation per remembered area for one acquisition date (YYYY-MM-DD).

        Without a date a re-run cannot be told from a revisit, so nothing is learned.
        """
        if not observed_on:
            return
        self._expire(observed_on)
        recorded = []
        for observation in observations:
            latitude, longitude = float(observation["latitude"]), float(observation["longitude"])
            if any(haversine_distance_nm(latitude, longitude, lat, lon) <= self.radius_nm
                   for lat, lon in recorded):
                continue
            self.record(latitude, longitude, observed_on)
            recorded.append((latitude, longitude))

    def _expire(self, today: str) -> None:
        cutoff = (date.fromisoformat(today) - timedelta(days=self.max_age_days)).isoformat()
        self.entries = [entry for entry in self.entries if max(entry.get("dates") or [today]) >= cutoff]

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