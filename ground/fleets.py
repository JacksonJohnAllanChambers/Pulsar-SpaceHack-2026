"""Ground-side dispatch of delivered dark-vessel images to nearby demo fleets."""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree

from applet.utils.geo import haversine_distance_nm


FLEETS = (
    {"id": "gibraltar", "name": "Gibraltar Patrol", "latitude": 36.04, "longitude": -5.35},
    {"id": "suez", "name": "Suez Watch", "latitude": 29.86, "longitude": 32.54},
    {"id": "longbeach", "name": "Long Beach Response", "latitude": 33.71, "longitude": -118.20},
    {"id": "halifax", "name": "Halifax Harbour Guard", "latitude": 44.60, "longitude": -63.54},
    {"id": "dover", "name": "Dover Strait Patrol", "latitude": 51.01, "longitude": 1.39},
    {"id": "singapore", "name": "Singapore Strait Watch", "latitude": 1.30, "longitude": 104.01},
    {"id": "newyork", "name": "New York Harbour Guard", "latitude": 40.45, "longitude": -73.91},
    {"id": "norfolk", "name": "Norfolk Coastal Watch", "latitude": 36.92, "longitude": -76.05},
    {"id": "galveston", "name": "Galveston Channel Patrol", "latitude": 29.27, "longitude": -94.74},
    {"id": "tampa", "name": "Tampa Bay Response", "latitude": 27.57, "longitude": -82.79},
    {"id": "savannah", "name": "Savannah Coastal Guard", "latitude": 31.95, "longitude": -80.84},
    {"id": "charleston", "name": "Charleston Harbour Watch", "latitude": 32.65, "longitude": -79.79},
    {"id": "sanfrancisco", "name": "San Francisco Bay Patrol", "latitude": 37.78, "longitude": -122.64},
    {"id": "pugetsound", "name": "Puget Sound Watch", "latitude": 48.13, "longitude": -122.64},
    {"id": "miami", "name": "Miami Coastal Response", "latitude": 25.72, "longitude": -80.12},
    {"id": "mississippi", "name": "Delta Channel Patrol", "latitude": 28.91, "longitude": -89.34},
    {"id": "boston", "name": "Boston Harbour Guard", "latitude": 42.30, "longitude": -70.89},
    {"id": "delaware", "name": "Delaware Bay Watch", "latitude": 38.77, "longitude": -75.04},
    {"id": "corpus", "name": "Corpus Christi Patrol", "latitude": 27.77, "longitude": -97.04},
    {"id": "honolulu", "name": "Honolulu Harbour Watch", "latitude": 21.25, "longitude": -157.94},
    {"id": "sandiego", "name": "San Diego Coastal Guard", "latitude": 32.62, "longitude": -117.29},
)

_ALERT_NAME = re.compile(
    r"^alert-v1__(?P<detection>[A-Za-z0-9_-]+)__(?P<scene>[A-Za-z0-9_-]+)"
    r"__lat-(?P<latitude>-?\d+(?:\.\d+)?)__lon-(?P<longitude>-?\d+(?:\.\d+)?)"
    r"__DARK_VESSEL\.jpg$"
)


def parse_alert_filename(path: Path) -> Optional[Dict[str, Any]]:
    """Read dark-vessel location metadata from a delivered crop filename."""
    match = _ALERT_NAME.match(Path(path).name)
    if not match:
        return None
    item = match.groupdict()
    return {
        "detection_id": item["detection"],
        "scene_id": item["scene"],
        "latitude": float(item["latitude"]),
        "longitude": float(item["longitude"]),
        "classification": "DARK_VESSEL",
    }


def nearest_fleet(latitude: float, longitude: float, fleets: Iterable[Dict[str, Any]] = FLEETS) -> Dict[str, Any]:
    """Return the closest fleet and its range in nautical miles."""
    candidates = list(fleets)
    if not candidates:
        raise ValueError("at least one fleet is required")
    fleet = min(candidates, key=lambda item: haversine_distance_nm(latitude, longitude, item["latitude"], item["longitude"]))
    return {**fleet, "distance_nm": round(haversine_distance_nm(latitude, longitude, fleet["latitude"], fleet["longitude"]), 2)}


def load_alerts(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            alerts = json.load(file)
        return alerts if isinstance(alerts, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _ping_directory(alert: Dict[str, Any], fleet_output_dir: Path) -> Path:
    return Path(fleet_output_dir) / alert["fleet"]["id"] / f"ping_{alert['detection_id']}"


def _write_ping_info(path: Path, alert: Dict[str, Any]) -> None:
    root = ElementTree.Element("fleet_ping")
    for key in ("detection_id", "scene_id", "classification", "latitude", "longitude", "distance_nm", "source_filename", "dispatched_at"):
        ElementTree.SubElement(root, key).text = str(alert[key])
    fleet = ElementTree.SubElement(root, "fleet")
    for key in ("id", "name", "latitude", "longitude"):
        ElementTree.SubElement(fleet, key).text = str(alert["fleet"][key])
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_fleet_ping(image_path: Path, alert: Dict[str, Any], fleet_output_dir: Path) -> Path:
    """Copy a delivered crop and its metadata into the receiving fleet's alert folder."""
    ping_dir = _ping_directory(alert, fleet_output_dir)
    ping_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, ping_dir / "image.jpg")
    _write_ping_info(ping_dir / "info.xml", alert)
    return ping_dir


def dispatch_sent_alerts(sent_dir: Path, alert_path: Path, fleet_output_dir: Path) -> List[Dict[str, Any]]:
    """Persist a single simulated fleet ping for each new delivered dark-vessel crop."""
    sent_dir, alert_path, fleet_output_dir = Path(sent_dir), Path(alert_path), Path(fleet_output_dir)
    alerts = load_alerts(alert_path)
    alerts_by_source = {alert.get("source_filename"): alert for alert in alerts}
    new_alerts = []
    for image_path in sorted(sent_dir.glob("*.jpg")) if sent_dir.is_dir() else []:
        target = parse_alert_filename(image_path)
        if target is None:
            continue
        alert = alerts_by_source.get(image_path.name)
        if alert is None:
            fleet = nearest_fleet(target["latitude"], target["longitude"])
            alert = {
                **target,
                "fleet": {key: fleet[key] for key in ("id", "name", "latitude", "longitude")},
                "distance_nm": fleet["distance_nm"],
                "source_filename": image_path.name,
                "dispatched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            alerts.append(alert)
            alerts_by_source[image_path.name] = alert
            new_alerts.append(alert)
        write_fleet_ping(image_path, alert, fleet_output_dir)
    if new_alerts:
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alert_path, "w", encoding="utf-8") as file:
            json.dump(alerts, file, indent=2)
    return new_alerts