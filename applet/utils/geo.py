"""
Dependency-free geolocation helpers.

Onboard we avoid GDAL/pyproj: the bundle manifest carries either the four corner
coordinates of each scene (preferred, any projection, any rotation) or a centre
point for a north-up scene. Pixel -> lat/lon is a bilinear blend of the corners,
which is accurate to a few metres over a 20 km swath.
"""

import math
from typing import Dict, Any, Tuple

EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_NM = 3440.065
METERS_PER_DEG_LAT = 111320.0
KNOTS_PER_MPS = 1.943844
G = 9.80665


class SceneGeoreference:
    def __init__(self, scene: Dict[str, Any], width: int, height: int, gsd_m: float):
        self.width = width
        self.height = height
        self.gsd_m = gsd_m
        corners = scene.get("corners")
        if corners and all(k in corners for k in ("ul", "ur", "lr", "ll")):
            # each corner is [lon, lat]
            self.ul, self.ur = corners["ul"], corners["ur"]
            self.lr, self.ll = corners["lr"], corners["ll"]
        else:
            lat0 = float(scene.get("center_lat", 0.0))
            lon0 = float(scene.get("center_lon", 0.0))
            half_h = (height / 2.0) * gsd_m / METERS_PER_DEG_LAT
            half_w = (width / 2.0) * gsd_m / (METERS_PER_DEG_LAT * max(math.cos(math.radians(lat0)), 1e-6))
            self.ul = [lon0 - half_w, lat0 + half_h]
            self.ur = [lon0 + half_w, lat0 + half_h]
            self.lr = [lon0 + half_w, lat0 - half_h]
            self.ll = [lon0 - half_w, lat0 - half_h]

        # Bearing of the image "up" direction (row decreasing) relative to true north
        up_lon = ((self.ul[0] + self.ur[0]) - (self.ll[0] + self.lr[0])) / 2.0
        up_lat = ((self.ul[1] + self.ur[1]) - (self.ll[1] + self.lr[1])) / 2.0
        mid_lat = (self.ul[1] + self.ll[1]) / 2.0
        self.north_offset_deg = math.degrees(
            math.atan2(up_lon * math.cos(math.radians(mid_lat)), up_lat)
        )

    def pixel_to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        u = x / max(self.width - 1, 1)
        v = y / max(self.height - 1, 1)
        lon = (1 - v) * ((1 - u) * self.ul[0] + u * self.ur[0]) + v * ((1 - u) * self.ll[0] + u * self.lr[0])
        lat = (1 - v) * ((1 - u) * self.ul[1] + u * self.ur[1]) + v * ((1 - u) * self.ll[1] + u * self.lr[1])
        return lon, lat

    def lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """Approximate inverse (exact for parallelogram footprints); used for AIS overlays."""
        ax, ay = self.ur[0] - self.ul[0], self.ur[1] - self.ul[1]
        bx, by = self.ll[0] - self.ul[0], self.ll[1] - self.ul[1]
        px, py = lon - self.ul[0], lat - self.ul[1]
        det = ax * by - ay * bx
        if abs(det) < 1e-18:
            return -1.0, -1.0
        u = (px * by - py * bx) / det
        v = (ax * py - ay * px) / det
        return u * (self.width - 1), v * (self.height - 1)

    def image_bearing_to_true(self, dx: float, dy: float) -> float:
        """Image-space direction (x right, y down) -> compass bearing in degrees."""
        bearing = math.degrees(math.atan2(dx, -dy))
        return (bearing + self.north_offset_deg) % 360.0


def project_dead_reckoning(
    lat: float, lon: float, speed_knots: float, course_deg: float, delta_hours: float
) -> Tuple[float, float]:
    """Great-circle forward projection of an AIS fix to the image shutter time."""
    distance_km = speed_knots * delta_hours * 1.852
    ang = distance_km / EARTH_RADIUS_KM
    course_rad = math.radians(course_deg)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(ang) + math.cos(lat_rad) * math.sin(ang) * math.cos(course_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(course_rad) * math.sin(ang) * math.cos(lat_rad),
        math.cos(ang) - math.sin(lat_rad) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    return EARTH_RADIUS_NM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def angular_difference_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def speed_from_kelvin_wavelength(wavelength_m: float) -> float:
    """Deep-water transverse wake wavelength lambda = 2*pi*V^2/g  ->  V in knots."""
    return math.sqrt(G * wavelength_m / (2.0 * math.pi)) * KNOTS_PER_MPS
