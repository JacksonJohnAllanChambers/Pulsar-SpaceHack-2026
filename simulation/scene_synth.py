"""
Physically-motivated synthetic VNIR ocean scenes (ground-side tooling, never flown).

Scenes are rendered in reflectance units at a chosen GSD so the onboard thresholds mean
the same thing here as on real imagery. Vessels are drawn 3x supersampled in the ship's
own frame and area-averaged down, so a 12 m boat really is a couple of mixed pixels.

What is modelled, and why it matters to the detector:
  sea        band-dependent reflectance (NIR ~1 %), swell, wind texture, sensor noise
  whitecaps  1-3 px spectrally flat specks           -> CFAR false alarms in rough seas
  vessels    tapered hull, deck colour, turbulent wake, Kelvin cusp arms and transverse
             waves with lambda = 2*pi*V^2/g          -> what the ray transform measures
  clouds     fractal alpha field, soft fringes       -> occlusion + fringe false alarms
  land/rocks coast with surf line, small islets      -> the classic optical confusers
  sunglint   broad NIR brightening with extra texture
"""

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

G = 9.80665
KNOT = 0.514444
KELVIN_HALF_ANGLE = math.radians(19.47)

# reflectance in [red, green, blue, nir]
SEA = np.array([0.022, 0.042, 0.060, 0.012], dtype=np.float32)
FOAM = np.array([1.00, 1.00, 1.00, 0.85], dtype=np.float32)
HULL_PALETTE = [
    np.array([0.38, 0.38, 0.37, 0.36], dtype=np.float32),  # white superstructure
    np.array([0.20, 0.20, 0.21, 0.22], dtype=np.float32),  # navy grey
    np.array([0.24, 0.10, 0.08, 0.26], dtype=np.float32),  # oxide-red deck
    np.array([0.10, 0.16, 0.13, 0.20], dtype=np.float32),  # green deck
    np.array([0.30, 0.27, 0.20, 0.30], dtype=np.float32),  # containers / mixed
]
VEGETATION = np.array([0.05, 0.08, 0.04, 0.36], dtype=np.float32)
ROCK = np.array([0.16, 0.14, 0.11, 0.21], dtype=np.float32)
CLOUD = np.array([0.62, 0.62, 0.63, 0.60], dtype=np.float32)


@dataclass
class Vessel:
    x: float  # pixel position of hull centre
    y: float
    heading_deg: float  # image frame, 0 = up, clockwise
    speed_knots: float
    length_m: float
    beam_m: float
    palette: int = 0
    ais: str = "on"  # on | dark | spoof_course | spoof_static
    name: str = ""


@dataclass
class SceneSpec:
    width: int = 1024
    height: int = 1024
    gsd_m: float = 4.75
    seed: int = 0
    wind: float = 0.3  # 0 calm .. 1 gale: texture + whitecaps
    glint: float = 0.0  # 0 .. 1
    cloud_cover: float = 0.0  # approximate fraction
    coast: bool = False
    islets: int = 0
    vessels: List[Vessel] = field(default_factory=list)


def fractal_noise(rng: np.random.Generator, h: int, w: int, octaves: int = 5, base: int = 4) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * (2 ** o)
        grid = rng.standard_normal((max(2, n * h // max(h, w) + 1), max(2, n * w // max(h, w) + 1))).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def render_sea(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    h, w = spec.height, spec.width
    large = fractal_noise(rng, h, w, octaves=3, base=2) - 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    swell_dir = rng.uniform(0, math.pi)
    swell_len = rng.uniform(90, 220) / spec.gsd_m
    swell = np.sin(2 * math.pi * (xx * math.cos(swell_dir) + yy * math.sin(swell_dir)) / swell_len)

    texture = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2)
    texture /= texture.std() + 1e-9

    scene = np.empty((h, w, 4), dtype=np.float32)
    for b in range(4):
        level = SEA[b] * (1.0 + 0.25 * large)
        level = level * (1.0 + 0.04 * (0.3 + spec.wind) * swell + 0.05 * spec.wind * texture)
        scene[:, :, b] = level

    if spec.glint > 0:
        gdir = rng.uniform(0, 2 * math.pi)
        ramp = ((xx - w / 2) * math.cos(gdir) + (yy - h / 2) * math.sin(gdir)) / max(h, w) + 0.5
        glint = spec.glint * 0.07 * np.clip(ramp, 0, 1) ** 1.5
        glint = glint * (1.0 + 0.35 * texture + 0.25 * swell)
        scene += np.clip(glint, 0, None)[:, :, None]

    n_caps = int(rng.poisson(max(spec.wind - 0.35, 0.0) * 60.0 * (h * w * spec.gsd_m ** 2) / 1e6))
    if n_caps:
        caps = np.zeros((h, w), dtype=np.float32)
        cx, cy = rng.integers(0, w, n_caps), rng.integers(0, h, n_caps)
        caps[cy, cx] = rng.uniform(0.05, 0.30, n_caps).astype(np.float32)
        streak = rng.uniform(0.6, 1.4)
        caps = cv2.GaussianBlur(caps, (0, 0), sigmaX=streak, sigmaY=0.6) * (2 * math.pi * streak * 0.6)
        scene += caps[:, :, None] * FOAM[None, None, :]
    return scene


def render_vessel(scene: np.ndarray, v: Vessel, gsd: float, rng: np.random.Generator, ss: int = 3) -> None:
    """Composites hull + wake system into `scene` in place."""
    h, w = scene.shape[:2]
    speed = v.speed_knots * KNOT
    moving = v.speed_knots >= 1.5
    wake_len_m = min(v.speed_knots * rng.uniform(28, 55), 1600.0) if moving else 0.0
    kelvin = moving and v.speed_knots >= 10 and v.length_m >= 45

    # bounding patch (pixels) big enough for the wedge
    reach_px = (wake_len_m + v.length_m) / gsd + 6
    half = int(math.ceil(reach_px if moving else v.length_m / gsd + 6))
    x0, x1 = max(int(v.x) - half, 0), min(int(v.x) + half + 1, w)
    y0, y1 = max(int(v.y) - half, 0), min(int(v.y) + half + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    pw, ph = (x1 - x0) * ss, (y1 - y0) * ss

    # supersampled metric coordinates relative to hull centre
    px = (np.arange(pw, dtype=np.float32) + 0.5) / ss + x0 - v.x
    py = (np.arange(ph, dtype=np.float32) + 0.5) / ss + y0 - v.y
    gx, gy = np.meshgrid(px * gsd, py * gsd)
    hd = math.radians(v.heading_deg)
    fwd_x, fwd_y = math.sin(hd), -math.cos(hd)
    along = gx * fwd_x + gy * fwd_y  # +ve toward the bow
    across = -gx * fwd_y + gy * fwd_x

    L, B = v.length_m, v.beam_m
    bow_zone = np.clip((L / 2 - along) / (0.28 * L), 0.0, 1.0)  # taper to a point at the bow
    hull = ((np.abs(along) <= L / 2) & (np.abs(across) <= (B / 2) * bow_zone)).astype(np.float32)

    foam = np.zeros_like(hull)
    if moving:
        astern = -(along + L / 2)  # metres behind the stern
        behind = astern > 0
        sigma = (B / 2 + 0.035 * np.clip(astern, 0, None)) / 1.4 + 0.8
        amp = (0.035 + 0.006 * v.speed_knots) * min(1.0, L / 60.0 + 0.35)
        foam += behind * amp * np.exp(-np.clip(astern, 0, None) / (0.33 * wake_len_m)) * np.exp(
            -0.5 * (across / sigma) ** 2
        )
        # bow wave
        foam += 0.5 * amp * np.exp(-0.5 * (((along - L / 2) / (0.08 * L + 2)) ** 2 + (across / (B / 2 + 2)) ** 2))

        if kelvin:
            from_bow = L / 2 - along
            wedge_half = np.tan(KELVIN_HALF_ANGLE) * np.clip(from_bow, 0, None)
            in_range = (from_bow > 0) & (from_bow < wake_len_m)
            lam = 2 * math.pi * speed ** 2 / G
            decay = np.exp(-np.clip(from_bow, 0, None) / (0.45 * wake_len_m))
            arm_sigma = 3.0 + 0.012 * np.clip(from_bow, 0, None)
            arm_amp = 0.020 * min(1.0, v.speed_knots / 18.0) * min(1.0, L / 120.0 + 0.3)
            arms = np.exp(-0.5 * ((np.abs(across) - wedge_half) / arm_sigma) ** 2)
            foam += in_range * arm_amp * decay * arms * (0.7 + 0.3 * np.sin(2 * math.pi * from_bow / (0.67 * lam)))
            inside = in_range & (np.abs(across) < wedge_half)
            shape = 1.0 - (across / (wedge_half + 1e-3)) ** 2
            trans = 0.35 * arm_amp * decay * shape * np.sin(2 * math.pi * from_bow / lam)
            foam += inside * trans

    size = (x1 - x0, y1 - y0)
    hull_a = cv2.resize(hull, size, interpolation=cv2.INTER_AREA)
    foam_a = cv2.resize(foam, size, interpolation=cv2.INTER_AREA)

    colour = HULL_PALETTE[v.palette % len(HULL_PALETTE)]
    patch = scene[y0:y1, x0:x1]
    patch += foam_a[:, :, None] * FOAM[None, None, :]
    # deck detail: brightness varies along the ship
    detail = 1.0 + 0.25 * np.sin(np.linspace(0, rng.uniform(4, 12), size[0], dtype=np.float32))[None, :]
    patch[:] = patch * (1 - hull_a[:, :, None]) + (colour[None, None, :] * detail[:, :, None]) * hull_a[:, :, None]


def render_coast(scene: np.ndarray, spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    """Adds land along one edge plus a surf line. Returns the land alpha (for vessel placement)."""
    h, w = spec.height, spec.width
    noise = fractal_noise(rng, h, w, octaves=5, base=3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    side = int(rng.integers(0, 4))
    ramp = [xx / w, 1 - xx / w, yy / h, 1 - yy / h][side]
    field_ = ramp * 1.6 + (noise - 0.5) * 0.9
    land = (field_ < 0.42).astype(np.float32)
    land_soft = cv2.GaussianBlur(land, (0, 0), 0.8)
    surf = np.clip(cv2.GaussianBlur(land, (0, 0), 2.0) - land_soft, 0, 1) * 0.5
    texture = 0.7 + 0.6 * fractal_noise(rng, h, w, octaves=4, base=8)
    mix = fractal_noise(rng, h, w, octaves=3, base=6)[:, :, None]
    ground = (VEGETATION[None, None, :] * mix + ROCK[None, None, :] * (1 - mix)) * texture[:, :, None]
    scene += surf[:, :, None] * FOAM[None, None, :] * 0.4
    scene[:] = scene * (1 - land_soft[:, :, None]) + ground * land_soft[:, :, None]
    return land_soft


def render_islets(scene: np.ndarray, spec: SceneSpec, rng: np.random.Generator, blocked: np.ndarray) -> List[Tuple[int, int]]:
    placed = []
    h, w = spec.height, spec.width
    for _ in range(spec.islets):
        cx, cy = int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30))
        if blocked[cy, cx] > 0.05:
            continue
        rx, ry = rng.uniform(8, 45) / spec.gsd_m, rng.uniform(8, 45) / spec.gsd_m
        alpha = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(alpha, (cx, cy), (max(int(rx), 1), max(int(ry), 1)), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
        scene[:] = scene * (1 - alpha[:, :, None]) + ROCK[None, None, :] * alpha[:, :, None]
        placed.append((cx, cy))
    return placed


def render_clouds(scene: np.ndarray, spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    h, w = spec.height, spec.width
    noise = fractal_noise(rng, h, w, octaves=6, base=3)
    thresh = float(np.quantile(noise, 1.0 - min(max(spec.cloud_cover, 0.0), 0.98)))
    alpha = np.clip((noise - thresh) / 0.06, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    bright = 0.8 + 0.4 * fractal_noise(rng, h, w, octaves=4, base=6)
    scene[:] = scene * (1 - alpha[:, :, None]) + (CLOUD[None, None, :] * bright[:, :, None]) * alpha[:, :, None]
    return alpha


def random_vessel(rng: np.random.Generator, spec: SceneSpec, margin_px: int = 40) -> Vessel:
    kind = rng.choice(["small", "fishing", "coaster", "cargo", "tanker"], p=[0.25, 0.25, 0.2, 0.2, 0.1])
    length = {"small": rng.uniform(10, 22), "fishing": rng.uniform(20, 45), "coaster": rng.uniform(50, 110),
              "cargo": rng.uniform(120, 260), "tanker": rng.uniform(200, 340)}[kind]
    beam = max(length / rng.uniform(5.5, 7.5), 3.5)
    speed = 0.0 if rng.random() < 0.18 else {"small": rng.uniform(6, 30), "fishing": rng.uniform(3, 11),
                                             "coaster": rng.uniform(8, 14), "cargo": rng.uniform(11, 21),
                                             "tanker": rng.uniform(9, 15)}[kind]
    ais = str(rng.choice(["on", "dark", "spoof_course", "spoof_static"], p=[0.55, 0.3, 0.1, 0.05]))
    if speed == 0.0 and ais.startswith("spoof"):
        ais = "on"  # a moored ship has no kinematics to falsify
    return Vessel(
        x=float(rng.uniform(margin_px, spec.width - margin_px)),
        y=float(rng.uniform(margin_px, spec.height - margin_px)),
        heading_deg=float(rng.uniform(0, 360)), speed_knots=float(speed), length_m=float(length),
        beam_m=float(beam), palette=int(rng.integers(0, len(HULL_PALETTE))),
        ais=ais,
    )


def random_spec(rng: np.random.Generator, size: int, gsd: float) -> SceneSpec:
    """A scene with randomly drawn weather, coast, clutter and traffic."""
    spec = SceneSpec(size, size, gsd, seed=int(rng.integers(0, 2 ** 31 - 1)))
    spec.wind = float(rng.uniform(0.0, 1.0))
    spec.glint = float(rng.uniform(0.2, 1.0)) if rng.random() < 0.3 else 0.0
    spec.cloud_cover = float(rng.uniform(0.04, 0.4)) if rng.random() < 0.55 else 0.0
    spec.coast = bool(rng.random() < 0.25)
    spec.islets = int(rng.integers(0, 5)) if rng.random() < 0.4 else 0
    spec.vessels = [random_vessel(rng, spec, margin_px=24) for _ in range(int(rng.integers(0, 6)))]
    return spec


def render_scene(spec: SceneSpec) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns (reflectance[H,W,4] float32, truth dict)."""
    rng = np.random.default_rng(spec.seed)
    scene = render_sea(spec, rng)

    land = render_coast(scene, spec, rng) if spec.coast else np.zeros((spec.height, spec.width), np.float32)
    islets = render_islets(scene, spec, rng, land) if spec.islets else []

    kept: List[Vessel] = []
    for v in spec.vessels:
        xi, yi = int(np.clip(v.x, 0, spec.width - 1)), int(np.clip(v.y, 0, spec.height - 1))
        if cv2.GaussianBlur(land, (0, 0), 8.0)[yi, xi] > 0.02:
            continue  # do not park ships on the beach
        render_vessel(scene, v, spec.gsd_m, rng)
        kept.append(v)

    cloud = render_clouds(scene, spec, rng) if spec.cloud_cover > 0 else np.zeros((spec.height, spec.width), np.float32)

    scene = cv2.GaussianBlur(scene, (0, 0), 0.55)  # optics + detector MTF
    noise_sigma = 0.0012 + 0.02 * np.sqrt(np.clip(scene, 0, None)) * 0.05
    scene = scene + rng.standard_normal(scene.shape).astype(np.float32) * noise_sigma
    np.clip(scene, 0.0005, 1.2, out=scene)

    truth_vessels = []
    for v in kept:
        xi, yi = int(np.clip(v.x, 0, spec.width - 1)), int(np.clip(v.y, 0, spec.height - 1))
        d = asdict(v)
        d["cloud_alpha"] = round(float(cloud[yi, xi]), 3)
        d["visible"] = bool(cloud[yi, xi] < 0.35)
        truth_vessels.append(d)
    truth = {"vessels": truth_vessels, "islets": [list(p) for p in islets],
             "cloud_fraction": round(float((cloud > 0.5).mean()), 4), "land_fraction": round(float((land > 0.5).mean()), 4)}
    return scene.astype(np.float32), truth


def to_uint16(reflectance: np.ndarray, scale: float = 10000.0) -> np.ndarray:
    return np.clip(reflectance * scale + 0.5, 1, 65535).astype(np.uint16)
