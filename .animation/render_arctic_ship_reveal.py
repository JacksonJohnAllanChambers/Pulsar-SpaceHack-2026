"""Render a realistic-style 10-second Arctic ship reveal using NumPy and OpenCV."""

from pathlib import Path

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SECONDS = 10
FRAME_COUNT = FPS * DURATION_SECONDS
OUTPUT = Path(__file__).with_name("arctic_ship_reveal.mp4")


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def natural_noise(shape: tuple[int, int], scale: int, rng: np.random.Generator) -> np.ndarray:
    small = rng.normal(0.0, 1.0, (max(2, shape[0] // scale), max(2, shape[1] // scale))).astype(np.float32)
    return cv2.GaussianBlur(cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC), (0, 0), scale * 0.38)


def normalized(values: np.ndarray) -> np.ndarray:
    return (values - values.min()) / max(float(values.max() - values.min()), 1e-6)


def draw_overhead_ship(image: np.ndarray, scale: float) -> None:
    """A low-contrast silhouette which initially resembles a long ice fragment."""
    layer = image.copy()
    ship_x, ship_y = int(WIDTH * 0.515), int(HEIGHT * 0.515)
    length, beam = int(172 * scale), int(43 * scale)
    hull = np.array([[ship_x, ship_y - length // 2], [ship_x - beam // 2, ship_y - length // 3],
                     [ship_x - beam // 2, ship_y + length // 2 - 18], [ship_x, ship_y + length // 2],
                     [ship_x + beam // 2, ship_y + length // 2 - 18], [ship_x + beam // 2, ship_y - length // 3]], np.int32)
    cv2.fillConvexPoly(layer, hull, (168, 178, 181), lineType=cv2.LINE_AA)
    cv2.rectangle(layer, (ship_x - int(beam * 0.34), ship_y - int(length * 0.10)),
                  (ship_x + int(beam * 0.34), ship_y + int(length * 0.28)), (142, 155, 160), -1)
    cv2.rectangle(layer, (ship_x - int(beam * 0.23), ship_y - int(length * 0.25)),
                  (ship_x + int(beam * 0.23), ship_y - int(length * 0.08)), (118, 136, 143), -1)
    cv2.line(layer, (ship_x, ship_y - int(length * 0.28)), (ship_x, ship_y - int(length * 0.43)), (86, 103, 108), 2)
    cv2.addWeighted(layer, 0.54, image, 0.46, 0, image)


def overhead_scene(progress: float, rng: np.random.Generator) -> np.ndarray:
    broad = normalized(natural_noise((HEIGHT, WIDTH), 105, rng))
    detail = normalized(natural_noise((HEIGHT, WIDTH), 14, rng))
    cells = cv2.resize(rng.random((34, 58)).astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    floes = cv2.GaussianBlur((cells > 0.48).astype(np.float32), (0, 0), 4)
    ocean = np.dstack((57 + 18 * broad, 82 + 27 * broad, 92 + 30 * broad))
    ice_tone = 169 + 57 * normalized(broad * 0.7 + detail * 0.3)
    ice = np.dstack((ice_tone * 0.92, ice_tone * 0.98, ice_tone))
    image = ocean * (1.0 - floes[..., None]) + ice * floes[..., None]
    cracks = cv2.Canny((floes * 255).astype(np.uint8), 20, 70)
    image[cracks > 0] *= np.array((0.58, 0.66, 0.70))
    image += rng.normal(0, 3.0, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    draw_overhead_ship(image, 0.58 + 0.16 * smoothstep(progress / 0.68))
    return cv2.addWeighted(image, 0.84, np.full_like(image, (205, 211, 209)), 0.16, 0)


def draw_landscape_ship(image: np.ndarray, horizon: int, progress: float) -> None:
    ship_x = int(WIDTH * (0.55 - 0.025 * progress))
    waterline = horizon + int(110 - 20 * progress)
    length, height = int(300 + 75 * progress), int(73 + 24 * progress)
    hull = np.array([[ship_x - length // 2, waterline], [ship_x - length // 2 + 38, waterline + height],
                     [ship_x + length // 2 - 28, waterline + height], [ship_x + length // 2, waterline + 12]], np.int32)
    cv2.fillConvexPoly(image, hull, (53, 65, 65), lineType=cv2.LINE_AA)
    cv2.polylines(image, [hull], True, (122, 135, 132), 2, cv2.LINE_AA)
    cv2.rectangle(image, (ship_x - 73, waterline - 58), (ship_x + 76, waterline + 2), (118, 132, 131), -1)
    cv2.rectangle(image, (ship_x - 32, waterline - 104), (ship_x + 33, waterline - 55), (93, 111, 114), -1)
    cv2.rectangle(image, (ship_x - 19, waterline - 93), (ship_x + 18, waterline - 67), (51, 71, 76), -1)
    cv2.line(image, (ship_x + 13, waterline - 106), (ship_x + 13, waterline - 162), (42, 56, 59), 4, cv2.LINE_AA)
    for offset in (-42, -16, 18, 44):
        cv2.rectangle(image, (ship_x + offset, waterline - 43), (ship_x + offset + 10, waterline - 31), (38, 52, 55), -1)
    wake = image.copy()
    cv2.ellipse(wake, (ship_x + length // 2 + 125, waterline + height + 21), (140, 24), 8, 0, 360, (205, 215, 209), -1)
    cv2.ellipse(wake, (ship_x + length // 2 + 220, waterline + height + 43), (150, 25), 8, 0, 360, (183, 200, 198), -1)
    cv2.addWeighted(wake, 0.50, image, 0.50, 0, image)


def landscape_scene(progress: float, rng: np.random.Generator) -> np.ndarray:
    horizon = int(HEIGHT * (0.42 - 0.035 * progress))
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    sky = np.linspace(0.0, 1.0, horizon)[:, None]
    image[:horizon] = np.dstack((174 - 56 * sky, 190 - 45 * sky, 193 - 35 * sky))
    image[:horizon] += (normalized(natural_noise((horizon, WIDTH), 85, rng))[..., None] - 0.55) * (13, 16, 17)
    water_height = HEIGHT - horizon
    wave = normalized(natural_noise((water_height, WIDTH), 16, rng))
    depth = np.linspace(0.0, 1.0, water_height)[:, None]
    ripples = np.sin(np.arange(water_height)[:, None] * 0.30 + np.arange(WIDTH)[None, :] * 0.018)
    water = np.dstack((56 + 20 * wave + 9 * depth, 83 + 29 * wave + 12 * depth, 89 + 32 * wave + 13 * depth))
    image[horizon:] = water + ripples[..., None] * (3 + 8 * depth[..., None])
    image = np.clip(image, 0, 255).astype(np.uint8)
    for index in range(15):
        y = horizon + int((index / 17) ** 2 * water_height)
        x, width = int(rng.uniform(-120, WIDTH + 120)), int(90 + 360 * ((y - horizon) / water_height) ** 1.6)
        cv2.ellipse(image, (x, y), (width, max(5, width // 10)), float(rng.uniform(-8, 8)), 0, 360, (175, 191, 193), -1)
    draw_landscape_ship(image, horizon, progress)
    return cv2.addWeighted(image, 0.89, np.full_like(image, (196, 205, 202)), 0.11, 0)


def render_frame(frame_index: int) -> np.ndarray:
    progress = frame_index / (FRAME_COUNT - 1)
    descent = smoothstep((progress - 0.26) / 0.56)
    overhead = overhead_scene(progress, np.random.default_rng(1000 + frame_index // 4))
    landscape = landscape_scene(descent, np.random.default_rng(50 + frame_index // 3))
    frame = cv2.addWeighted(overhead, 1.0 - descent, landscape, descent, 0)
    return cv2.GaussianBlur(frame, (0, 0), 0.35)


def main() -> None:
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer. Install FFmpeg and retry.")
    for frame_index in range(FRAME_COUNT):
        writer.write(render_frame(frame_index))
    writer.release()
    print(f"Wrote {OUTPUT} ({DURATION_SECONDS}s, {WIDTH}x{HEIGHT}, {FPS} fps)")


if __name__ == "__main__":
    main()