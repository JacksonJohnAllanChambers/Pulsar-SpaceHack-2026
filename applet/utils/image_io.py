"""
Resilient image I/O for satellite sensor ingestion.

Every scene is normalised to one internal representation so the rest of the
pipeline never has to care about file format, bit depth or band order:

    float32 array, shape (H, W, 4), band order [red, green, blue, nir],
    values in reflectance units (0..1), plus a boolean no-data mask.

Loading never raises: corrupt headers, empty files, NaNs and missing bands all
come back as a status dict the validator can report.
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

CANONICAL_BANDS = ("red", "green", "blue", "nir")
_MAX_BANDS = 16


def _read_pixels(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npy":
        return np.load(file_path, allow_pickle=False)
    if ext in (".tif", ".tiff"):
        try:
            import tifffile

            return tifffile.imread(file_path)
        except ImportError:
            import cv2

            arr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError("TIFF decode failed")
            return arr
    from PIL import Image

    with Image.open(file_path) as pil_img:
        pil_img.verify()
    with Image.open(file_path) as pil_img:
        return np.array(pil_img)


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"unsupported array rank {arr.ndim}")
    # Band-first rasters (C, H, W) are the GeoTIFF norm
    if arr.shape[0] <= _MAX_BANDS and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] > _MAX_BANDS:
        raise ValueError(f"implausible band count {arr.shape[2]}")
    return arr


def _default_band_names(channels: int) -> List[str]:
    if channels >= 4:
        return ["red", "green", "blue", "nir"] + [f"extra_{i}" for i in range(channels - 4)]
    if channels == 3:
        return ["red", "green", "blue"]
    return ["nir"] if channels == 1 else ["red", "green"]


def load_scene_raster(
    file_path: str,
    band_names: Optional[List[str]] = None,
    reflectance_scale: Optional[float] = None,
    reflectance_offset: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns (reflectance[H,W,4] float32, nodata_mask[H,W] bool, status).
    On any failure returns (None, None, status) with status["error"] set.
    """
    status: Dict[str, Any] = {
        "file_path": file_path,
        "is_valid": False,
        "width": 0,
        "height": 0,
        "source_channels": 0,
        "source_dtype": None,
        "raw_sensor_bytes": 0,
        "missing_bands": [],
        "warnings": [],
        "error": None,
    }

    if not os.path.exists(file_path):
        status["error"] = "FILE_NOT_FOUND"
        return None, None, status
    if os.path.getsize(file_path) == 0:
        status["error"] = "ZERO_BYTE_FILE"
        return None, None, status

    try:
        raw = _to_hwc(_read_pixels(file_path))
        if raw.size == 0 or raw.shape[0] < 16 or raw.shape[1] < 16:
            status["error"] = "EMPTY_OR_TINY_RASTER"
            return None, None, status

        h, w, c = raw.shape
        status.update(width=w, height=h, source_channels=c, source_dtype=str(raw.dtype))
        status["raw_sensor_bytes"] = int(raw.nbytes)

        if reflectance_scale is None:
            if raw.dtype == np.uint8:
                reflectance_scale = 255.0
            elif np.issubdtype(raw.dtype, np.integer):
                reflectance_scale = 10000.0
            else:
                reflectance_scale = 1.0

        names = [n.lower() for n in (band_names or _default_band_names(c))][:c]

        out = np.zeros((h, w, 4), dtype=np.float32)
        nodata = np.ones((h, w), dtype=bool)
        present: Dict[str, bool] = {}

        for idx, target in enumerate(CANONICAL_BANDS):
            if target not in names:
                present[target] = False
                continue
            band = raw[:, :, names.index(target)].astype(np.float32)
            finite = np.isfinite(band)
            if not finite.all():
                status["warnings"].append(f"NON_FINITE_VALUES_{target.upper()}")
                band = np.where(finite, band, 0.0)
            nodata &= band == 0
            band = (band + reflectance_offset) / reflectance_scale
            np.clip(band, 0.0, 1.5, out=band)
            # A band that is constant carries no information: dead detector / dropped packet
            if float(band.max()) - float(band.min()) < 1e-6:
                present[target] = False
                status["warnings"].append(f"DEAD_BAND_{target.upper()}")
                continue
            out[:, :, idx] = band
            present[target] = True

        missing = [b for b in CANONICAL_BANDS if not present.get(b)]
        status["missing_bands"] = missing
        if len(missing) == 4:
            status["error"] = "NO_USABLE_BANDS"
            return None, None, status

        _fill_missing_bands(out, present)
        out[nodata] = 0.0

        status["is_valid"] = True
        return out, nodata, status

    except Exception as e:  # corrupt header, truncated stream, bad pickle, ...
        # The class, never the message. This string is downlinked in scene_report.json, and a
        # library's wording changes between versions: tifffile 2025.5 says "corrupted tag list @8"
        # where 2026.x says "suspicious number of tags 20291" for the same truncated file, which
        # was the one byte-level difference between the x86-64 and linux/arm64 tarballs.
        status["error"] = f"DECODE_ERROR: {type(e).__name__}"
        return None, None, status


def _fill_missing_bands(out: np.ndarray, present: Dict[str, bool]) -> None:
    """Substitute the spectrally nearest surviving band so downstream maths stays defined."""
    order = {"nir": ["red", "green", "blue"], "red": ["green", "nir", "blue"],
             "green": ["red", "blue", "nir"], "blue": ["green", "red", "nir"]}
    for idx, band in enumerate(CANONICAL_BANDS):
        if present.get(band):
            continue
        for sub in order[band]:
            if present.get(sub):
                out[:, :, idx] = out[:, :, CANONICAL_BANDS.index(sub)]
                break


def split_bands(reflectance: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: reflectance[:, :, i] for i, name in enumerate(CANONICAL_BANDS)}
