"""
Input bundle validation and error boundaries: nothing on disk may crash the applet.
"""

import json
import numpy as np
import pytest
from PIL import Image

from applet.core.validator import InputBundleValidator
from applet.core.exceptions import InvalidManifestError
from applet.utils.image_io import load_scene_raster


def test_validator_missing_directory():
    with pytest.raises(InvalidManifestError):
        InputBundleValidator("/non/existent/path/for/testing").validate()


def test_corrupt_and_empty_files_are_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"II*\x00NOT_A_REAL_TIFF")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    arr, _, status = load_scene_raster(str(bad))
    assert arr is None and not status["is_valid"] and "DECODE_ERROR" in status["error"]
    # Downlinked verbatim, so it may name the exception class but never quote the library: the
    # message text differs between tifffile versions and broke cross-platform byte identity.
    kind, _, detail = status["error"].partition(": ")
    assert kind == "DECODE_ERROR" and detail.isidentifier()
    arr, _, status = load_scene_raster(str(empty))
    assert arr is None and status["error"] == "ZERO_BYTE_FILE"
    arr, _, status = load_scene_raster(str(tmp_path / "missing.tif"))
    assert arr is None and status["error"] == "FILE_NOT_FOUND"


def test_8bit_png_is_scaled_to_reflectance(tmp_path):
    data = (np.random.default_rng(0).random((64, 64, 4)) * 255).astype(np.uint8)
    path = tmp_path / "scene.png"
    Image.fromarray(data).save(path)
    arr, _, status = load_scene_raster(str(path))
    assert status["is_valid"] and arr.shape == (64, 64, 4) and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_16bit_data_is_not_clipped(tmp_path):
    """Regression: the original loader clipped uint16 rasters to 0..255."""
    data = np.full((32, 32, 4), 3000, dtype=np.uint16)
    data[8:12, 8:12, 3] = 9000
    path = tmp_path / "scene.npy"
    np.save(path, data)
    arr, _, status = load_scene_raster(str(path), reflectance_scale=10000)
    assert status["is_valid"]
    assert arr[10, 10, 3] == pytest.approx(0.9) and arr[0, 0, 3] == pytest.approx(0.3)


def test_missing_nir_band_degrades_gracefully(tmp_path):
    data = (np.random.default_rng(1).random((48, 48, 3)) * 3000 + 100).astype(np.uint16)
    path = tmp_path / "rgb_only.npy"
    np.save(path, data)
    arr, _, status = load_scene_raster(str(path), band_names=["red", "green", "blue"])
    assert status["is_valid"] and status["missing_bands"] == ["nir"]
    assert np.array_equal(arr[:, :, 3], arr[:, :, 0])  # red stands in for NIR


def test_dead_band_is_flagged(tmp_path):
    data = (np.random.default_rng(2).random((48, 48, 4)) * 3000 + 100).astype(np.uint16)
    data[:, :, 3] = 0
    path = tmp_path / "dead_nir.npy"
    np.save(path, data)
    _, _, status = load_scene_raster(str(path))
    assert status["is_valid"] and "nir" in status["missing_bands"]


def test_nan_and_inf_are_neutralised(tmp_path):
    data = np.random.default_rng(3).random((32, 32, 4)).astype(np.float32) * 0.1
    data[0, 0, :] = np.nan
    data[1, 1, :] = np.inf
    path = tmp_path / "nan.npy"
    np.save(path, data)
    arr, _, status = load_scene_raster(str(path))
    assert status["is_valid"] and np.isfinite(arr).all()


def test_bundle_ingest_separates_good_from_bad(bundle_dir):
    v = InputBundleValidator(bundle_dir)
    stats = v.validate()
    assert stats["valid_scenes_count"] >= 6
    assert stats["rejected_scenes_count"] >= 1
    assert stats["ais_vessels_in_catalog"] > 0
    assert stats["total_raw_bytes"] > 0
    # ground truth is for scoring only and must never reach the pipeline
    assert all("ground_truth" not in s for s in v.valid_scenes)


def test_path_traversal_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"scenes": [{"id": "X", "file": "../../etc/passwd"}]}))
    v = InputBundleValidator(str(tmp_path))
    v.validate()
    assert v.rejected_scenes and not v.valid_scenes
