"""
Synthetic input-bundle generator.

Writes a self-contained bundle (16-bit 4-band GeoTIFF-style rasters + manifest.json +
ais_catalog.json) covering the conditions the applet must survive: clear ocean, heavy
cloud, broken cloud, rough sea, coast with islets, sunglint, a missing NIR band and a
corrupt file. Fully seeded: the same command always produces the same bundle.

Ground truth is stored in the manifest under "ground_truth"; the onboard validator
strips that key before the pipeline sees the scene, so it is only available for scoring.
"""

import os
import sys
import json
import math
import argparse
from typing import List, Dict, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.scene_synth import SceneSpec, Vessel, render_scene, random_vessel, random_spec, to_uint16  # noqa: E402
from applet.utils.geo import SceneGeoreference, project_dead_reckoning  # noqa: E402

SHUTTER = "2026-09-19T14:32:10Z"
AIS_FIX = "2026-09-19T14:30:40Z"  # 90 s before shutter
AIS_DT_H = 90.0 / 3600.0
GSD = 4.75


def write_raster(path: str, reflectance: np.ndarray, bands: int = 4) -> None:
    data = to_uint16(reflectance)[:, :, :bands]
    try:
        import tifffile

        tifffile.imwrite(path, data, photometric="minisblack", compression="zlib", planarconfig="contig")
    except ImportError:
        np.save(os.path.splitext(path)[0] + ".npy", data)


def fixed_vessels(kind: str) -> List[Vessel]:
    if kind == "clear":
        return [
            Vessel(300, 260, 315, 18.0, 190, 30, palette=4, ais="on", name="ATLANTIC CARRIER"),
            Vessel(720, 330, 80, 9.0, 34, 8, palette=0, ais="dark", name=""),
            Vessel(520, 700, 200, 14.0, 95, 15, palette=2, ais="spoof_course", name="NORDIC TRADER"),
            Vessel(180, 820, 20, 24.0, 16, 4.5, palette=0, ais="dark", name=""),
            Vessel(850, 820, 140, 0.0, 240, 40, palette=1, ais="on", name="HALIFAX SPIRIT"),
        ]
    if kind == "rough":
        return [
            Vessel(400, 380, 250, 13.0, 140, 22, palette=1, ais="on", name="GRAND BANKS"),
            Vessel(760, 640, 30, 8.0, 28, 7, palette=0, ais="dark", name=""),
            Vessel(200, 760, 100, 16.0, 70, 12, palette=3, ais="spoof_static", name="SEA WANDERER"),
        ]
    if kind == "coast":
        return [
            Vessel(600, 300, 170, 11.0, 110, 18, palette=2, ais="on", name="COASTAL STAR"),
            Vessel(700, 640, 300, 7.0, 24, 6, palette=0, ais="dark", name=""),
            Vessel(450, 800, 60, 0.0, 60, 11, palette=1, ais="dark", name=""),
            Vessel(820, 450, 10, 20.0, 18, 5, palette=0, ais="on", name="PILOT 7"),
        ]
    if kind == "glint":
        return [
            Vessel(350, 500, 45, 15.0, 160, 26, palette=4, ais="on", name="PACIFIC DAWN"),
            Vessel(700, 300, 220, 10.0, 40, 9, palette=0, ais="dark", name=""),
        ]
    return []


def build_ais(scene_id: str, vessels: List[Dict[str, Any]], georef: SceneGeoreference, mmsi_base: int) -> List[Dict[str, Any]]:
    out = []
    for i, v in enumerate(vessels):
        if v["ais"] == "dark":
            continue
        lon, lat = georef.pixel_to_lonlat(v["x"], v["y"])
        # where the ship really was at fix time
        fix_lat, fix_lon = project_dead_reckoning(lat, lon, v["speed_knots"], (v["heading_deg"] + 180) % 360, AIS_DT_H)
        cog, sog = v["heading_deg"], v["speed_knots"]
        if v["ais"] == "spoof_course":
            cog = (cog + 105.0) % 360
        elif v["ais"] == "spoof_static":
            sog, fix_lat, fix_lon = 0.0, lat, lon
        out.append({
            "mmsi": mmsi_base + i, "name": v["name"] or f"VESSEL_{mmsi_base + i}", "timestamp": AIS_FIX,
            "latitude": round(fix_lat, 6), "longitude": round(fix_lon, 6),
            "sog_knots": round(sog, 1), "cog_deg": round(cog, 1), "source_scene": scene_id,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic satellite input bundle")
    parser.add_argument("--output", "-o", default="data/sample_bundle", help="Target output folder")
    parser.add_argument("--full-swath", action="store_true", help="Add a 4096x4096 (19.4 km) benchmark scene")
    parser.add_argument("--size", type=int, default=1024, help="Edge length of the standard scenes in pixels")
    parser.add_argument("--random", type=int, default=0, metavar="N",
                        help="Instead of the curated scenes, write N randomly drawn scenes (held-out evaluation)")
    parser.add_argument("--seed", type=int, default=777, help="Seed for --random (training used 2026)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    for stale in os.listdir(out_dir):
        if stale.lower().endswith((".png", ".tif", ".npy")):
            os.remove(os.path.join(out_dir, stale))
    print(f"[INFO] Generating synthetic bundle in {out_dir}")

    n = args.size
    k = n / 1024.0

    def scaled(vs: List[Vessel]) -> List[Vessel]:
        for v in vs:
            v.x, v.y = v.x * k, v.y * k
        return vs

    rng = np.random.default_rng(606)
    cloud_spec = SceneSpec(n, n, GSD, seed=16, wind=0.3, cloud_cover=0.28)
    cloud_spec.vessels = [random_vessel(rng, cloud_spec) for _ in range(7)]

    plan = [
        ("SCENE_01", "scene_01_clear.tif", 44.40, -63.20, "Clear open ocean, mixed traffic",
         SceneSpec(n, n, GSD, seed=11, wind=0.25, vessels=scaled(fixed_vessels("clear"))), 4),
        ("SCENE_02", "scene_02_overcast.tif", 44.40, -63.10, "Heavy cloud deck (should be rejected)",
         SceneSpec(n, n, GSD, seed=12, wind=0.3, cloud_cover=0.9, vessels=scaled(fixed_vessels("glint"))), 4),
        ("SCENE_03", "scene_03_rough_sea.tif", 44.35, -63.20, "Gale: dense whitecaps",
         SceneSpec(n, n, GSD, seed=13, wind=0.95, vessels=scaled(fixed_vessels("rough"))), 4),
        ("SCENE_04", "scene_04_corrupted.tif", 44.35, -63.10, "Truncated file", None, 4),
        ("SCENE_05", "scene_05_coast.tif", 44.55, -63.45, "Coastline, surf and islets",
         SceneSpec(n, n, GSD, seed=15, wind=0.35, coast=True, islets=5, vessels=scaled(fixed_vessels("coast"))), 4),
        ("SCENE_06", "scene_06_broken_cloud.tif", 44.30, -63.20, "Broken cumulus over traffic", cloud_spec, 4),
        ("SCENE_07", "scene_07_sunglint.tif", 44.30, -63.10, "Sunglint gradient",
         SceneSpec(n, n, GSD, seed=17, wind=0.5, glint=0.9, vessels=scaled(fixed_vessels("glint"))), 4),
        ("SCENE_08", "scene_08_no_nir.tif", 44.25, -63.20, "NIR band lost (3-band file)",
         SceneSpec(n, n, GSD, seed=18, wind=0.3, vessels=scaled(fixed_vessels("rough"))), 3),
    ]
    if args.random:
        eval_rng = np.random.default_rng(args.seed)
        plan = []
        for i in range(args.random):
            spec = random_spec(eval_rng, n, GSD)
            plan.append((f"EVAL_{i + 1:03d}", f"eval_{i + 1:03d}.tif", 44.0 + 0.05 * (i // 10), -63.0 + 0.07 * (i % 10),
                         f"wind {spec.wind:.2f}, cloud {spec.cloud_cover:.2f}, glint {spec.glint:.2f}"
                         f"{', coast' if spec.coast else ''}", spec, 4))
    if args.full_swath:
        big = SceneSpec(4096, 4096, GSD, seed=99, wind=0.4, cloud_cover=0.12, coast=True, islets=6)
        big_rng = np.random.default_rng(99)
        big.vessels = [random_vessel(big_rng, big, margin_px=80) for _ in range(24)]
        plan.append(("SCENE_99", "scene_99_full_swath.tif", 44.10, -63.00, "Full 19.4 km swath benchmark", big, 4))

    scenes_manifest, ais_all = [], []
    for idx, (sid, fname, lat, lon, desc, spec, bands) in enumerate(plan):
        entry: Dict[str, Any] = {"id": sid, "file": fname, "center_lat": lat, "center_lon": lon,
                                 "description": desc, "shutter_time": SHUTTER}
        path = os.path.join(out_dir, fname)
        if spec is None:
            with open(path, "wb") as f:
                f.write(b"II*\x00\x08\x00\x00\x00CORRUPT_DOWNLINK_BUFFER" + bytes(range(64)))
            scenes_manifest.append(entry)
            continue

        refl, truth = render_scene(spec)
        write_raster(path, refl, bands)
        if not os.path.exists(path):  # tifffile absent -> .npy fallback
            entry["file"] = os.path.splitext(fname)[0] + ".npy"
        entry["bands"] = ["red", "green", "blue", "nir"][:bands]
        entry["ground_truth"] = truth
        scenes_manifest.append(entry)

        georef = SceneGeoreference(entry, spec.width, spec.height, GSD)
        ais_all += build_ais(sid, truth["vessels"], georef, 316000000 + idx * 100)
        print(f"  {sid}: {spec.width}x{spec.height}, {len(truth['vessels'])} vessels, "
              f"cloud {truth['cloud_fraction'] * 100:.0f}%, land {truth['land_fraction'] * 100:.0f}%")

    # A broadcaster that is not where it claims to be (clear water in SCENE_01)
    ghost_ref = SceneGeoreference(scenes_manifest[0], n, n, GSD)
    glon, glat = ghost_ref.pixel_to_lonlat(0.78 * n, 0.12 * n)
    if not args.random:
        ais_all.append({"mmsi": 316999001, "name": "GHOST TRANSPONDER", "timestamp": AIS_FIX, "latitude": round(glat, 6),
                        "longitude": round(glon, 6), "sog_knots": 0.0, "cog_deg": 0.0, "source_scene": "SCENE_01"})

    with open(os.path.join(out_dir, "ais_catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": AIS_FIX, "orbital_pass_id": "PASS_001", "vessels": ais_all}, f, indent=2)

    manifest = {
        "bundle_version": "2.0", "satellite": "MOBIUS-1", "sensor": "Simera-HyperScape100-Proxy",
        "gsd_meters": GSD, "reflectance_scale": 10000, "bands": ["red", "green", "blue", "nir"],
        "shutter_time": SHUTTER, "scenes": scenes_manifest,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] {len(plan)} scenes + {len(ais_all)} AIS records written.")


if __name__ == "__main__":
    main()
