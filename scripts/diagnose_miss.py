"""
Why was a specific AIS broadcaster not detected? Walk the cascade and find the stage that dropped it.

A recall number tells you how many you missed. It does not tell you WHERE in the pipeline the miss
happened, and those are completely different repairs: a screener mask is a threshold, a CFAR non-fire
is a sensitivity problem, and a physics-score rejection is a scoring problem. Guessing wrong wastes a
day tuning the wrong stage -- which is exactly what happened with LE BOREAL, where the obvious suspect
(the `round_and_big` hard cap in `_physics_score`) turned out to be innocent: `scripts/stationary_study.py`
sweeps an escape hatch for that cap across five thresholds and recovers nothing, because the contact
never reached scoring at all.

This script takes an MMSI, dead-reckons it to the shutter the same way the AIS correlator does, and
reports what every stage of the cascade saw at that pixel.

    python scripts/diagnose_miss.py -i data/real/svalbard_poc --mmsi 578000500
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.pipelines.ais_correlator import AISKinematicCorrelator, _parse_iso  # noqa: E402
from applet.pipelines.quality_screener import meters_to_px  # noqa: E402
from applet.runner import run_pass  # noqa: E402


def patch_stats(a, y, x, half, label):
    """Local window around a pixel, guarded against the scene edge."""
    h, w = a.shape[:2]
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    win = a[y0:y1, x0:x1]
    if win.size == 0:
        return f"    {label:<22} (outside the raster)"
    return (f"    {label:<22} median {np.median(win):+.4f}  max {np.max(win):+.4f}  "
            f"min {np.min(win):+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--mmsi", type=int, required=True)
    ap.add_argument("--window-m", type=float, default=300.0,
                    help="half-width of the window summarised around the predicted position")
    args = ap.parse_args()

    cfg = AppletConfig()
    cfg.downlink.write_queues = False
    out = os.path.join(os.environ.get("TEMP", "."), "diagnose_miss_out")
    ctx, tel, dl = run_pass(args.input, out, cfg, keep_rasters=True)

    catalog = ctx.get("ais_catalog", [])
    ship = next((s for s in catalog if int(s.get("mmsi", -1)) == args.mmsi), None)
    if ship is None:
        print(f"[!] MMSI {args.mmsi} is not in this bundle's AIS catalog")
        return 1

    print()
    print("=" * 78)
    print(f"  {ship.get('name') or args.mmsi}  (MMSI {args.mmsi})")
    print("=" * 78)
    print(f"  reported   {ship.get('sog_knots')} kn, cog {ship.get('cog_deg')}, "
          f"length {ship.get('length_m')} m, fix age {ship.get('fix_age_s')} s")

    # What did the pipeline itself say about it?
    for g in ctx.get("ais_not_observed", []):
        if int(g.get("mmsi", -1)) == args.mmsi:
            print(f"  pipeline verdict: {g.get('reason')}")

    correlator = AISKinematicCorrelator(cfg)
    for scene in ctx.get("screened_scenes", []):
        georef = scene.get("georef")
        if georef is None:
            continue
        shutter = _parse_iso(scene.get("shutter_time"))
        lat, lon = correlator._predict(ship, shutter)
        x, y = georef.lonlat_to_pixel(lon, lat)
        arr = scene.get("array")
        if arr is None:
            continue
        h, w = arr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            continue

        ix, iy = int(round(x)), int(round(y))
        gsd = scene["gsd_m"]
        half = max(2, meters_to_px(args.window_m, gsd, minimum=2))

        print(f"\n  --- {scene['id']} --- predicted pixel ({ix}, {iy}), {gsd} m GSD, "
              f"window +-{args.window_m:.0f} m")

        masks = {"sea_mask": "in searchable sea", "land_mask": "in LAND mask",
                 "cloud_mask": "in CLOUD mask", "nodata_mask": "in NODATA",
                 "ice_mask": "in ICE mask", "det_mask": "in CFAR detections"}
        for key, label in masks.items():
            m = scene.get(key)
            if m is None:
                continue
            here = bool(m[iy, ix])
            near = bool(m[max(0, iy - half):iy + half + 1, max(0, ix - half):ix + half + 1].any())
            print(f"    {label:<22} at pixel: {str(here):<5}   anywhere in window: {near}")

        print()
        print(patch_stats(arr[:, :, 3], iy, ix, half, "NIR reflectance"))
        print(patch_stats(arr[:, :, 0], iy, ix, half, "red reflectance"))
        zmap = scene.get("zmap")
        if zmap is not None:
            print(patch_stats(zmap, iy, ix, half, "CFAR z-score"))
            peak = np.max(zmap[max(0, iy - half):iy + half + 1, max(0, ix - half):ix + half + 1])
            print(f"\n    peak CFAR z in window: {peak:.2f}   (gate is cfar_k_sigma = "
                  f"{cfg.detection.cfar_k_sigma})")
            print("    -> " + ("CFAR DID fire nearby; the miss is downstream (linking, area, "
                               "physics score, CNN)." if peak >= cfg.detection.cfar_k_sigma
                               else "CFAR NEVER FIRED here. The miss is at detection sensitivity, "
                                    "not at scoring."))

        # Nearest thing we did find, for context.
        best, bestd = None, 1e18
        for t in ctx["classified_targets"]:
            if t.get("scene_id") != scene["id"]:
                continue
            px = t.get("apex_px") or [None, None]
            if px[0] is None:
                continue
            d = ((px[0] - ix) ** 2 + (px[1] - iy) ** 2) ** 0.5
            if d < bestd:
                best, bestd = t, d
        if best is not None:
            print(f"\n    nearest contact we DID report: {best['detection_id']} at "
                  f"{bestd * gsd:.0f} m, physics {best.get('physics_score')}, "
                  f"class {best.get('classification')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
