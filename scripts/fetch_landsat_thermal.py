"""
Builds a thermal-infrared bundle from Landsat 8/9 Collection-2 Level-2 (ground-side; needs
rasterio + internet).

The companion to scripts/fetch_sentinel2.py, and deliberately the same shape. Where that script
cuts the four 10 m VNIR bands we fly, this one cuts the single 30 m surface-temperature band
(ST_B10, derived from TIRS band 10 at 10.6-11.2 um) plus the QA_PIXEL classification layer, over
the same Arctic areas of interest. The point is a second, physically independent look at the open
water a vessel leaves behind it in pack ice: in VNIR a lead is dark because water reflects ~1 %
where ice reflects 20-30 %; in the thermal it is bright because water cannot fall below its
freezing point of -1.8 C while the ice surface radiates down to -20 C or colder. Same geometry,
different mechanism.

    python scripts/fetch_landsat_thermal.py --survey                    # what exists, no download
    python scripts/fetch_landsat_thermal.py --season winter -o data/real/arctic_thermal

Source is the Microsoft Planetary Computer STAC. Browsing needs no key; the underlying blobs are
read through the public /api/sas/v1/sign token endpoint, which is also keyless and anonymous. The
Element84 and USGS LandsatLook STAC APIs index the same scenes, but their asset hrefs point at
requester-pays S3 and at an ERS-login redirect respectively, so neither can be read without an
account.

Only a window is pulled by HTTP range request from the cloud-optimised GeoTIFFs -- a few MB per
band, not the ~1 GB full scene.

KNOWN LIMIT, measured by --survey: USGS only generates the Collection-2 Level-2 product for
acquisitions with a solar zenith angle under ~76 deg. Above 70 N that gate closes from early
November to late February, so there is NO free surface-temperature product during the polar
night -- exactly the season a VNIR payload most needs help. Level-1 thermal (B10 radiance) is
acquired right through the dark season, but it is only served from requester-pays S3 or behind a
USGS ERS login. Run --survey to see the month-by-month numbers for yourself.

Landsat data courtesy of the U.S. Geological Survey (public domain).
"""

import os
import sys
import json
import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
COLLECTION = "landsat-c2-l2"

# (our name, STAC asset key). ST_B10 is the atmospherically corrected surface temperature;
# QA_PIXEL is the CFMask output and is what lets the ice/water split be made without us
# inventing a threshold.
BANDS = [("surface_temperature", "lwir11"), ("qa_pixel", "qa_pixel")]

# Collection-2 Level-2 surface temperature: kelvin = DN * 0.00341802 + 149.0
ST_SCALE, ST_OFFSET = 0.00341802, 149.0

# QA_PIXEL bit assignments, Collection 2 (Landsat 8/9 OLI-TIRS).
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW, QA_CLEAR, QA_WATER = range(8)

# The same four Arctic AOIs as data/real/arctic_probe, same centres, so the two bundles look at
# the same water. The last two match the spare AOIs in fetch_sentinel2.py.
ARCTIC_AOIS = [
    ("ARC_UTQIAGVIK", "Utqiagvik (Barrow) / Beaufort ice edge", -156.60, 71.35),
    ("ARC_PRUDHOE", "Prudhoe Bay / Beaufort pack ice", -148.50, 70.55),
    ("ARC_KOTZEBUE", "Kotzebue Sound freeze-up", -162.60, 66.90),
    ("ARC_PT_HOPE", "Point Hope / Chukchi marginal ice zone", -166.80, 68.35),
    ("ARC_BERING_STRAIT", "Bering Strait / Diomede Islands", -168.90, 65.78),
    ("ARC_WAINWRIGHT", "Wainwright / Chukchi coast", -160.00, 70.65),
]

# Seasons. "winter" is the one that matters: the ice is cold and consolidated, so the contrast
# across a lead is at its largest, and it is still late enough in the spring for USGS to make an
# L2 product at all. "coincident" brackets the two acquisition dates in arctic_probe.
SEASONS = {
    "winter": ("2025-02-20", "2025-04-20"),
    "freezeup": ("2024-10-10", "2024-11-10"),
    "melt": ("2024-06-15", "2024-08-15"),
    "coincident": ("2024-07-01", "2024-11-05"),
    "year": ("2024-09-01", "2025-08-31"),
}


def stac_search(lon: float, lat: float, start: str, end: str, max_cloud: float = 100.0,
                limit: int = 100):
    body = {
        "collections": [COLLECTION],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
    }
    if max_cloud < 100.0:
        body["query"] = {"eo:cloud_cover": {"lt": max_cloud}}
    req = urllib.request.Request(STAC_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r).get("features", [])


def sign(href: str, tries: int = 6) -> str:
    """
    Planetary Computer read token. Anonymous, no account, ~1 h lifetime.

    The endpoint rate-limits an unauthenticated caller fairly hard (HTTP 429), so back off
    rather than treating it as a missing scene -- otherwise a busy minute looks like a data gap.
    """
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(SIGN_URL + urllib.parse.quote(href, safe=""), timeout=60) as r:
                return json.load(r)["href"]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(4.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def read_window(item, lon: float, lat: float, size: int):
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform

    stack, corners, gsd = [], None, 30.0
    for _, key in BANDS:
        with rasterio.open(sign(item["assets"][key]["href"])) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            col0 = int(min(max(col - size // 2, 0), max(src.width - size, 0)))
            row0 = int(min(max(row - size // 2, 0), max(src.height - size, 0)))
            win = Window(col0, row0, min(size, src.width), min(size, src.height))
            stack.append(src.read(1, window=win))
            if corners is None:
                gsd = float(abs(src.transform.a))
                t = src.window_transform(win)
                w, h = stack[0].shape[1], stack[0].shape[0]
                px = [(0, 0), (w, 0), (w, h), (0, h)]  # ul, ur, lr, ll
                ex, ey = zip(*[t * p for p in px])
                lons, lats = transform(src.crs, "EPSG:4326", list(ex), list(ey))
                corners = {k: [round(lo, 7), round(la, 7)]
                           for k, lo, la in zip(("ul", "ur", "lr", "ll"), lons, lats)}
    return np.stack(stack, axis=-1), corners, gsd


def window_stats(data: np.ndarray):
    """Cloud and class fractions inside the window we actually cut, not over the whole scene."""
    st, qa = data[..., 0], data[..., 1]
    valid = (st > 0) & (((qa >> QA_FILL) & 1) == 0)
    n = max(int(valid.sum()), 1)
    bit = lambda b: (((qa >> b) & 1) == 1) & valid
    obscured = bit(QA_CLOUD) | bit(QA_DILATED) | bit(QA_CIRRUS) | bit(QA_SHADOW)
    return {
        "valid_frac": round(float(valid.mean()), 4),
        "window_cloud_frac": round(float(obscured.sum()) / n, 4),
        "window_water_frac": round(float((bit(QA_WATER) & ~obscured).sum()) / n, 4),
        "window_snowice_frac": round(float((bit(QA_SNOW) & ~obscured).sum()) / n, 4),
    }


def survey(aois, start, end):
    """What acquisitions exist, by month, without downloading a pixel."""
    print(f"Landsat C2 L2 ({COLLECTION}) availability {start} .. {end}\n")
    totals = defaultdict(lambda: [0, 0])
    for sid, desc, lon, lat in aois:
        try:
            items = stac_search(lon, lat, start, end)
        except Exception as e:
            print(f"[{sid}] search failed: {e}")
            continue
        bym = defaultdict(list)
        for it in items:
            p = it["properties"]
            bym[p["datetime"][:7]].append((p.get("eo:cloud_cover"), p.get("view:sun_elevation")))
        print(f"[{sid}] {desc}: {len(items)} L2 acquisitions")
        for m in sorted(bym):
            v = bym[m]
            low = sum(1 for c, _ in v if c is not None and c < 30.0)
            sun = [s for _, s in v if s is not None]
            totals[m][0] += len(v)
            totals[m][1] += low
            span = f"sun {min(sun):5.1f}..{max(sun):5.1f} deg" if sun else ""
            print(f"    {m}  n={len(v):3d}   cloud<30%: {low:3d}   {span}")
        missing = "none" if bym else "NO L2 PRODUCT IN WINDOW"
        if not bym:
            print(f"    {missing}")
        print()
    print("all AOIs combined, by month:")
    for m in sorted(totals):
        n, low = totals[m]
        print(f"    {m}  n={n:3d}   cloud<30%: {low:3d}   usable share {low / max(n, 1):.0%}")
    gap = [f"{y}-{mm:02d}" for y in (2024, 2025) for mm in range(1, 13)
           if f"{y}-{mm:02d}" not in totals
           and start[:7] <= f"{y}-{mm:02d}" <= end[:7]]
    if gap:
        print(f"\nmonths with ZERO L2 product at every AOI: {', '.join(gap)}")
        print("  (USGS only generates Collection-2 Level-2 below ~76 deg solar zenith; above 70 N")
        print("   that gate closes for the polar night. Level-1 B10 is still acquired but is")
        print("   served only from requester-pays S3 or behind a USGS ERS login.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", "-o", default="data/real/arctic_thermal")
    ap.add_argument("--size", type=int, default=1024,
                    help="window edge in pixels (1024 px = 30.7 km at 30 m)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--season", choices=sorted(SEASONS), default="winter",
                    help="date preset; --start/--end override it")
    ap.add_argument("--max-cloud", type=float, default=30.0, help="scene-wide eo:cloud_cover limit")
    ap.add_argument("--max-window-cloud", type=float, default=0.35,
                    help="reject a cut whose own QA_PIXEL says it is this cloudy")
    ap.add_argument("--min-water", type=float, default=0.0,
                    help="require at least this fraction of the window classed water by QA_PIXEL")
    ap.add_argument("--only", nargs="*", help="subset of AOI ids")
    ap.add_argument("--tries", type=int, default=6, help="candidate scenes to attempt per AOI")
    ap.add_argument("--survey", action="store_true",
                    help="report what exists, month by month, and download nothing")
    args = ap.parse_args()

    start, end = SEASONS[args.season]
    start, end = args.start or start, args.end or end
    aois = [a for a in ARCTIC_AOIS if not args.only or a[0] in args.only]

    if args.survey:
        return survey(aois, args.start or SEASONS["year"][0], args.end or SEASONS["year"][1])

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    import tifffile

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    # Keep anything an earlier run put in this bundle for an AOI we are not fetching now: a
    # thermal bundle is usually assembled AOI by AOI, because the date that is clear over one
    # of these places is rarely clear over the next.
    manifest_path = os.path.join(out_dir, "manifest.json")
    scenes = []
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            scenes = [s for s in json.load(f).get("scenes", [])
                      if s["id"] not in {a[0] for a in aois}
                      and os.path.exists(os.path.join(out_dir, s["file"]))]
    for sid, desc, lon, lat in aois:
        print(f"[{sid}] searching {desc}  {start}..{end} ...")
        try:
            items = stac_search(lon, lat, start, end, args.max_cloud)
        except Exception as e:
            print(f"  STAC search failed: {e}")
            continue
        items.sort(key=lambda it: it["properties"].get("eo:cloud_cover") or 100.0)
        done = False
        for item in items[: args.tries]:
            props = item["properties"]
            try:
                data, corners, gsd = read_window(item, lon, lat, args.size)
            except Exception as e:
                print(f"  {item['id']}: read failed ({type(e).__name__}: {str(e)[:80]})")
                continue
            st = window_stats(data)
            if st["valid_frac"] < 0.85:
                print(f"  {item['id']}: {1 - st['valid_frac']:.0%} fill in window, next")
                continue
            if st["window_cloud_frac"] > args.max_window_cloud:
                print(f"  {item['id']}: window {st['window_cloud_frac']:.0%} cloud by QA, next")
                continue
            if st["window_water_frac"] + st["window_snowice_frac"] < args.min_water:
                print(f"  {item['id']}: window water+ice {st['window_water_frac']:.0%}"
                      f"+{st['window_snowice_frac']:.0%} below --min-water, next")
                continue
            fname = f"{sid.lower()}_st.tif"
            tifffile.imwrite(os.path.join(out_dir, fname), data.astype(np.uint16),
                             photometric="minisblack", compression="zlib", planarconfig="contig")
            k = data[..., 0].astype(np.float64) * ST_SCALE + ST_OFFSET
            kv = k[data[..., 0] > 0]
            scenes.append({
                "id": sid, "file": fname,
                "description": f"{desc} ({props['datetime'][:10]}, {item['id']})",
                "shutter_time": props["datetime"], "gsd_meters": gsd, "corners": corners,
                "center_lat": lat, "center_lon": lon,
                "temperature_scale": ST_SCALE, "temperature_offset": ST_OFFSET,
                "temperature_units": "kelvin",
                "window": st,
                "source": {"stac_item": item["id"], "collection": COLLECTION,
                           "cloud_cover_pct": props.get("eo:cloud_cover"),
                           "sun_elevation_deg": props.get("view:sun_elevation"),
                           "platform": props.get("platform"),
                           "wrs_path": props.get("landsat:wrs_path"),
                           "wrs_row": props.get("landsat:wrs_row")},
            })
            print(f"  {item['id']}: scene cloud {props.get('eo:cloud_cover'):.1f}%, "
                  f"window cloud {st['window_cloud_frac']:.0%}, water {st['window_water_frac']:.0%}, "
                  f"ice {st['window_snowice_frac']:.0%}, "
                  f"T {kv.min() - 273.15:.1f}..{kv.max() - 273.15:.1f} C -> {fname}")
            done = True
            break
        if not done:
            print("  no usable scene found")

    manifest = {
        "bundle_version": "2.0",
        "satellite": "Landsat 8/9 (thermal reference, not a MOBIUS-1 proxy)",
        "sensor": "TIRS band 10 via Collection-2 Level-2 surface temperature",
        "gsd_meters": 30.0,
        "native_gsd_meters": 100.0,
        "resampling_note": "ST_B10 is delivered on the 30 m OLI grid; TIRS band 10 is sampled at "
                           "100 m and cubic-convolved up, so the effective resolution is ~100 m.",
        "bands": [b for b, _ in BANDS],
        "temperature_scale": ST_SCALE, "temperature_offset": ST_OFFSET,
        "attribution": "Landsat 8/9 Collection-2 Level-2, courtesy of the U.S. Geological Survey, "
                       "via the Microsoft Planetary Computer",
        "scenes": scenes,
    }
    scenes.sort(key=lambda s: [a[0] for a in ARCTIC_AOIS].index(s["id"]))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] {len(scenes)} scenes written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
