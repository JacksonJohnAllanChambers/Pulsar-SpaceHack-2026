"""
Builds a real-imagery input bundle from Sentinel-2 L2A (ground-side; needs rasterio + internet).

For each area of interest the least-cloudy recent scene is found through the public Element84
Earth Search STAC API, and only a window of the four 10 m bands we fly (B04, B03, B02, B08) is
read by HTTP range request from the public cloud-optimised GeoTIFFs -- tens of MB, not the
~800 MB full tile. The result is an ordinary applet bundle: 16-bit 4-band rasters + manifest.json
with corner coordinates, so the onboard code needs no GDAL.

    python scripts/fetch_sentinel2.py --output data/real/s2_bundle --size 2048

Sentinel-2 data: Copernicus programme, free and open licence ("contains modified Copernicus
Sentinel data"). There is no matching open AIS archive, so the bundle ships an empty AIS
catalogue and every vessel will classify as DARK_VESSEL.
"""

import os
import sys
import json
import argparse
import urllib.request

import numpy as np

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = [("red", "red"), ("green", "green"), ("blue", "blue"), ("nir", "nir")]  # (our name, STAC asset key)

# (id, description, lon, lat) -- anchorages and traffic lanes that are reliably busy
AOIS = [
    ("S2_GIBRALTAR", "Strait of Gibraltar / Algeciras Bay anchorage", -5.38, 36.08),
    ("S2_SUEZ", "Gulf of Suez southern anchorage", 32.58, 29.82),
    ("S2_LONGBEACH", "Los Angeles / Long Beach outer anchorage", -118.17, 33.68),
    ("S2_HALIFAX", "Halifax harbour approaches", -63.50, 44.57),
    ("S2_DOVER", "Dover Strait traffic separation scheme", 1.45, 51.03),
    ("S2_SINGAPORE", "Singapore Strait eastern anchorage", 104.05, 1.27),
]

# US coastal AOIs. These exist so that every scene can be paired with the same day's public NOAA
# Marine Cadastre AIS (scripts/fetch_noaa_ais.py): the only configuration in which the imagery is
# geolocated independently of AIS, so a position match is evidence rather than an artefact of how
# the chip was cut. NOAA publishes Jul-Dec 2024, hence the default date window for this set.
US_AOIS = [
    ("S2_LONGBEACH", "Los Angeles / Long Beach outer anchorage", -118.17, 33.68),
    ("S2_NEWYORK", "New York / Ambrose Channel approaches", -73.95, 40.47),
    ("S2_NORFOLK", "Chesapeake Bay mouth / Norfolk approaches", -76.02, 36.95),
    ("S2_GALVESTON", "Galveston / Houston ship channel anchorage", -94.70, 29.30),
    ("S2_TAMPA", "Tampa Bay approaches", -82.75, 27.60),
    ("S2_SAVANNAH", "Savannah / Tybee Roads anchorage", -80.80, 31.98),
    ("S2_CHARLESTON", "Charleston harbour approaches", -79.75, 32.68),
    ("S2_SANFRANCISCO", "San Francisco Bay approaches / pilot area", -122.60, 37.75),
    ("S2_PUGETSOUND", "Puget Sound / Admiralty Inlet", -122.60, 48.10),
    ("S2_MIAMI", "Miami / Fort Lauderdale anchorage", -80.08, 25.75),
    ("S2_MISSISSIPPI", "Mississippi River delta / Southwest Pass", -89.30, 28.95),
    ("S2_BOSTON", "Boston harbour approaches", -70.85, 42.33),
    ("S2_DELAWARE", "Delaware Bay entrance", -75.00, 38.80),
    ("S2_CORPUS", "Corpus Christi / Aransas Pass anchorage", -97.00, 27.80),
    ("S2_HONOLULU", "Honolulu harbour approaches", -157.90, 21.28),
    ("S2_SANDIEGO", "San Diego approaches", -117.25, 32.65),
]

REGIONS = {"world": AOIS, "us": US_AOIS}
# NOAA Marine Cadastre AIS coverage: Jul-Dec 2024 (earlier 2024 months and 2025 return 404)
US_WINDOW = ("2024-07-01", "2024-12-31")


def stac_search(lon: float, lat: float, start: str, end: str, max_cloud: float, limit: int = 12):
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": limit,
    }
    req = urllib.request.Request(STAC_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("features", [])


def read_window(item, lon: float, lat: float, size: int):
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform

    stack, corners, gsd = [], None, 10.0
    for _, key in BANDS:
        href = item["assets"][key]["href"]
        with rasterio.open(href) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            col0 = int(min(max(col - size // 2, 0), src.width - size))
            row0 = int(min(max(row - size // 2, 0), src.height - size))
            win = Window(col0, row0, size, size)
            stack.append(src.read(1, window=win))
            if corners is None:
                gsd = float(abs(src.transform.a))
                t = src.window_transform(win)
                px = [(0, 0), (size, 0), (size, size), (0, size)]  # ul, ur, lr, ll
                ex, ey = zip(*[t * p for p in px])
                lons, lats = transform(src.crs, "EPSG:4326", list(ex), list(ey))
                corners = {k: [round(lo, 7), round(la, 7)] for k, lo, la in zip(("ul", "ur", "lr", "ll"), lons, lats)}
    return np.stack(stack, axis=-1), corners, gsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default="data/real/s2_bundle")
    ap.add_argument("--size", type=int, default=2048, help="window edge in pixels (2048 px = 20.5 km)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-cloud", type=float, default=5.0)
    ap.add_argument("--only", nargs="*", help="subset of AOI ids")
    ap.add_argument("--region", choices=sorted(REGIONS), default="world",
                    help="'us' picks AOIs inside NOAA AIS coverage and defaults to its Jul-Dec 2024 window")
    args = ap.parse_args()

    aois = REGIONS[args.region]
    default_start, default_end = US_WINDOW if args.region == "us" else ("2025-04-01", "2026-09-15")
    args.start = args.start or default_start
    args.end = args.end or default_end

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    import tifffile

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    scenes = []
    for sid, desc, lon, lat in aois:
        if args.only and sid not in args.only:
            continue
        print(f"[{sid}] searching {desc} ...")
        try:
            items = stac_search(lon, lat, args.start, args.end, args.max_cloud)
        except Exception as e:
            print(f"  STAC search failed: {e}")
            continue
        done = False
        for item in items:
            props = item["properties"]
            try:
                data, corners, gsd = read_window(item, lon, lat, args.size)
            except Exception as e:
                print(f"  {item['id']}: read failed ({type(e).__name__}: {str(e)[:80]})")
                continue
            nodata_frac = float((data == 0).all(axis=-1).mean())
            if nodata_frac > 0.15:
                print(f"  {item['id']}: {nodata_frac:.0%} no-data in window, trying next")
                continue
            fname = f"{sid.lower()}.tif"
            tifffile.imwrite(os.path.join(out_dir, fname), data.astype(np.uint16), photometric="minisblack",
                             compression="zlib", planarconfig="contig")
            # Processing baseline >= 04.00 adds 1000 DN; Earth Search says whether it already removed it
            offset = 0.0 if props.get("earthsearch:boa_offset_applied", False) else (
                -1000.0 if float(str(props.get("s2:processing_baseline", "0")).replace(",", ".")) >= 4.0 else 0.0)
            scenes.append({
                "id": sid, "file": fname, "description": f"{desc} ({props['datetime'][:10]}, {item['id']})",
                "shutter_time": props["datetime"], "gsd_meters": gsd, "corners": corners,
                "center_lat": lat, "center_lon": lon, "reflectance_offset": offset,
                "source": {"stac_item": item["id"], "cloud_cover_pct": props.get("eo:cloud_cover"),
                           "platform": props.get("platform")},
            })
            print(f"  {item['id']}: cloud {props.get('eo:cloud_cover'):.1f}%, offset {offset:.0f}, "
                  f"median DN {np.median(data, axis=(0, 1)).astype(int).tolist()} -> {fname}")
            done = True
            break
        if not done:
            print("  no usable scene found")

    manifest = {
        "bundle_version": "2.0", "satellite": "Sentinel-2 (proxy for MOBIUS-1)", "sensor": "MSI L2A",
        "gsd_meters": 10.0, "reflectance_scale": 10000, "bands": [b for b, _ in BANDS],
        "attribution": "Contains modified Copernicus Sentinel data, via Element84 Earth Search / AWS Open Data",
        "scenes": scenes,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "ais_catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "no open AIS archive matches these scenes (run scripts/fetch_noaa_ais.py for a --region us bundle)", "vessels": []}, f, indent=2)
    print(f"[DONE] {len(scenes)} scenes written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
