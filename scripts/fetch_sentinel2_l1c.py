"""
Rebuilds an existing Sentinel-2 L2A bundle from the matching **L1C** products (ground-side).

Why: L2A is surface reflectance -- atmospherically corrected on the ground with far more compute
than a satellite has. The sensor sees top-of-atmosphere radiance: the same scene plus path
radiance (haze), strongest in blue and weakest in NIR. The organisers' track guide warns against
validating on cleaner data than the satellite captures, so this fetches the uncorrected product
for the *same acquisitions*.

L1C and L2A share the MGRS tile grid, so every scene here is cut from the identical pixel window:
same AIS, same shutter time, same geolocation, and hand labels transfer by position
(scripts/transfer_labels.py). The only thing that changes is the atmosphere.

Source: Google's public mirror (gs://gcp-public-data-sentinel-2), anonymous HTTPS, JPEG 2000.
Earth Search also lists L1C but on a requester-pays bucket. JP2 is not cloud-optimised, so a
window read still moves a large share of each ~100 MB band; expect a few GB for 16 scenes.

    python scripts/fetch_sentinel2_l1c.py --from data/real/s2_us_bundle -o data/real/s2_us_l1c_bundle

Contains modified Copernicus Sentinel data.
"""

import os
import re
import sys
import json
import shutil
import argparse
import urllib.parse
import urllib.request

import numpy as np

GCS = "https://storage.googleapis.com"
BUCKET = "gcp-public-data-sentinel-2"
BANDS = [("red", "B04"), ("green", "B03"), ("blue", "B02"), ("nir", "B08")]
ITEM = re.compile(r"^(S2[ABC])_(\d{1,2})([A-Z])([A-Z]{2})_(\d{8})_(\d+)_L2A$")


def gcs_list(prefix: str, delimiter: str = ""):
    """Anonymous object listing; returns (object names, sub-prefixes)."""
    names, prefixes, token = [], [], None
    while True:
        query = {"prefix": prefix, "fields": "items(name),prefixes,nextPageToken"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["pageToken"] = token
        url = f"{GCS}/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        names += [i["name"] for i in page.get("items", [])]
        prefixes += page.get("prefixes", [])
        token = page.get("nextPageToken")
        if not token:
            return names, prefixes


def find_l1c_products(stac_item: str):
    """Every L1C product of the same platform, tile and day as an Earth Search L2A item id."""
    m = ITEM.match(stac_item)
    if not m:
        raise ValueError(f"unrecognised Earth Search item id: {stac_item}")
    platform, zone, band, square, day, _ = m.groups()
    _, products = gcs_list(f"tiles/{int(zone):02d}/{band}/{square}/{platform}_MSIL1C_{day}", delimiter="/")
    return sorted(products)


def band_urls(product_prefix: str):
    names, _ = gcs_list(product_prefix + "GRANULE/")
    urls = {}
    for _, code in BANDS:
        hits = [n for n in names if n.endswith(f"_{code}.jp2") and "/IMG_DATA/" in n]
        if len(hits) != 1:
            raise RuntimeError(f"expected one {code}.jp2 under {product_prefix}, found {len(hits)}")
        urls[code] = f"{GCS}/{BUCKET}/{urllib.parse.quote(hits[0])}"
    return urls


def radiometry(product_prefix: str):
    """(quantification, additive offset in DN) from the product metadata; baseline >= 04.00 adds -1000."""
    url = f"{GCS}/{BUCKET}/{urllib.parse.quote(product_prefix + 'MTD_MSIL1C.xml')}"
    with urllib.request.urlopen(url, timeout=60) as r:
        xml = r.read().decode("utf-8", "replace")
    quant = re.search(r"<QUANTIFICATION_VALUE[^>]*>([\d.]+)<", xml)
    offsets = {float(v) for v in re.findall(r"<RADIO_ADD_OFFSET[^>]*>(-?[\d.]+)<", xml)}
    if len(offsets) > 1:
        raise RuntimeError(f"per-band offsets differ ({offsets}); the bundle format carries one per scene")
    return (float(quant.group(1)) if quant else 10000.0), (offsets.pop() if offsets else 0.0)


def read_window(urls, corners, size_hw):
    """The window whose upper-left pixel corner is corners['ul'], on the L1C grid."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform

    height, width = size_hw
    stack, origin = [], None
    for _, code in BANDS:
        with rasterio.open("/vsicurl/" + urls[code]) as src:
            if origin is None:
                xs, ys = transform("EPSG:4326", src.crs, [corners["ul"][0]], [corners["ul"][1]])
                inv = ~src.transform
                col, row = inv * (xs[0], ys[0])
                origin = (int(round(col)), int(round(row)))
                # The manifest stores corners to 1e-7 deg (~1 cm), so this must land on a pixel corner
                if abs(col - origin[0]) > 0.05 or abs(row - origin[1]) > 0.05:
                    raise RuntimeError(f"window origin is off-grid by ({col - origin[0]:.3f}, {row - origin[1]:.3f}) px")
            stack.append(src.read(1, window=Window(origin[0], origin[1], width, height)))
    return np.stack(stack, axis=-1), origin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", default="data/real/s2_us_bundle", help="existing L2A bundle to mirror")
    ap.add_argument("--output", "-o", default="data/real/s2_us_l1c_bundle")
    ap.add_argument("--only", nargs="*", help="subset of scene ids")
    args = ap.parse_args()

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".jp2")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", str(256 * 1024 * 1024))
    import tifffile
    from applet.utils.image_io import load_scene_raster  # noqa: F401  (import check before a long download)

    src_dir, out_dir = os.path.abspath(args.source), os.path.abspath(args.output)
    with open(os.path.join(src_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(out_dir, exist_ok=True)

    # Resumable: a scene already fetched into this bundle is kept, not downloaded again
    done = {}
    if os.path.exists(os.path.join(out_dir, "manifest.json")):
        with open(os.path.join(out_dir, "manifest.json"), "r", encoding="utf-8") as f:
            done = {s["id"]: s for s in json.load(f).get("scenes", [])
                    if os.path.exists(os.path.join(out_dir, s["file"]))}

    scenes = []
    for scene in manifest["scenes"]:
        sid = scene["id"]
        if sid in done and not (args.only and sid in args.only):
            scenes.append(done[sid])
            continue
        if args.only and sid not in args.only:
            continue
        item = scene["source"]["stac_item"]
        l2a = tifffile.imread(os.path.join(src_dir, scene["file"]))
        print(f"[{sid}] {item}")
        try:
            products = find_l1c_products(item)
        except Exception as e:
            print(f"  listing failed: {e}")
            continue
        best = None
        for product in products:
            try:
                data, origin = read_window(band_urls(product), scene["corners"], l2a.shape[:2])
                quant, offset = radiometry(product)
            except Exception as e:
                print(f"  {product.split('/')[-2]}: {type(e).__name__}: {str(e)[:100]}")
                continue
            # Same acquisition, same grid => NIR is almost untouched by the atmosphere. A low value
            # means the wrong product or a shifted window, and would silently void label transfer.
            valid = (l2a[..., 3] > 0) & (data[..., 3] > 0)
            corr = float(np.corrcoef(l2a[..., 3][valid].ravel(), data[..., 3][valid].ravel())[0, 1]) if valid.sum() > 1000 else 0.0
            print(f"  {product.split('/')[-2]}: origin {origin}, NIR correlation with L2A {corr:.4f}")
            if best is None or corr > best[0]:
                best = (corr, product, data, quant, offset)
        if best is None or best[0] < 0.9:
            print("  no L1C product matches the L2A window; scene skipped")
            continue
        corr, product, data, quant, offset = best
        tifffile.imwrite(os.path.join(out_dir, scene["file"]), data.astype(np.uint16), photometric="minisblack",
                         compression="zlib", planarconfig="contig")
        l1c_scene = dict(scene)
        l1c_scene["reflectance_offset"] = offset
        l1c_scene["description"] = scene.get("description", "").replace("_L2A", "_L1C")
        l1c_scene["source"] = {**scene["source"], "l1c_product": product.split("/")[-2], "level": "L1C",
                               "nir_correlation_with_l2a": round(corr, 4)}
        scenes.append(l1c_scene)
        toa = (np.median(data, axis=(0, 1)) + offset) / quant
        boa = (np.median(l2a, axis=(0, 1)) + scene.get("reflectance_offset", 0.0)) / manifest.get("reflectance_scale", 10000)
        print(f"  median reflectance R,G,B,NIR  TOA {np.round(toa, 4).tolist()}  vs surface {np.round(boa, 4).tolist()}")

    out_manifest = {**manifest, "sensor": "MSI L1C (top of atmosphere, no atmospheric correction)",
                    "attribution": "Contains modified Copernicus Sentinel data, via Google Cloud public datasets",
                    "scenes": scenes}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2)
    for extra in ("ais_catalog.json", "known_structures.json"):
        if os.path.exists(os.path.join(src_dir, extra)):
            shutil.copy2(os.path.join(src_dir, extra), os.path.join(out_dir, extra))
    print(f"[DONE] {len(scenes)} L1C scenes written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
