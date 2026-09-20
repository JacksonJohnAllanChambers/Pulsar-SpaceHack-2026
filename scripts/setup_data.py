"""
One-command data setup for a fresh clone. Nothing large lives in git; this rebuilds all of it.

    python scripts/setup_data.py                 # synthetic bundles only (no downloads, ~1 min)
    python scripts/setup_data.py --all           # + every real dataset below (~2.5 GB of downloads)
    python scripts/setup_data.py --benchmarks    # just the three AIS-scored benchmarks the docs quote
    python scripts/setup_data.py --all --data-root D:\\spacehack-data     # keep the big files on another drive

Steps (each is skipped if its output already exists; --force redoes it):

  --synthetic   data/sample_bundle (demo), data/eval_bundle (seed 777, tuning), data/heldout_bundle (seed 4242, test)
  --sentinel2   data/real/s2_bundle        five 20 km Sentinel-2 L2A scenes, ~95 MB, public AWS, no login
  --sen2ms      data/real/sen2ms           SEN2MS Vessel BBoxes zip (565 MB, Zenodo, CC-BY-4.0) + the real-AIS chip bundle
  --noaa        data/real/s2_ais_bundle    Long Beach 2024-11-08 scene + that day's NOAA AIS (363 MB, public domain)

The three benchmarks every headline number in the README comes from (--benchmarks does all three):

  --us          data/real/s2_us_bundle     16 US coastal scenes + same-day NOAA AIS. THE primary benchmark:
                                           recall 0.912, precision 0.697 over 571 hand-adjudicated contacts.
                                           ~400 MB per AIS day file; this is the slowest step by far.
  --svalbard    data/real/svalbard_poc     2 Svalbard scenes (2024-06-22) + Kystverket AIS. The only Arctic
                                           benchmark with AIS truth -- NOAA stops at 50.195 N. Kystdatahuset
                                           needs no credentials; data is NLOD, credit Kystverket.
  --fundy       data/real/fundy_bundle     Bay of Fundy 2024-08-08 + NOAA AIS. Canadian water, US feed: USCG
                                           receivers in Maine hear MMSI 316* across the boundary.

Optional extras:

  --arctic      data/real/arctic_probe     4 Alaskan sea-ice scenes. NO AIS exists above 50.195 N, so these
                                           are scored for precision by hand only.
  --thermal     data/real/arctic_thermal   Landsat 8/9 winter thermal over the same Arctic AOIs, ~6 MB per
                                           AOI (ST_B10 cut by HTTP range, not whole scenes).

--data-root puts data/real, data/eval_bundle and data/heldout_bundle on another drive and links them back into the
repo (directory junction on Windows, symlink elsewhere), so every path in the docs keeps working.

The flight models (applet/models/*.onnx, 59 KB) are committed; retraining is optional:
    pip install -r requirements-train.txt
    python training/sen2ms.py mine && python training/train_verifier.py --real-npz data/real/sen2ms/train_chips.npz
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LINKED_DIRS = ("real", "eval_bundle", "heldout_bundle")
SEN2MS_URL = "https://zenodo.org/api/records/15571607/files/SEN2MS_Vessel_BBoxes.zip/content"
SEN2MS_BYTES = 565141487


def run(*args: str) -> None:
    print(f"\n$ python {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def has_files(path: str, suffix: str = "") -> bool:
    return os.path.isdir(path) and any(f.endswith(suffix) for f in os.listdir(path))


def link_data_root(data_root: str) -> None:
    data_root = os.path.abspath(data_root)
    os.makedirs(DATA, exist_ok=True)
    for name in LINKED_DIRS:
        link, target = os.path.join(DATA, name), os.path.join(data_root, name)
        if os.path.islink(link) or (os.path.isdir(link) and os.path.realpath(link) != os.path.abspath(link)):
            print(f"[link] data/{name} already links to {os.path.realpath(link)}")
            continue
        if os.path.isdir(link) and os.listdir(link):
            print(f"[link] data/{name} already holds files in the repo; leaving it where it is")
            continue
        if os.path.isdir(link):
            os.rmdir(link)
        os.makedirs(target, exist_ok=True)
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True, stdout=subprocess.DEVNULL)
        else:
            os.symlink(target, link, target_is_directory=True)
        print(f"[link] data/{name} -> {target}")


def download(url: str, path: str, expected: int = 0) -> None:
    """Resumable download (the Zenodo file is large enough to be worth resuming)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if expected and have == expected:
        print(f"[ok] {os.path.basename(path)} already complete")
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        resumed = r.status == 206
        total = have + int(r.headers.get("Content-Length", 0)) if resumed else int(r.headers.get("Content-Length", 0))
        with open(path, "ab" if resumed else "wb") as f:
            done = have if resumed else 0
            while True:
                block = r.read(4 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if done % (64 << 20) < (4 << 20):
                    print(f"     {done / 1e6:6.0f} / {total / 1e6:.0f} MB")
    print(f"[ok] {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} MB)")


def ais_empty(bundle: str) -> bool:
    """True when a bundle has no ais_catalog.json, or one with no vessels in it."""
    path = os.path.join(bundle, "ais_catalog.json")
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return True
    rows = blob if isinstance(blob, list) else blob.get("vessels") or blob.get("catalog") or []
    return not rows


def need(module: str, hint: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        print(f"[skip] needs `{module}`: {hint}")
        return False


def step_synthetic(force: bool) -> None:
    if force or not has_files(os.path.join(DATA, "sample_bundle"), ".tif"):
        run("scripts/generate_synthetic_data.py")
    if force or not has_files(os.path.join(DATA, "eval_bundle"), ".tif"):
        run("scripts/generate_synthetic_data.py", "--random", "40", "--seed", "777", "--size", "768", "-o", "data/eval_bundle")
    if force or not has_files(os.path.join(DATA, "heldout_bundle"), ".tif"):
        run("scripts/generate_synthetic_data.py", "--random", "60", "--seed", "4242", "--size", "768", "-o", "data/heldout_bundle")


def step_sentinel2(force: bool) -> None:
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    if force or not has_files(os.path.join(DATA, "real", "s2_bundle"), ".tif"):
        run("scripts/fetch_sentinel2.py", "-o", "data/real/s2_bundle")


def step_sen2ms(force: bool) -> None:
    folder = os.path.join(DATA, "real", "sen2ms")
    download(SEN2MS_URL, os.path.join(folder, "SEN2MS_Vessel_BBoxes.zip"), SEN2MS_BYTES)
    if force or not has_files(os.path.join(folder, "ais_bundle"), ".tif"):
        run("training/sen2ms.py", "bundle")


def step_noaa(force: bool) -> None:
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    bundle = os.path.join(DATA, "real", "s2_ais_bundle")
    if force or not has_files(bundle, ".tif"):
        # pinned to the one date the README numbers come from
        run("scripts/fetch_sentinel2.py", "-o", "data/real/s2_ais_bundle", "--only", "S2_LONGBEACH",
            "--start", "2024-11-08", "--end", "2024-11-08", "--max-cloud", "10")
    with open(os.path.join(bundle, "known_structures.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "THUMS artificial oil island off Long Beach; position read from the scene itself",
                   "structures": [{"name": "THUMS Island White (oil island)", "latitude": 33.7535,
                                   "longitude": -118.1599, "radius_m": 150}]}, f, indent=1)
    catalog = os.path.join(bundle, "ais_catalog.json")
    empty = True
    if os.path.exists(catalog):
        with open(catalog, "r", encoding="utf-8") as f:
            empty = not json.load(f).get("vessels")
    if force or empty:
        run("scripts/fetch_noaa_ais.py", "--bundle", "data/real/s2_ais_bundle")


def step_us(force: bool) -> None:
    """The primary benchmark. 16 scenes, then one NOAA day file per distinct acquisition date."""
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    bundle = os.path.join(DATA, "real", "s2_us_bundle")
    if force or not has_files(bundle, ".tif"):
        run("scripts/fetch_sentinel2.py", "--region", "us", "-o", "data/real/s2_us_bundle")
    if force or ais_empty(bundle):
        # Each distinct scene date pulls its own ~400 MB NOAA archive. Slow, but cached.
        run("scripts/fetch_noaa_ais.py", "--bundle", "data/real/s2_us_bundle")


def step_svalbard(force: bool) -> None:
    """
    The only Arctic benchmark with AIS ground truth.

    Pinned to 2024-06-22 because that is the acquisition the README numbers come from: both scenes
    carry real Kystverket traffic within the correlation window and 9-42 % sea ice.
    """
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    bundle = os.path.join(DATA, "real", "svalbard_poc")
    if force or not has_files(bundle, ".tif"):
        run("scripts/fetch_sentinel2.py", "--region", "svalbard", "-o", "data/real/svalbard_poc",
            "--only", "SVA_KONGSFJORD", "SVA_ISFJORDEN",
            "--start", "2024-06-22", "--end", "2024-06-22", "--max-cloud", "5")
    if force or ais_empty(bundle):
        run("scripts/fetch_kystverket_ais.py", "--bundle", "data/real/svalbard_poc")


def step_fundy(force: bool) -> None:
    """
    Atlantic Canada, scored against NOAA AIS.

    Pinned to 2024-08-08: NOAA's reach into Canadian water is propagation-dependent and varies
    enormously by date, and this one measured 15 broadcasters against a 0.07 % cloud acquisition.
    Probe any other candidate date with scripts/probe_canada_ais.py before trusting it. NOAA's
    archive is 2024-07-01..2024-12-31 only -- earlier 2024 and all 2025 return 404.
    """
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    bundle = os.path.join(DATA, "real", "fundy_bundle")
    if force or not has_files(bundle, ".tif"):
        run("scripts/fetch_sentinel2.py", "--region", "atlantic", "-o", "data/real/fundy_bundle",
            "--start", "2024-08-08", "--end", "2024-08-08", "--max-cloud", "5")
    if force or ais_empty(bundle):
        run("scripts/fetch_noaa_ais.py", "--bundle", "data/real/fundy_bundle")


def step_arctic(force: bool) -> None:
    """Alaskan sea ice. No AIS exists up here, so these are hand-adjudicated for precision only."""
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    if force or not has_files(os.path.join(DATA, "real", "arctic_probe"), ".tif"):
        run("scripts/fetch_sentinel2.py", "--region", "arctic", "--min-ice", "5",
            "-o", "data/real/arctic_probe")


def step_thermal(force: bool) -> None:
    """Landsat winter thermal. Only ST_B10 + QA windows are cut, so ~6 MB per AOI rather than ~1 GB."""
    if not need("rasterio", "pip install -r requirements-dev.txt"):
        return
    if force or not has_files(os.path.join(DATA, "real", "arctic_thermal"), ".tif"):
        run("scripts/fetch_landsat_thermal.py", "--season", "winter", "-o", "data/real/arctic_thermal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="folder on another drive for the large data (linked back into data/)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sentinel2", action="store_true")
    ap.add_argument("--sen2ms", action="store_true")
    ap.add_argument("--noaa", action="store_true")
    ap.add_argument("--us", action="store_true", help="16 US scenes + NOAA AIS (the primary benchmark)")
    ap.add_argument("--svalbard", action="store_true", help="Svalbard + Kystverket AIS (the Arctic benchmark)")
    ap.add_argument("--fundy", action="store_true", help="Bay of Fundy + NOAA AIS (Atlantic Canada)")
    ap.add_argument("--arctic", action="store_true", help="Alaskan sea-ice scenes (no AIS exists there)")
    ap.add_argument("--thermal", action="store_true", help="Landsat winter thermal over the Arctic AOIs")
    ap.add_argument("--benchmarks", action="store_true",
                    help="the three AIS-scored benchmarks the docs quote: --us --svalbard --fundy")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo steps whose output already exists")
    args = ap.parse_args()

    if args.data_root:
        link_data_root(args.data_root)
    ALL_STEPS = ("synthetic", "sentinel2", "sen2ms", "noaa", "us", "svalbard", "fundy", "arctic", "thermal")
    BENCHMARKS = ("us", "svalbard", "fundy")
    chosen = [s for s in ALL_STEPS
              if args.all or getattr(args, s) or (args.benchmarks and s in BENCHMARKS)] or ["synthetic"]
    for name in chosen:
        print(f"\n=== {name} ===")
        try:
            globals()[f"step_{name}"](args.force)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[FAILED] {name}: {e}  (the other steps are independent; re-run this one later)")

    print("\nBundles ready:")
    for dirpath, dirnames, filenames in os.walk(DATA, followlinks=True):
        dirnames[:] = [d for d in dirnames if d != "outputs"]
        if "manifest.json" in filenames:
            print("  ", os.path.relpath(dirpath, ROOT).replace("\\", "/"))
    print("\nNext:  python -m pytest -q   |   python -m ground.server   |   "
          "python -m applet run -i data/sample_bundle -o data/outputs -c config.example.yaml")
    print("\nScore the benchmarks (each prints the numbers the README quotes):")
    for bundle in ("s2_us_bundle", "svalbard_poc", "fundy_bundle"):
        if os.path.exists(os.path.join(DATA, "real", bundle, "manifest.json")):
            print(f"  python scripts/scorecard.py -i data/real/{bundle}")


if __name__ == "__main__":
    main()
