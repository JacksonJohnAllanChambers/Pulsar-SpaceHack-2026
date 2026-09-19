"""
Builds a real ais_catalog.json for a bundle of US-waters scenes from NOAA / BOEM Marine Cadastre
(free, public-domain historical AIS, one zip per UTC day, ~340 MB; ground-side tool).

For every scene in the bundle manifest the day's file is fetched once into --cache, streamed
(never unpacked: it is ~800 MB of CSV), filtered to the scene footprint plus a margin and to
+-window minutes around the shutter time, and the single fix closest in time is kept per MMSI.
That is exactly what a ground station would uplink before the pass.

    python scripts/fetch_noaa_ais.py --bundle data/real/s2_ais_bundle

Coverage: US coastal waters only, published with a lag of roughly a year.
"""

import io
import os
import csv
import sys
import json
import zipfile
import argparse
import urllib.request
from datetime import datetime, timezone, timedelta

URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip"


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def download(day: datetime, cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    url = URL.format(y=day.year, m=day.month, d=day.day)
    path = os.path.join(cache, os.path.basename(url))
    if os.path.exists(path) and zipfile.is_zipfile(path):
        return path
    print(f"[NOAA] downloading {url}")
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while True:
            block = r.read(4 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total and done % (64 << 20) < (4 << 20):
                print(f"       {done / 1e6:6.0f} / {total / 1e6:.0f} MB")
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cache", default="data/real/noaa_ais")
    ap.add_argument("--window-min", type=float, default=20.0, help="keep fixes within +- this many minutes of shutter")
    ap.add_argument("--margin-deg", type=float, default=0.03, help="footprint margin (~3 km) for ships about to enter")
    args = ap.parse_args()

    with open(os.path.join(args.bundle, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    best = {}  # (scene id, mmsi) -> (abs dt, record)
    by_day = {}
    for sc in manifest["scenes"]:
        t = parse_time(sc["shutter_time"])
        lons = [c[0] for c in sc["corners"].values()]
        lats = [c[1] for c in sc["corners"].values()]
        box = (min(lons) - args.margin_deg, max(lons) + args.margin_deg,
               min(lats) - args.margin_deg, max(lats) + args.margin_deg)
        by_day.setdefault(t.date(), []).append((sc["id"], t, box))

    for day, scenes in by_day.items():
        path = download(datetime(day.year, day.month, day.day), args.cache)
        lo = min(t for _, t, _ in scenes) - timedelta(minutes=args.window_min)
        hi = max(t for _, t, _ in scenes) + timedelta(minutes=args.window_min)
        lo_s, hi_s = lo.strftime("%Y-%m-%dT%H:%M:%S"), hi.strftime("%Y-%m-%dT%H:%M:%S")
        n_rows = 0
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(name) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace")):
                    n_rows += 1
                    ts = row["BaseDateTime"]
                    if ts < lo_s or ts > hi_s:  # ISO strings sort chronologically: cheap pre-filter
                        continue
                    try:
                        lat, lon = float(row["LAT"]), float(row["LON"])
                    except ValueError:
                        continue
                    for sid, t, (x0, x1, y0, y1) in scenes:
                        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                            continue
                        dt = abs((parse_time(ts + "Z") - t).total_seconds())
                        if dt > args.window_min * 60:
                            continue
                        key = (sid, row["MMSI"])
                        if key not in best or dt < best[key][0]:
                            best[key] = (dt, row)
        print(f"[NOAA] {day}: scanned {n_rows:,} fixes")

    def num(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    vessels = []
    for (sid, mmsi), (dt, r) in sorted(best.items()):
        cog, hdg = num(r.get("COG"), -1), num(r.get("Heading"), 511)
        vessels.append({
            "mmsi": int(mmsi), "name": (r.get("VesselName") or f"MMSI {mmsi}").strip(), "timestamp": r["BaseDateTime"] + "Z",
            "latitude": num(r["LAT"]), "longitude": num(r["LON"]), "sog_knots": max(num(r.get("SOG")), 0.0),
            # COG is meaningless at rest; fall back to true heading when it is valid
            "cog_deg": cog if 0 <= cog < 360 else (hdg if hdg < 360 else 0.0),
            "heading_deg": hdg if hdg < 360 else None, "length_m": num(r.get("Length"), 0) or None,
            "vessel_type": r.get("VesselType"), "transceiver_class": r.get("TransceiverClass"),
            "fix_age_s": round(dt, 1), "source_scene": sid,
        })
    out = os.path.join(args.bundle, "ais_catalog.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "NOAA Office for Coastal Management / BOEM Marine Cadastre AIS (public domain)",
                   "vessels": vessels}, f, indent=1)
    print(f"[DONE] {len(vessels)} vessels within +-{args.window_min:.0f} min of shutter -> {out}")


if __name__ == "__main__":
    sys.exit(main())
