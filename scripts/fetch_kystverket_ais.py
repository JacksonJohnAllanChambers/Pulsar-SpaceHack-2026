"""
Builds a real ais_catalog.json for Arctic scenes from Kystverket's Kystdatahuset Open API
(free, NLOD, no account; ground-side tool).

This is the Arctic counterpart to scripts/fetch_noaa_ais.py. NOAA Marine Cadastre stops at
50.195 N, so the four Arctic Alaska scenes in data/real/arctic_probe have no AIS at all and
their precision is hand-adjudicated. Kystverket's archive covers the Norwegian economic zone
*and the protection zones off Svalbard and Jan Mayen*, and its satellite feed reaches past 84 N
-- genuinely Arctic, ice-adjacent, with real traffic -- so a Svalbard bundle can be scored for
recall against AIS the imagery never saw, exactly like the 16 US scenes.

    python scripts/fetch_kystverket_ais.py --bundle data/real/svalbard_bundle
    python scripts/fetch_kystverket_ais.py --bbox 9,76,30,81 --start 2024-07-15 --end 2024-07-15 \
        --out data/real/svalbard_survey

Two things about this API were measured rather than assumed, and both shape the code:

1. `start`/`end` are NOT a UTC filter. The service resolves them against per-day PostgreSQL
   partitions in Europe/Oslo local time and includes the partition one day early: asking for
   2024-07-15 12:00-12:01 returns fixes stamped 2024-07-14T14:00:00 .. 2024-07-15T14:00:59
   (+2 h in CEST, +1 h in CET -- the offset tracks Oslo DST, so the *parameter* is local and the
   returned `date_time_utc` really is UTC). We therefore ask wide and filter in Python.
2. `speed_over_ground` is in KNOTS despite the swagger labelling the neighbouring `calc_speed`
   column "kph" -- checked by recomputing speed from consecutive positions (11.4 reported vs
   11.2 recomputed knots / 20.7 km/h). So it maps straight onto `sog_knots`.

Licence: Norwegian Licence for Open Government Data (NLOD), declared by the API itself
(https://data.norge.no/nlod/en/1.0). Kystverket must be credited as the source.
Coverage: Norwegian waters, EEZ and the Svalbard / Jan Mayen protection zones.
Archive: roughly 2010 to about six months behind today (2026-03-17 was the last day available
when this was written); vessels are excluded below the privacy thresholds Kystverket applies.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

BASE = "https://kystdatahuset.no/ws"
POSITIONS = "/api/ais/positions/within-bbox-time"
STATINFO = "/api/ais/statinfo/for-mmsis-time"

# The API docs ask callers to identify themselves so Kystverket can get in touch about load
# rather than simply throttling; there is no key and no rate limit.
UA = "Pulsar-SpaceHack-2026 TacticalEdgeSentinel (scripts/fetch_kystverket_ais.py)"

# Columns of a position row, in order, from the swagger response example.
I_MMSI, I_TIME, I_LON, I_LAT, I_COG, I_SOG, I_MSG, I_CALC, I_DSEC, I_DIST, I_HDG, I_ROT = range(12)

# Message 1/2/3 are Class A position reports, 18/19 Class B. The API does not carry the
# transceiver class as its own column, so derive it -- the correlator uses it only as a hint.
CLASS_A_MSGS = {1, 2, 3}
CLASS_B_MSGS = {18, 19}


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def post(path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_day(day, box, cache: str) -> list:
    """All position rows the service will give us for one UTC day inside `box`.

    Cached because a busy Svalbard day is ~600k rows and 30-60 s of server time, and the
    recon loop revisits the same days. `box` is (lon_min, lon_max, lat_min, lat_max)."""
    lon0, lon1, lat0, lat1 = box
    bbox = f"{lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f}"
    key = hashlib.sha256(f"{day.isoformat()}|{bbox}".encode()).hexdigest()[:16]
    path = os.path.join(cache, f"kdh_{day:%Y%m%d}_{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Asking for [day 0000, day 2359] already reaches back into the previous partition (the
    # service starts a day early, see above), which is exactly enough to cover the whole UTC
    # day and no more. Widening it further only pulls in more partitions, and a single missing
    # one fails the whole query -- 2024-08-01 fails this way because ais_20240730 is absent.
    start = day.strftime("%Y%m%d0000")
    end = day.strftime("%Y%m%d2359")
    print(f"[KDH] {day:%Y-%m-%d} bbox {bbox} ...", flush=True)
    t0 = time.time()
    try:
        resp = post(POSITIONS, {"bbox": bbox, "start": start, "end": end, "minSpeed": 0.0})
    except urllib.error.HTTPError as e:
        print(f"       HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}")
        return []
    except Exception as e:  # the service kills queries at four minutes; a smaller box is the fix
        print(f"       {type(e).__name__}: {str(e)[:120]}")
        return []
    if not resp.get("success"):
        print(f"       service said no: {(resp.get('msg') or '').strip().splitlines()[0][:120]}")
        return []
    rows = resp.get("data") or []
    os.makedirs(cache, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)
    print(f"       {len(rows):,} fixes in {time.time() - t0:.0f} s -> {os.path.basename(path)}")
    return rows


def fetch_statinfo(mmsis, day, cache: str) -> dict:
    """Name / IMO / length / AIS ship type per MMSI. Open to anonymous callers, unlike
    /api/ship/data/ais/for-mmsis-imos which is 401 for the unauthenticated role."""
    if not mmsis:
        return {}
    key = hashlib.sha256(",".join(str(m) for m in sorted(mmsis)).encode()).hexdigest()[:16]
    path = os.path.join(cache, f"static_{day:%Y%m%d}_{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    out = {}
    mmsis = sorted(mmsis)
    for i in range(0, len(mmsis), 200):  # keep each request well inside the four-minute ceiling
        chunk = mmsis[i:i + 200]
        try:
            resp = post(STATINFO, {"mmsiIds": chunk, "start": day.strftime("%Y%m%d0000"),
                                   "end": day.strftime("%Y%m%d2359")})
        except Exception as e:
            print(f"[KDH] static info failed for {len(chunk)} MMSI: {type(e).__name__}: {str(e)[:80]}")
            continue
        for rec in (resp.get("data") or []):
            out[int(rec["mmsi"])] = rec
    os.makedirs(cache, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    print(f"[KDH] static info for {len(out)} of {len(mmsis)} MMSI")
    return out


def scene_boxes(manifest, margin_deg: float):
    """(scene id, shutter time, lon/lat box) per scene, with a margin for ships about to enter.

    The longitude margin is stretched by 1/cos(lat) so that it is the same number of kilometres
    as the latitude margin -- at 78 N a degree of longitude is only 23 km, and an unstretched
    0.03 deg would be a 700 m buffer instead of 3 km."""
    import math
    out = []
    for sc in manifest["scenes"]:
        t = parse_time(sc["shutter_time"])
        lons = [c[0] for c in sc["corners"].values()]
        lats = [c[1] for c in sc["corners"].values()]
        mid = math.radians(sum(lats) / len(lats))
        mlon = margin_deg / max(math.cos(mid), 0.05)
        out.append((sc["id"], t, (min(lons) - mlon, max(lons) + mlon,
                                  min(lats) - margin_deg, max(lats) + margin_deg)))
    return out


def nearest_fixes(rows, scenes, window_s: float):
    """Keep, per (scene, MMSI), the single position fix closest in time to that scene's shutter.

    That is what a ground station would uplink before the pass, and it is what
    fetch_noaa_ais.py does, so the two catalogues mean the same thing."""
    best = {}
    for r in rows:
        try:
            lon, lat = float(r[I_LON]), float(r[I_LAT])
            t = parse_time(r[I_TIME] + "Z")
        except (TypeError, ValueError, IndexError):
            continue
        for sid, shutter, (x0, x1, y0, y1) in scenes:
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                continue
            dt = abs((t - shutter).total_seconds())
            if dt > window_s:
                continue
            key = (sid, int(r[I_MMSI]))
            if key not in best or dt < best[key][0]:
                best[key] = (dt, r)
    return best


def to_vessels(best, static) -> list:
    def num(v, default=None):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return f

    vessels = []
    for (sid, mmsi), (dt, r) in sorted(best.items()):
        cog, hdg = num(r[I_COG], -1.0), num(r[I_HDG], 511.0)
        msg = int(num(r[I_MSG], 0) or 0)
        st = static.get(mmsi, {})
        length = num(st.get("length"), None)
        vessels.append({
            "mmsi": mmsi,
            "name": (st.get("name") or f"MMSI {mmsi}").strip(),
            "timestamp": r[I_TIME] + "Z",
            "latitude": round(num(r[I_LAT], 0.0), 6),
            "longitude": round(num(r[I_LON], 0.0), 6),
            "sog_knots": max(num(r[I_SOG], 0.0), 0.0),
            # COG is meaningless at rest; fall back to true heading when it is valid
            "cog_deg": cog if 0 <= cog < 360 else (hdg if hdg < 360 else 0.0),
            "heading_deg": hdg if hdg < 360 else None,
            "length_m": length if length and length > 0 else None,
            "vessel_type": st.get("type"),
            "transceiver_class": "A" if msg in CLASS_A_MSGS else ("B" if msg in CLASS_B_MSGS else None),
            "fix_age_s": round(dt, 1),
            "source_scene": sid,
        })
    return vessels


SOURCE = ("Kystverket (Norwegian Coastal Administration) Kystdatahuset Open API, "
          "Norwegian Licence for Open Government Data (NLOD)")


def write_catalog(out_dir: str, vessels: list, note: str = None):
    os.makedirs(out_dir, exist_ok=True)
    doc = {"source": SOURCE, "vessels": vessels}
    if note:
        doc["note"] = note
    path = os.path.join(out_dir, "ais_catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1])
    ap.add_argument("--bundle", help="bundle directory holding manifest.json; writes ais_catalog.json into it")
    ap.add_argument("--bbox", help="survey mode: lon_min,lat_min,lon_max,lat_max (no bundle needed)")
    ap.add_argument("--start", help="survey mode: first UTC day, YYYY-MM-DD")
    ap.add_argument("--end", help="survey mode: last UTC day, YYYY-MM-DD (default: --start)")
    ap.add_argument("--at", help="survey mode: reference UTC time for fix_age_s "
                                 "(default: midday of the first day)")
    ap.add_argument("--out", help="survey mode: output directory")
    ap.add_argument("--cache", default="data/real/kystverket_ais")
    ap.add_argument("--window-min", type=float, default=20.0,
                    help="keep fixes within +- this many minutes of shutter")
    ap.add_argument("--margin-deg", type=float, default=0.03,
                    help="footprint margin in latitude (~3 km), stretched in longitude by 1/cos(lat)")
    args = ap.parse_args()

    if not args.bundle and not (args.bbox and args.start):
        ap.error("give --bundle, or --bbox with --start (and --out)")

    if args.bundle:
        with open(os.path.join(args.bundle, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        scenes = scene_boxes(manifest, args.margin_deg)
        out_dir = args.bundle
    else:
        lon0, lat0, lon1, lat1 = (float(v) for v in args.bbox.split(","))
        day0 = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        day1 = datetime.strptime(args.end or args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        at = parse_time(args.at) if args.at else day0 + timedelta(hours=12)
        # One synthetic "scene" covering the whole box, so the survey path reuses the bundle path
        # rather than growing a second, differently-behaved code path.
        scenes = [(None, at, (lon0, lon1, lat0, lat1))]
        out_dir = args.out or "."
        args._survey_days = [day0 + timedelta(days=i) for i in range((day1 - day0).days + 1)]

    # One query per UTC day covering every scene shot that day, which is what the per-day
    # partitioning of the archive wants anyway.
    by_day = {}
    for sid, t, box in scenes:
        for d in (getattr(args, "_survey_days", None) or [t]):
            by_day.setdefault(d.date(), []).append((sid, t, box))

    window_s = args.window_min * 60.0
    if not args.bundle:
        # A survey is not tied to a shutter, so do not throw away the rest of the day.
        window_s = 12 * 3600.0

    best, days = {}, sorted(by_day)
    for day in days:
        todays = by_day[day]
        box = (min(b[0] for _, _, b in todays), max(b[1] for _, _, b in todays),
               min(b[2] for _, _, b in todays), max(b[3] for _, _, b in todays))
        rows = fetch_day(datetime(day.year, day.month, day.day, tzinfo=timezone.utc), box, args.cache)
        kept = nearest_fixes(rows, todays, window_s)
        print(f"[KDH] {day}: {len(rows):,} fixes scanned, {len(kept)} kept")
        for k, v in kept.items():
            if k not in best or v[0] < best[k][0]:
                best[k] = v

    static = fetch_statinfo({m for _, m in best}, datetime(days[0].year, days[0].month, days[0].day),
                            args.cache) if best else {}
    vessels = to_vessels(best, static)
    note = None
    if not args.bundle:
        note = (f"survey of bbox {args.bbox} over {args.start}..{args.end or args.start}; "
                f"fix_age_s is measured against {args.at or 'midday of the first day'}")
    path = write_catalog(out_dir, vessels, note)
    n_named = sum(1 for v in vessels if not v["name"].startswith("MMSI "))
    print(f"[DONE] {len(vessels)} vessels ({len({v['mmsi'] for v in vessels})} distinct MMSI, "
          f"{n_named} named) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
