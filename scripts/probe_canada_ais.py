"""
Probe: how much real AIS does NOAA Marine Cadastre actually carry in ATLANTIC CANADA
at a given Sentinel-2 shutter time? (research tool, ground-side, writes nothing)

Why this exists. NOAA/BOEM Marine Cadastre is described as "US waters", and the repo's
Arctic work records that it stops at ~50.2 N. Both are true, but neither says anything
about the *eastern* edge. Measured on the day files in data/real/noaa_ais, the feed does
reach across the maritime boundary into Canadian water -- the Bay of Fundy and southwest
Nova Scotia sit inside the range of the US Coast Guard NAIS receivers in Maine, and
Canadian-flag vessels (MMSI 316*) appear with normal 1-2 minute Class A/B cadence.

The catch, and the reason you must probe rather than assume: that coverage is strongly
date- and hour-dependent. On 2024-09-03 the Canadian box is essentially empty from 02:00
to 18:00 UTC and then carries 2,112 fixes in the 22:00 hour alone -- the signature of
extended VHF propagation at night, not of a permanent station. On 2024-07-05 it is flat
and populated around the clock. Sentinel-2 crosses the Bay of Fundy at ~15:30 UTC, so a
scene is only worth cutting if *that hour* on *that date* is covered.

So: pick candidate acquisitions from the STAC search first, then run this for each one
and keep the dates that come back with broadcasters.

    python scripts/probe_canada_ais.py 2024-07-14 15:30:41
    python scripts/probe_canada_ais.py 2024-10-17 15:30:38 --window-min 20

Nothing is written to disk and the day file is never saved: the zip is decompressed from
the socket as it arrives (~350-400 MB streamed, a few minutes). If you already have the
day file in data/real/noaa_ais it is read from there instead.

Coverage limits, measured rather than assumed:
  * NOAA publishes 2024-07-01 .. 2024-12-31 for this purpose; 2025 and 2026 return 404.
  * Halifax, the Scotian Shelf, the Gulf of St Lawrence and the Grand Banks are NOT
    covered in any useful way (1 MMSI per day file, if that). Do not plan a scene there
    expecting AIS.
  * open.canada.ca publishes no point-level AIS at all -- every federal AIS dataset is
    DFO's gridded "Vessel Density Mapping", which cannot serve as per-vessel truth.

Licence: NOAA Office for Coastal Management / BOEM Marine Cadastre, public domain.
"""

import io
import os
import csv
import sys
import zlib
import struct
import zipfile
import argparse
import collections
import urllib.request

URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m}_{d}.zip"
CACHE = os.path.join("data", "real", "noaa_ais")

# ~20 km candidate footprints on the Canadian side, plus the wide box they sit in.
SITES = {
    "NB_SAINT_JOHN":  (-66.20, 45.10, -65.95, 45.28),
    "NB_GRAND_MANAN": (-66.95, 44.60, -66.70, 44.78),
    "GRAND_MANAN_CH": (-66.85, 44.72, -66.62, 44.92),
    "NS_DIGBY_NECK":  (-66.40, 44.35, -66.15, 44.53),
    "NS_YARMOUTH":    (-66.30, 43.60, -66.05, 43.78),
    "FUNDY_WIDE":     (-67.20, 44.30, -64.80, 45.60),
    "SWNS_WIDE":      (-66.60, 43.00, -65.00, 44.40),
    "NS_HALIFAX":     (-63.70, 44.45, -63.45, 44.63),
}


class _ZipSocket(io.RawIOBase):
    """Inflate the first deflate member of a zip as it streams off the socket."""

    def __init__(self, resp):
        self.r, self.buf, self.bytes_in = resp, b"", 0
        self.d = zlib.decompressobj(-zlib.MAX_WBITS)
        sig, _v, _f, meth, _t, _dt, _c, _cs, _us, nlen, elen = struct.unpack(
            "<IHHHHHIIIHH", self._raw(30))
        if sig != 0x04034B50 or meth != 8:
            raise RuntimeError("not a streamable deflate zip")
        self._raw(nlen)
        self._raw(elen)

    def readable(self):
        return True

    def _raw(self, n):
        out = b""
        while len(out) < n:
            c = self.r.read(n - len(out))
            if not c:
                break
            out += c
        self.bytes_in += len(out)
        return out

    def readinto(self, b):
        while not self.buf:
            c = self.r.read(1 << 20)
            if not c:
                return 0
            self.bytes_in += len(c)
            self.buf = self.d.decompress(c, 1 << 22)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def open_day(day):
    """Yield a text stream of the day's CSV, from the local cache or straight off the wire."""
    y, m, d = day.split("-")
    local = os.path.join(CACHE, f"AIS_{y}_{m}_{d}.zip")
    if os.path.exists(local) and zipfile.is_zipfile(local):
        print(f"[NOAA] reading cached {local}")
        z = zipfile.ZipFile(local)
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace")
    url = URL.format(y=y, m=m, d=d)
    print(f"[NOAA] streaming {url} (nothing is saved)")
    req = urllib.request.Request(url, headers={"User-Agent": "IRIS research probe"})
    resp = urllib.request.urlopen(req, timeout=300)
    print(f"[NOAA] HTTP {resp.status}, {int(resp.headers.get('Content-Length', 0)) / 1e6:.0f} MB")
    return io.TextIOWrapper(io.BufferedReader(_ZipSocket(resp), 1 << 20),
                            encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day", help="UTC date of the acquisition, YYYY-MM-DD")
    ap.add_argument("shutter", nargs="?", default="15:30:40", help="UTC shutter time HH:MM:SS")
    ap.add_argument("--window-min", type=float, default=20.0,
                    help="keep fixes within +- this many minutes of the shutter")
    args = ap.parse_args()

    sh = (int(args.shutter[0:2]) * 60 + int(args.shutter[3:5]) + int(args.shutter[6:8]) / 60.0)
    lo, hi = sh - args.window_min, sh + args.window_min

    fixes = collections.Counter()
    best = collections.defaultdict(dict)   # site -> mmsi -> (dt_minutes, row)

    rdr = csv.reader(open_day(args.day))
    hdr = next(rdr)
    iM, iT = hdr.index("MMSI"), hdr.index("BaseDateTime")
    iA, iO = hdr.index("LAT"), hdr.index("LON")
    iN, iS, iC, iL = (hdr.index("VesselName"), hdr.index("SOG"),
                      hdr.index("COG"), hdr.index("Length"))

    for row in rdr:
        try:
            t = row[iT]
            cur = int(t[11:13]) * 60 + int(t[14:16]) + int(t[17:19]) / 60.0
        except (ValueError, IndexError):
            continue
        if not (lo <= cur <= hi):
            continue
        try:
            lat, lon = float(row[iA]), float(row[iO])
        except (ValueError, IndexError):
            continue
        for k, (x0, y0, x1, y1) in SITES.items():
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                fixes[k] += 1
                dd = abs(cur - sh)
                prev = best[k].get(row[iM])
                if prev is None or dd < prev[0]:
                    best[k][row[iM]] = (dd, row)

    print(f"\n== {args.day} shutter {args.shutter}Z, +-{args.window_min:.0f} min ==")
    print(f"{'footprint':16s} {'fixes':>7} {'broadcasters':>13} {'CA-flag(316*)':>14} {'underway>=3kn':>14}")
    for k in SITES:
        ms = best[k]
        ca = sum(1 for m in ms if m.startswith("316"))
        under = sum(1 for _d, r in ms.values() if (r[iS] or "0") not in ("", "nan")
                    and float(r[iS] or 0) >= 3.0)
        print(f"{k:16s} {fixes[k]:7d} {len(ms):13d} {ca:14d} {under:14d}")

    for k in SITES:
        if not best[k]:
            continue
        print(f"\n-- {k}, nearest the shutter --")
        for m, (dd, r) in sorted(best[k].items(), key=lambda kv: kv[1][0])[:25]:
            print(f"   {m} {r[iN][:22]:22s} fix_age={dd * 60:6.0f}s "
                  f"lat={r[iA]:>9} lon={r[iO]:>10} sog={r[iS]:>5} cog={r[iC]:>6} len={r[iL]:>5}")

    if not any(best.values()):
        print("\nNo AIS in any footprint at this time. Try another candidate date -- coverage of "
              "Canadian water here is propagation-dependent and varies day to day.")


if __name__ == "__main__":
    main()
