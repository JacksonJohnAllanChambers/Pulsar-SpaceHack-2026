"""
Builds a single self-contained HTML page for adjudicating every contact in a real-scene pass.

Precision on real imagery cannot be measured from AIS alone: a contact with no transponder is
either a genuine unlisted vessel (which is the product) or a false alarm (which is not), and only
a human looking at the pixels can say which. This page shows each contact as a true-colour crop
beside its NIR crop, at two zooms, with what the detector claimed about it, and records a verdict
per contact. It also shows the reverse case -- AIS broadcasters in clear water where the detector
found nothing -- so a miss can be confirmed as a real miss rather than a stale or bogus fix.

    python scripts/scorecard.py -i data/real/s2_us_bundle -o data/outputs/us_scorecard
    python scripts/build_contact_sheet.py -s data/outputs/us_scorecard -o data/outputs/review

Label in the browser (V vessel / N not a vessel / S fixed structure / U unsure), press Download,
then feed the file back in:

    python scripts/scorecard.py -i data/real/s2_us_bundle --labels data/outputs/review/labels.json
"""

import os
import io
import sys
import json
import base64
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.utils.geo import SceneGeoreference  # noqa: E402
from applet.utils.image_io import load_scene_raster, CANONICAL_BANDS  # noqa: E402

ZOOMS = (96, 320)  # crop half-widths in metres: tight hull view, and enough context for a wake
TILE_PX = 150


def stretch(tile: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile stretch to 8-bit. Water is dark and nearly flat, so a fixed scale shows nothing."""
    finite = tile[np.isfinite(tile)]
    if finite.size == 0:
        return np.zeros(tile.shape, np.uint8)
    lo, hi = np.percentile(finite, lo_pct), np.percentile(finite, hi_pct)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((tile - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def encode(rgb: np.ndarray, size: int = TILE_PX) -> str:
    import cv2

    if rgb.shape[0] != size:
        interp = cv2.INTER_NEAREST if rgb.shape[0] < size else cv2.INTER_AREA
        rgb = cv2.resize(rgb, (size, size), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def crop(refl: np.ndarray, cx: float, cy: float, half_px: int) -> Optional[np.ndarray]:
    h, w = refl.shape[:2]
    x0, y0 = int(round(cx)) - half_px, int(round(cy)) - half_px
    x1, y1 = x0 + 2 * half_px, y0 + 2 * half_px
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)), (0, 0))
    tile = refl[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if any(sum(p) for p in pad[:2]):
        tile = np.pad(tile, pad, mode="constant", constant_values=0.0)
    return tile


def views(refl: np.ndarray, cx: float, cy: float, gsd: float) -> List[str]:
    """One true-colour and one NIR crop per zoom. NIR is where a hull actually stands out."""
    out = []
    for half_m in ZOOMS:
        tile = crop(refl, cx, cy, max(8, int(round(half_m / gsd))))
        if tile is None:
            out.extend(["", ""])
            continue
        rgb = np.dstack([stretch(tile[:, :, i]) for i in (0, 1, 2)])
        nir = stretch(tile[:, :, 3])
        out.append(encode(rgb))
        out.append(encode(np.dstack([nir, nir, nir])))
    return out


def scene_lookup(bundle: str) -> Dict[str, Dict[str, Any]]:
    with open(os.path.join(bundle, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {s["id"]: s for s in manifest["scenes"]}, manifest


def load(bundle: str, scene: Dict[str, Any], manifest: Dict[str, Any]
         ) -> Tuple[Optional[np.ndarray], Optional[SceneGeoreference]]:
    refl, _, status = load_scene_raster(
        os.path.join(bundle, scene["file"]),
        band_names=manifest.get("bands", list(CANONICAL_BANDS)),
        reflectance_scale=manifest.get("reflectance_scale"),
        reflectance_offset=float(scene.get("reflectance_offset", 0.0)),
    )
    if refl is None:
        print(f"  ! {scene['id']}: {status['error']}")
        return None, None
    gsd = float(scene.get("gsd_meters", manifest.get("gsd_meters", 10.0)))
    return refl, SceneGeoreference(scene, status["width"], status["height"], gsd)


def build_items(sc_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(sc_dir, "contacts.json"), "r", encoding="utf-8") as f:
        payload = json.load(f)
    bundle = payload["bundle"]
    scenes, manifest = scene_lookup(bundle)

    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for det in payload["contacts"]:
        by_scene.setdefault(det["scene_id"], []).append(det)
    misses: Dict[str, List[Dict[str, Any]]] = {}
    for miss in payload.get("ais_not_observed", []):
        if miss["reason"] == "CLEAR_WATER_NO_TARGET":
            misses.setdefault(miss["scene_id"], []).append(miss)

    items: List[Dict[str, Any]] = []
    for sid in sorted(set(by_scene) | set(misses)):
        scene = scenes.get(sid)
        if scene is None:
            continue
        print(f"[{sid}] {len(by_scene.get(sid, []))} contacts, {len(misses.get(sid, []))} AIS misses")
        refl, georef = load(bundle, scene, manifest)
        if refl is None:
            continue
        gsd = georef.gsd_m
        for det in by_scene.get(sid, []):
            x, y = det["apex_px"]
            items.append({
                "kind": "contact",
                "id": det["detection_id"],
                "scene": sid,
                "cls": det["classification"],
                "imgs": views(refl, x, y, gsd),
                "lat": round(det["world_coordinates"]["latitude"], 5),
                "lon": round(det["world_coordinates"]["longitude"], 5),
                "facts": [
                    f"{det.get('size_class', '?').replace('_', ' ').lower()}, "
                    f"hull {det.get('hull_length_m', 0):.0f} m"
                    + ("" if det.get("hull_resolved", True) else " (unresolved)"),
                    f"heading {det['heading_deg']:.0f} deg"
                    + (" (+-180)" if det.get("heading_ambiguous_180") else "")
                    + f", wake {det.get('wake_length_m', 0):.0f} m",
                    f"confidence {det.get('confidence', 0):.2f}"
                    + (f", CNN {det['verifier_score']:.2f}" if det.get("verifier_score") is not None else ""),
                    det.get("intelligence_notes", ""),
                ],
                "prelabel": "vessel" if det["classification"] in
                            ("CONFIRMED_KNOWN_VESSEL", "AIS_KINEMATIC_MISMATCH") else "",
            })
        for miss in misses.get(sid, []):
            x, y = georef.lonlat_to_pixel(miss["predicted_longitude"], miss["predicted_latitude"])
            items.append({
                "kind": "miss",
                "id": f"MISS_{sid}_{miss['mmsi']}",
                "scene": sid,
                "cls": "AIS_NOT_OBSERVED",
                "imgs": views(refl, x, y, gsd),
                "lat": miss["predicted_latitude"],
                "lon": miss["predicted_longitude"],
                "facts": [
                    f"MMSI {miss['mmsi']} - {miss.get('name', '')}".strip(" -"),
                    "AIS places a vessel here in clear water; the detector found nothing.",
                    "Is a vessel visible at the centre of these crops?",
                    "",
                ],
                "prelabel": "",
            })
        del refl
    return items


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Contact review</title>
<style>
:root{--bg:#0e1116;--panel:#171c24;--line:#2a3240;--fg:#e6edf3;--dim:#8b98a8;
--vessel:#3fb950;--not:#f85149;--struct:#d29922;--unsure:#8b949e;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.keys{color:var(--dim);font-size:12px}
.keys b{color:var(--fg);background:#222a35;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-weight:600}
button{background:var(--accent);color:#04121f;border:0;border-radius:6px;padding:7px 13px;font-weight:600;cursor:pointer}
button.ghost{background:#222a35;color:var(--fg);border:1px solid var(--line)}
#bar{flex:1;min-width:160px;height:7px;background:#222a35;border-radius:4px;overflow:hidden}
#fill{height:100%;width:0;background:var(--vessel);transition:width .2s}
main{padding:16px;display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px;scroll-margin:90px}
.card.cur{outline:2px solid var(--accent);outline-offset:2px}
.card.miss{border-color:#3d2f12}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:8px}
.id{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;color:var(--dim)}
.cls{font-size:11px;font-weight:700;letter-spacing:.3px}
.imgs{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}
.imgs figure{margin:0}
.imgs img{width:100%;display:block;border-radius:4px;background:#000;image-rendering:pixelated}
.imgs figcaption{font-size:9px;color:var(--dim);text-align:center;margin-top:2px}
.facts{margin:8px 0 0;font-size:12px;color:var(--dim)}
.facts div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.facts div:last-child{white-space:normal;color:#6e7b8c;font-size:11px;margin-top:3px}
.verdict{margin-top:9px;display:flex;gap:5px}
.verdict span{flex:1;text-align:center;font-size:11px;font-weight:700;padding:5px 0;border-radius:5px;
border:1px solid var(--line);color:var(--dim);cursor:pointer;user-select:none}
.v-vessel.on{background:var(--vessel);color:#04120a;border-color:var(--vessel)}
.v-not_vessel.on{background:var(--not);color:#1a0505;border-color:var(--not)}
.v-structure.on{background:var(--struct);color:#160f02;border-color:var(--struct)}
.v-unsure.on{background:var(--unsure);color:#0d1117;border-color:var(--unsure)}
.done{opacity:.55}
</style></head><body>
<header>
<h1>Contact review</h1>
<div id="bar"><div id="fill"></div></div>
<div id="count" class="keys"></div>
<div class="keys"><b>V</b> vessel <b>N</b> not <b>S</b> structure <b>U</b> unsure <b>&larr;&rarr;</b> move</div>
<button id="dl">Download labels.json</button>
<button id="clr" class="ghost">Reset</button>
</header>
<main id="grid"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "contact_labels_v1";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
ITEMS.forEach(it => { if (it.prelabel && !labels[it.id]) labels[it.id] = it.prelabel; });
let cur = 0;
const VERDICTS = ["vessel", "not_vessel", "structure", "unsure"];
const LBL = {vessel: "VESSEL", not_vessel: "NOT", structure: "STRUCT", unsure: "?"};
const CAPS = ["RGB close", "NIR close", "RGB wide", "NIR wide"];
const grid = document.getElementById("grid");

ITEMS.forEach((it, i) => {
  const card = document.createElement("div");
  card.className = "card" + (it.kind === "miss" ? " miss" : "");
  card.id = "c" + i;
  const colour = it.cls === "DARK_VESSEL" ? "#f85149" : it.cls === "AIS_NOT_OBSERVED" ? "#d29922"
    : it.cls === "CONFIRMED_KNOWN_VESSEL" ? "#3fb950" : "#58a6ff";
  card.innerHTML = `
    <div class="hd"><span class="cls" style="color:${colour}">${it.cls.replace(/_/g, " ")}</span>
    <span class="id">${it.scene}</span></div>
    <div class="imgs">${it.imgs.map((s, k) => s
      ? `<figure><img src="${s}" alt=""><figcaption>${CAPS[k]}</figcaption></figure>`
      : `<figure><img alt=""><figcaption>${CAPS[k]}</figcaption></figure>`).join("")}</div>
    <div class="facts">${it.facts.map(f => `<div>${f}</div>`).join("")}
      <div>${it.lat}, ${it.lon}</div></div>
    <div class="verdict">${VERDICTS.map(v =>
      `<span class="v-${v}" data-v="${v}" data-i="${i}">${LBL[v]}</span>`).join("")}</div>`;
  grid.appendChild(card);
});

function paint() {
  ITEMS.forEach((it, i) => {
    const card = document.getElementById("c" + i);
    card.classList.toggle("cur", i === cur);
    card.classList.toggle("done", !!labels[it.id] && i !== cur);
    VERDICTS.forEach(v => card.querySelector(".v-" + v).classList.toggle("on", labels[it.id] === v));
  });
  const n = ITEMS.filter(it => labels[it.id]).length;
  document.getElementById("fill").style.width = (100 * n / ITEMS.length) + "%";
  document.getElementById("count").textContent = n + " / " + ITEMS.length + " labelled";
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
}

function setLabel(i, v) {
  labels[ITEMS[i].id] = v;
  if (i === cur && cur < ITEMS.length - 1) cur++;
  paint();
  document.getElementById("c" + cur).scrollIntoView({block: "nearest", behavior: "smooth"});
}

grid.addEventListener("click", e => {
  const t = e.target.closest("[data-v]");
  if (t) { cur = +t.dataset.i; setLabel(cur, t.dataset.v); }
});

document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  const map = {v: "vessel", n: "not_vessel", s: "structure", u: "unsure"};
  if (map[k]) { e.preventDefault(); setLabel(cur, map[k]); }
  else if (k === "arrowright" || k === "j") { e.preventDefault(); cur = Math.min(cur + 1, ITEMS.length - 1); paint();
    document.getElementById("c" + cur).scrollIntoView({block: "nearest", behavior: "smooth"}); }
  else if (k === "arrowleft" || k === "k") { e.preventDefault(); cur = Math.max(cur - 1, 0); paint();
    document.getElementById("c" + cur).scrollIntoView({block: "nearest", behavior: "smooth"}); }
});

document.getElementById("dl").onclick = () => {
  const payload = {generated: new Date().toISOString(),
    labelled: Object.keys(labels).length, total: ITEMS.length, labels: labels};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 1)], {type: "application/json"}));
  a.download = "labels.json"; a.click();
};
document.getElementById("clr").onclick = () => {
  if (confirm("Clear every verdict, including the AIS-matched pre-labels?")) {
    labels = {}; cur = 0; paint();
  }
};
paint();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", "-s", default="data/outputs/us_scorecard",
                    help="directory holding contacts.json written by scripts/scorecard.py")
    ap.add_argument("--output", "-o", default="data/outputs/review")
    args = ap.parse_args()

    items = build_items(args.scorecard)
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "contact_sheet.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__ITEMS__", json.dumps(items)))
    n_miss = sum(1 for i in items if i["kind"] == "miss")
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[done] {len(items) - n_miss} contacts + {n_miss} AIS misses -> {path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
