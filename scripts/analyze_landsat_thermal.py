"""
Measures what a Landsat thermal bundle actually shows over Arctic sea ice (ground-side).

Everything here is a number read off real pixels. The four questions it answers:

1. HOW BIG IS THE CONTRAST. Split the window by QA_PIXEL, which is CFMask's classification and
   is computed from the reflective bands and the cirrus test -- it never looks at ST_B10 -- so
   using it to define "water" and "snow/ice" makes the temperature difference an independent
   measurement rather than a threshold we chose and then rediscovered.

2. HOW WIDE A CHANNEL SURVIVES THE OPTICS. TIRS samples at 100 m and the L2 product is
   cubic-convolved onto the 30 m OLI grid, so a lead narrower than the thermal footprint is a
   mixed pixel and shows a diluted temperature. Water pixels are stratified by how enclosed they
   are (all 8 neighbours water == interior; 1-3 == a channel about one pixel wide) and the
   apparent contrast is reported per stratum. A ship-made channel is 25-50 m wide, so the
   thin-stratum number is the one that decides whether this is worth building on.

3. WHAT THE EFFECTIVE RESOLUTION IS. A semivariogram of the ST field, and the same semivariogram
   of the 10 m Sentinel-2 NIR band from data/real/arctic_probe as a control. The lag at which
   each reaches half its sill is a resolution proxy that does not depend on trusting metadata.

4. ARE LEADS VISIBLY RESOLVABLE. Warm anomalies are segmented at the ice mode + k sigma,
   connected-component-labelled, and measured for width (2x the peak of the distance transform)
   and elongation. Leads are long and thin; melt ponds and coastline are not.

    python scripts/analyze_landsat_thermal.py data/real/arctic_thermal
    python scripts/analyze_landsat_thermal.py data/real/arctic_thermal --png data/outputs/thermal

Landsat data courtesy of the U.S. Geological Survey.
"""

import os
import sys
import json
import argparse

import numpy as np

ST_SCALE, ST_OFFSET = 0.00341802, 149.0
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW, QA_CLEAR, QA_WATER = range(8)

# Seawater at Arctic salinity freezes at -1.8 C, so no open-water pixel can sit below that and a
# fully filled one should sit at it. It is the fixed point the whole argument rests on.
SEAWATER_FREEZING_C = -1.8


def load_scene(bundle: str, scene: dict):
    import tifffile
    data = tifffile.imread(os.path.join(bundle, scene["file"]))
    st, qa = data[..., 0], data[..., 1]
    scale = scene.get("temperature_scale", ST_SCALE)
    offset = scene.get("temperature_offset", ST_OFFSET)
    celsius = st.astype(np.float64) * scale + offset - 273.15
    valid = (st > 0) & (((qa >> QA_FILL) & 1) == 0)
    return celsius, qa, valid


def qa_masks(qa: np.ndarray, valid: np.ndarray):
    bit = lambda b: (((qa >> b) & 1) == 1) & valid
    obscured = bit(QA_CLOUD) | bit(QA_DILATED) | bit(QA_CIRRUS) | bit(QA_SHADOW)
    clear = valid & ~obscured
    water = bit(QA_WATER) & clear
    snowice = bit(QA_SNOW) & clear & ~water
    other = clear & ~water & ~snowice          # land, and any ice CFMask did not call snow
    return {"clear": clear, "cloud": obscured, "water": water, "ice": snowice, "other": other}


def describe(x: np.ndarray):
    if x.size < 30:
        return None
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "p5": round(float(p[0]), 2), "p25": round(float(p[1]), 2),
            "median": round(float(p[2]), 2), "p75": round(float(p[3]), 2),
            "p95": round(float(p[4]), 2), "max": round(float(x.max()), 2),
            "std": round(float(x.std()), 2)}


def modes(c: np.ndarray, lo=-40.0, hi=15.0, bins=440, smooth=5):
    """Peaks of a lightly smoothed temperature histogram -- the cold ice mode and any warm mode."""
    hist, edges = np.histogram(c, bins=bins, range=(lo, hi))
    k = np.ones(smooth) / smooth
    h = np.convolve(hist.astype(float), k, mode="same")
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks = [(centres[i], h[i]) for i in range(1, len(h) - 1)
             if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.02 * h.max()]
    peaks.sort(key=lambda t: -t[1])
    return [(round(float(t), 2), round(float(v / max(h.sum(), 1)), 4)) for t, v in peaks[:4]]


def neighbour_strata(celsius, masks, ice_median):
    """
    Apparent water temperature as a function of how enclosed the water pixel is.

    An interior water pixel (8 water neighbours) sits in a body at least ~90 m across and so
    fills a 100 m thermal footprint. A pixel with 1-3 water neighbours is a one-pixel-wide
    filament -- geometrically what a ship's channel looks like at 30 m sampling. The difference
    between the two rows is the mixed-pixel dilution, measured rather than modelled.
    """
    from scipy.ndimage import convolve
    w = masks["water"].astype(np.uint8)
    nb = convolve(w, np.ones((3, 3), np.uint8), mode="constant", cval=0) - w
    out = {}
    for label, sel in (("interior (8 water nbrs)", nb == 8),
                       ("edge (5-7)", (nb >= 5) & (nb <= 7)),
                       ("narrow (4)", nb == 4),
                       ("filament (1-3)", (nb >= 1) & (nb <= 3)),
                       ("isolated (0)", nb == 0)):
        m = masks["water"] & sel
        d = describe(celsius[m])
        if d:
            d["contrast_vs_ice_K"] = round(float(d["median"] - ice_median), 2)
            out[label] = d
    return out


def pixel_residual(celsius, masks, size=5):
    """
    Robust sigma of the residual after a 5x5 median filter, over ice only.

    NOT a noise-equivalent temperature difference, and it must not be used as one. The L2 product
    is a ~100 m TIRS footprint cubic-convolved onto the 30 m OLI grid, so neighbouring 30 m
    samples are largely the same measurement and this residual comes out near zero (a few
    hundredths of a kelvin) however noisy the instrument is. It is reported because it is the
    honest measure of how little independent information the 30 m sampling carries, which is the
    same fact the semivariogram shows. The clutter figure that actually governs detection is the
    ice-texture sigma over the window.
    """
    from scipy.ndimage import median_filter
    m = masks["ice"]
    if int(m.sum()) < 5000:
        return None
    resid = (celsius - median_filter(celsius, size=size))[m]
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return round(1.4826 * mad, 3)


def width_response(celsius, masks, gsd, ice_median):
    """
    The decisive curve: apparent thermal contrast as a function of how wide the channel really is.

    The geometry comes from QA_PIXEL's water mask, which is a 30 m optical classification and
    owes nothing to ST_B10, so binning by it is not circular the way binning by temperature
    would be. Width is 2 x the Euclidean distance transform, i.e. the local channel width in
    metres. If TIRS's 100 m footprint is dissolving narrow leads, this table is where it shows:
    the sub-100 m bins should fall away from the wide ones.
    """
    from scipy.ndimage import distance_transform_edt
    w = masks["water"]
    if int(w.sum()) < 200:
        return {}
    width = 2.0 * distance_transform_edt(w) * gsd
    bins = [(0, 60), (60, 90), (90, 120), (120, 180), (180, 300), (300, 600), (600, 1e9)]
    out = {}
    for lo, hi in bins:
        sel = w & (width > lo) & (width <= hi)
        n = int(sel.sum())
        if n < 40:
            continue
        v = celsius[sel]
        label = f"{lo}-{hi:.0f} m" if hi < 1e8 else ">600 m"
        out[label] = {"n": n, "median_C": round(float(np.median(v)), 2),
                      "p95_C": round(float(np.percentile(v, 95)), 2),
                      "contrast_vs_ice_K": round(float(np.median(v) - ice_median), 2)}
    return out


def semivariogram(field: np.ndarray, mask: np.ndarray, gsd: float, max_lag: int = 12):
    """gamma(h) = 0.5 * mean((z(x) - z(x+h))^2), averaged over the two axes. Returns (lags_m, gamma)."""
    f = np.where(mask, field, np.nan)
    lags, gam = [], []
    for h in range(1, max_lag + 1):
        d = np.concatenate([(f[:, h:] - f[:, :-h]).ravel(), (f[h:, :] - f[:-h, :]).ravel()])
        d = d[np.isfinite(d)]
        if d.size < 100:
            break
        lags.append(h * gsd)
        gam.append(0.5 * float(np.mean(d * d)))
    return np.array(lags), np.array(gam)


def half_sill_lag(lags, gam):
    """Lag (m) at which the semivariogram first reaches half its plateau. A resolution proxy."""
    if len(gam) < 3:
        return None
    sill = float(gam[-1])
    half = 0.5 * sill
    for i, g in enumerate(gam):
        if g >= half:
            if i == 0:
                return float(lags[0])
            t = (half - gam[i - 1]) / max(gam[i] - gam[i - 1], 1e-9)
            return round(float(lags[i - 1] + t * (lags[i] - lags[i - 1])), 1)
    return None


def warm_features(celsius, masks, gsd, ice_median, ice_std, k=3.0, min_px=8):
    """Segment warm anomalies and measure how lead-shaped they are."""
    from scipy.ndimage import label, distance_transform_edt, find_objects
    thr = ice_median + k * ice_std
    warm = masks["clear"] & (celsius >= thr)
    lab, n = label(warm, structure=np.ones((3, 3)))
    if n == 0:
        return {"threshold_C": round(float(thr), 2), "n_features": 0, "features": []}
    dt = distance_transform_edt(warm)
    feats = []
    for i, sl in enumerate(find_objects(lab), start=1):
        m = lab[sl] == i
        area = int(m.sum())
        if area < min_px:
            continue
        half_w = float(dt[sl][m].max())
        width_m = 2.0 * half_w * gsd
        # length proxy: a ribbon of area A and width w has length ~ A/w
        length_m = (area * gsd * gsd) / max(width_m, gsd)
        feats.append({"area_px": area, "width_m": round(width_m, 1),
                      "length_m": round(length_m, 1),
                      "elongation": round(length_m / max(width_m, 1e-6), 1),
                      "peak_C": round(float(celsius[sl][m].max()), 2),
                      "median_C": round(float(np.median(celsius[sl][m])), 2)})
    feats.sort(key=lambda f: -f["area_px"])
    thin = [f for f in feats if f["width_m"] <= 120.0]
    ribbons = [f for f in feats if f["elongation"] >= 4.0 and f["area_px"] >= 20]
    return {
        "threshold_C": round(float(thr), 2),
        "n_features": len(feats),
        "n_ribbonlike_elong_ge4": len(ribbons),
        "median_width_m": round(float(np.median([f["width_m"] for f in feats])), 1) if feats else None,
        "min_width_m": round(min(f["width_m"] for f in feats), 1) if feats else None,
        "thin_le120m_share": round(len(thin) / len(feats), 3) if feats else None,
        "thin_median_peak_C": round(float(np.median([f["peak_C"] for f in thin])), 2) if thin else None,
        "wide_median_peak_C": (round(float(np.median([f["peak_C"] for f in feats
                                                      if f["width_m"] > 120.0])), 2)
                               if any(f["width_m"] > 120.0 for f in feats) else None),
        "largest": feats[:6],
    }


def control_semivariogram(max_lag_m=360.0):
    """Same statistic on the 10 m Sentinel-2 NIR of data/real/arctic_probe, as a yardstick."""
    probe = os.path.join("data", "real", "arctic_probe")
    mpath = os.path.join(probe, "manifest.json")
    if not os.path.exists(mpath):
        return None
    import tifffile
    with open(mpath, encoding="utf-8") as f:
        man = json.load(f)
    out = {}
    for sc in man["scenes"]:
        p = os.path.join(probe, sc["file"])
        if not os.path.exists(p):
            continue
        nir = tifffile.imread(p)[..., 3].astype(np.float64)
        gsd = float(sc.get("gsd_meters", 10.0))
        mask = nir > 0
        lags, gam = semivariogram(nir, mask, gsd, max_lag=int(max_lag_m / gsd))
        out[sc["id"]] = {"gsd_m": gsd, "half_sill_lag_m": half_sill_lag(lags, gam)}
    return out


def contact_sheet(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.3 * n), squeeze=False)
    for r, (sid, title, celsius, masks) in enumerate(rows):
        c = np.where(masks["clear"], celsius, np.nan)
        finite = c[np.isfinite(c)]
        lo, hi = np.percentile(finite, [1, 99.7]) if finite.size else (-30, 0)
        ax = axes[r][0]
        im = ax.imshow(c, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="surface T (C)")
        ax = axes[r][1]
        for key, colour in (("ice", "tab:blue"), ("water", "tab:red"), ("other", "tab:grey")):
            v = celsius[masks[key]]
            if v.size > 200:
                ax.hist(v, bins=120, range=(float(lo) - 4, float(hi) + 4), histtype="step",
                        color=colour, label=f"{key} (n={v.size:,})", density=True)
        ax.axvline(SEAWATER_FREEZING_C, color="k", ls="--", lw=1, label="seawater freezing -1.8 C")
        ax.legend(fontsize=6); ax.set_xlabel("surface T (C)", fontsize=7)
        ax.tick_params(labelsize=6); ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def lead_transform_check(celsius, masks, gsd, ice_med, ice_std, n=400, seed=7):
    """
    Does the applet's OWN lead detector fire on the thermal field, with nothing changed?

    applet.pipelines.vessel_detector.VesselDetector._lead_transform hunts a contiguous run of
    DEFICIT contrast leaving a candidate on one bearing -- in VNIR a lead is dark. In the thermal
    a lead is bright, so the field is negated before it is handed over. That single sign flip is
    the whole adaptation: the ray table, the occupancy walk, the outlier test across bearings and
    the accept thresholds are the flight code, untouched.

    Two populations are sampled so the answer is a contrast and not just a hit rate:
      - candidates sitting in narrow (<=120 m) open water, i.e. where a vessel working a channel
        would be;
      - control candidates on ice more than 1 km from any water, where there is no channel to
        find and anything the transform reports is a false alarm.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from applet.config import AppletConfig
    from applet.pipelines.vessel_detector import VesselDetector

    if int(masks["water"].sum()) < 2000:
        return None
    det = VesselDetector(AppletConfig())
    z = np.clip((celsius - ice_med) / max(ice_std, 0.2), -3.0, 12.0)
    z_smooth = cv2.blur(-z.astype(np.float32), (3, 3)).astype(np.float64)

    w = masks["water"]
    width = 2.0 * distance_transform_edt(w) * gsd
    dist_to_water = distance_transform_edt(~w) * gsd
    narrow = np.argwhere(w & (width <= 120.0))
    far_ice = np.argwhere(masks["ice"] & (dist_to_water > 1000.0))
    rng = np.random.default_rng(seed)
    out = {}
    for label, pts in (("in_narrow_lead", narrow), ("control_open_ice", far_ice)):
        if len(pts) < 50:
            continue
        pick = pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]
        res = [det._lead_transform(z_smooth, float(x), float(y), gsd) for y, x in pick]
        snr = np.array([r["snr"] for r in res])
        out[label] = {
            "n": len(res),
            "found_rate": round(float(np.mean([r["found"] for r in res])), 3),
            "median_snr": round(float(np.median(snr)), 2),
            "p90_snr": round(float(np.percentile(snr, 90)), 2),
            "median_length_m": round(float(np.median([r["length_px"] for r in res]) * gsd), 1),
        }
    return out or None


def zoom_sheet(rows, path, crop_px=200):
    """
    Close-ups on the busiest patch of NARROW lead, at the scale a ship channel would occupy.

    The crop is centred where thin (<=120 m) optical water is densest, and is annotated with a
    1 km bar and with the 150 m hull length the detector is built around -- 5 pixels at 30 m.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter, distance_transform_edt
    panels = []
    for sid, title, celsius, masks in rows:
        if int(masks["water"].sum()) < 500:
            continue
        # Width in pixels, from the optical water mask. Score a crop by how much NARROW lead it
        # holds minus how much wide open water: otherwise the busiest patch of thin water is
        # always the ragged edge of a big polynya, which is not the case under test.
        wpx = 2.0 * distance_transform_edt(masks["water"])
        thin = masks["water"] & (wpx > 0) & (wpx <= 4.0)
        wide = masks["water"] & (wpx > 8.0)
        k = crop_px // 2
        dens = (uniform_filter(thin.astype(np.float32), size=k)
                - 0.5 * uniform_filter(wide.astype(np.float32), size=k))
        cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
        h, w = celsius.shape
        y0 = int(np.clip(cy - crop_px // 2, 0, max(h - crop_px, 0)))
        x0 = int(np.clip(cx - crop_px // 2, 0, max(w - crop_px, 0)))
        panels.append((sid, title, celsius[y0:y0 + crop_px, x0:x0 + crop_px],
                       masks["clear"][y0:y0 + crop_px, x0:x0 + crop_px]))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6), squeeze=False)
    for ax, (sid, title, c, clear) in zip(axes[0], panels):
        v = np.where(clear, c, np.nan)
        finite = v[np.isfinite(v)]
        lo, hi = np.percentile(finite, [2, 99])
        im = ax.imshow(v, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{title}\n{crop_px * 30 / 1000:.0f} km crop, 30 m pixels", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        n = c.shape[0]
        ax.plot([0.06 * n, 0.06 * n + 1000 / 30.0], [0.94 * n] * 2, "w-", lw=3)
        ax.text(0.06 * n, 0.915 * n, "1 km", color="w", fontsize=8)
        ax.plot([0.06 * n, 0.06 * n + 150 / 30.0], [0.87 * n] * 2, "c-", lw=3)
        ax.text(0.06 * n, 0.845 * n, "150 m hull (5 px)", color="c", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, label="surface T (C)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", help="a bundle made by scripts/fetch_landsat_thermal.py")
    ap.add_argument("--png", default=None, help="write <prefix>_sheet.png")
    ap.add_argument("--json", default=None, help="also dump every measurement here")
    ap.add_argument("--sigma", type=float, default=3.0, help="warm-anomaly threshold, in ice sigma")
    ap.add_argument("--lead-transform", action="store_true",
                    help="run the applet's own _lead_transform over the thermal field")
    ap.add_argument("--control", action="store_true",
                    help="also run the semivariogram on the 10 m Sentinel-2 arctic_probe")
    args = ap.parse_args()

    with open(os.path.join(args.bundle, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    report = {"bundle": args.bundle, "sensor": man.get("sensor"), "scenes": {}}
    rows = []

    for sc in man["scenes"]:
        sid = sc["id"]
        gsd = float(sc.get("gsd_meters", 30.0))
        celsius, qa, valid = load_scene(args.bundle, sc)
        masks = qa_masks(qa, valid)
        ice = celsius[masks["ice"]]
        water = celsius[masks["water"]]
        ice_med = float(np.median(ice)) if ice.size > 30 else float(np.median(celsius[masks["clear"]]))
        ice_std = float(ice.std()) if ice.size > 30 else float(celsius[masks["clear"]].std())

        lags, gam = semivariogram(celsius, masks["clear"], gsd)
        entry = {
            "description": sc["description"],
            "date": sc["shutter_time"][:10],
            "platform": sc["source"].get("platform"),
            "sun_elevation_deg": sc["source"].get("sun_elevation_deg"),
            "gsd_m": gsd,
            "window_km": round(celsius.shape[1] * gsd / 1000.0, 1),
            "class_fractions": {k: round(float(v.mean()), 4)
                                for k, v in masks.items() if k != "clear"},
            "temperature_C": {k: describe(celsius[masks[k]])
                              for k in ("ice", "water", "other", "clear")},
            "histogram_modes_C": modes(celsius[masks["clear"]]),
            "contrast_K": {},
            "mixed_pixel_dilution": neighbour_strata(celsius, masks, ice_med) if water.size > 100 else {},
            "contrast_by_optical_channel_width": width_response(celsius, masks, gsd, ice_med),
            "semivariogram": {"lags_m": [round(float(x), 1) for x in lags],
                              "gamma_K2": [round(float(g), 3) for g in gam],
                              "half_sill_lag_m": half_sill_lag(lags, gam)},
            "warm_features": warm_features(celsius, masks, gsd, ice_med, ice_std, k=args.sigma),
        }
        if ice.size > 30 and water.size > 30:
            entry["contrast_K"]["median_water_minus_median_ice"] = round(
                float(np.median(water) - np.median(ice)), 2)
            entry["contrast_K"]["p95_water_minus_median_ice"] = round(
                float(np.percentile(water, 95) - np.median(ice)), 2)
        entry["contrast_K"]["scene_max_minus_median_ice"] = round(
            float(celsius[masks["clear"]].max() - ice_med), 2)
        entry["contrast_K"]["ice_sigma_K"] = round(ice_std, 2)
        # See pixel_residual(): this is a smoothness diagnostic, not an NEdT, so no SNR is
        # derived from it. The detection-relevant ratio is against the ice texture.
        entry["contrast_K"]["px_to_px_residual_K_not_nedt"] = pixel_residual(celsius, masks)
        if ice.size > 30 and water.size > 30:
            entry["contrast_K"]["snr_vs_ice_texture"] = round(
                float(np.median(water) - np.median(ice)) / max(ice_std, 1e-6), 1)
        entry["contrast_K"]["warmest_clear_pixel_C"] = round(float(celsius[masks["clear"]].max()), 2)
        report["scenes"][sid] = entry
        rows.append((sid, f"{sid}  {entry['date']}  {sc['source'].get('platform')}  "
                          f"({entry['window_km']} km across)", celsius, masks))

        w = entry["temperature_C"]["water"]
        i = entry["temperature_C"]["ice"]
        print(f"\n=== {sid}  {entry['date']}  {entry['description'][:70]}")
        print(f"  window {entry['window_km']} km at {gsd:.0f} m  |  cloud "
              f"{entry['class_fractions']['cloud']:.1%}  water {entry['class_fractions']['water']:.1%}"
              f"  snow/ice {entry['class_fractions']['ice']:.1%}  other {entry['class_fractions']['other']:.1%}")
        if i:
            print(f"  ice    median {i['median']:7.2f} C   p5..p95 {i['p5']:7.2f}..{i['p95']:7.2f}  sigma {i['std']:.2f}")
        if w:
            print(f"  water  median {w['median']:7.2f} C   p5..p95 {w['p5']:7.2f}..{w['p95']:7.2f}  max {w['max']:.2f}")
        for k, v in entry["contrast_K"].items():
            print(f"    contrast {k}: {v:+.2f} K" if isinstance(v, float) else f"    {k}: {v}")
        if entry["mixed_pixel_dilution"]:
            print("  mixed-pixel dilution (apparent water T by enclosure):")
            for k, v in entry["mixed_pixel_dilution"].items():
                print(f"    {k:24s} n={v['n']:7,d}  median {v['median']:7.2f} C  "
                      f"vs ice {v['contrast_vs_ice_K']:+6.2f} K")
        if entry["contrast_by_optical_channel_width"]:
            print("  contrast vs channel width (width from the 30 m optical water mask):")
            for k, v in entry["contrast_by_optical_channel_width"].items():
                print(f"    {k:>10s}  n={v['n']:7,d}  median {v['median_C']:7.2f} C  "
                      f"vs ice {v['contrast_vs_ice_K']:+6.2f} K")
        if args.lead_transform:
            lt = lead_transform_check(celsius, masks, gsd, ice_med, ice_std)
            entry["applet_lead_transform"] = lt
            if lt:
                print("  applet _lead_transform on the negated thermal z-map (flight code, unmodified):")
                for k, v in lt.items():
                    print(f"    {k:18s} n={v['n']:4d}  found {v['found_rate']:.1%}  "
                          f"median snr {v['median_snr']:5.2f}  p90 {v['p90_snr']:5.2f}  "
                          f"median run {v['median_length_m']:.0f} m")
        wf = entry["warm_features"]
        print(f"  warm features > {wf['threshold_C']} C: {wf['n_features']} "
              f"(ribbon-like elong>=4: {wf.get('n_ribbonlike_elong_ge4')}), median width "
              f"{wf.get('median_width_m')} m, narrowest {wf.get('min_width_m')} m")
        if wf.get("thin_median_peak_C") is not None:
            print(f"    features <=120 m wide: {wf['thin_le120m_share']:.0%} of them, "
                  f"median peak {wf['thin_median_peak_C']} C "
                  f"(wider ones: {wf.get('wide_median_peak_C')} C)")
        print(f"  semivariogram half-sill lag {entry['semivariogram']['half_sill_lag_m']} m "
              f"(sampling {gsd:.0f} m)")
        print(f"  histogram modes (C, weight): {entry['histogram_modes_C']}")

    if args.control:
        ctl = control_semivariogram()
        report["control_sentinel2_nir"] = ctl
        if ctl:
            print("\n=== control: Sentinel-2 NIR, data/real/arctic_probe, same statistic")
            for sid, v in ctl.items():
                print(f"  {sid:18s} gsd {v['gsd_m']:.0f} m   half-sill lag {v['half_sill_lag_m']} m")

    if args.png:
        p = contact_sheet(rows, args.png + "_sheet.png")
        z = zoom_sheet(rows, args.png + "_zoom.png")
        if z:
            print(f"[png] {z}")
        print(f"\n[png] {p}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[json] {args.json}")


if __name__ == "__main__":
    sys.exit(main())
