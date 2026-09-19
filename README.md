# Tactical Edge Sentinel

**Onboard dark-vessel detection for a 4.75 m VNIR nanosatellite — Pulsar SpaceHack 2026, Track 1.**

A 19 km swath is ~180 MB of raw samples and the ground pass is minutes long. This applet runs on the
satellite's Jetson Orin NX, finds every vessel in the scene, measures its heading (and speed when the
wake allows it), checks it against the uplinked AIS picture, and downlinks a **~10 KB** bundle that says
*which ships are not broadcasting, which are lying about their course, and which broadcasters are not
where they claim to be* — instead of the image.

```
 raw 4-band scene (R,G,B,NIR, 16-bit)
        │
 1  ImageQualityScreener   NDWI water / whiteness cloud / land masks, usability verdict      CPU, OpenCV
        │  cloudy, empty or corrupt scenes stop here — no further compute spent
 2  VesselDetector         ring-CFAR on NIR → blobs → hull / wake ray transform → physics     CPU, OpenCV
        │  a few dozen 64×64 chips, not megapixels
 3  ChipVerifier           47k-parameter INT8 CNN through ONNX Runtime                        CPU / TensorRT
        │
 4  AISKinematicCorrelator dead-reckon AIS to shutter time, 1:1 gate, classify anomalies
        │
 5  DownlinkPackager       GeoJSON + scene report + priority-ordered JPEG chips, byte-reproducible
```

## Why it is built this way

| Decision | Reason |
| :-- | :-- |
| **Physics first, CNN second** | Water reflects ~1 % in NIR; steel and foam 10–40 %. A local-contrast (CFAR) test finds a 10-pixel hull for a few OpenCV calls. Running a detector network over 16 Mpx of empty ocean would cost orders of magnitude more energy to reach the same candidates. |
| **CNN only on candidate chips** | Whitecaps, cloud puffs and islets share the ship's NIR signature; shape and context separate them, which is what a CNN is good at. On held-out synthetic scenes it removes 92 % of the physics stage's false alarms for 1 % recall; on held-out real Sentinel-2 it halves them and loses no real ship — for ~0.1 ms per chip. |
| **Wake = a line, not a "V"** | At 4.75 m GSD the Kelvin cusp arms are rarely resolved; the turbulent centreline always is. Each candidate gets a 360-ray transform centred on the hull: the ray with contiguous excess contrast is the wake, the opposite bearing is the heading. A Kelvin-arm test (paired ridges at ±19.47° cast from the bow) is wired in but experimental: see limitations. |
| **Speed from wave physics, or not at all** | Transverse wake waves obey λ = 2πV²/g. The detector reports a speed only when a clean spectral line with ≥ 8 cycles exists astern **and is absent dead ahead** (the same line through the hull, so swell cancels). Otherwise `estimated_speed_knots` is `null` — never a guess. It is the least mature measurement here, so a speed disagreement with AIS is advisory and cannot raise an anomaly on its own. |
| **Reflectance units, metres not pixels** | Every threshold is in reflectance or metres, converted per scene from the manifest GSD. The same config runs 4.75 m HyperScape100 and 10 m Sentinel-2. |
| **ONNX Runtime** | One graph: CPU provider inside the emulated container (no GPU exists there), TensorRT/CUDA providers on real Jetson hardware, selected automatically. No code change, nothing to re-validate but speed. |

## Measured results

### Real imagery (Sentinel-2, 10 m)

Scored on **SEN2MS Vessel BBoxes** (Dalhousie, CC-BY-4.0): real Sentinel-2 chips whose vessel boxes were
derived from AIS, so each ship comes with a true heading, speed and length. Sentinel-2 products are split
train / test (`crc32(product) % 4`), all tuning was done on train products, and these numbers are from the
**held-out test products** (225 vessel chips, 760 chips with no AIS vessel):

| | Physics only | + INT8 CNN verifier |
| :-- | --: | --: |
| Recall, open-water chips | **0.82** (107 / 131) | **0.82** (no real ship lost to the CNN) |
| Recall, ships under way (AIS SOG >= 3 kn) | **0.83** (75 / 90) | **0.83** |
| Recall, moored (SOG < 1 kn) | 0.48 | 0.48 |
| Recall, every labelled vessel incl. in-port | 0.32 | 0.32 |
| False alarms per 1000 km2 of scenes with no AIS vessel | 73 | **37** |
| Heading vs AIS heading (n = 51) | | median **4 deg**, 86 % within 20 deg, 4 % flipped 180 deg |
| Hull length vs AIS length | | 32 % mean error |

How to read this honestly: SEN2MS is dominated by ships **alongside quays inside harbours**; the detector's
200 m shoreline keep-out drops those on purpose (a moored ship is not a dark contact at sea), which is why the
all-vessel recall is 0.32 while open-water and under-way recall is 0.82-0.83. The false-alarm figure is an upper
bound: "no AIS vessel" does not mean "no vessel", and several of the alarms are visibly small craft. Remaining
real false alarms are wispy cumulus, whitecaps, shoals and jetties. The published arrays are percentile-stretched
per product rather than radiometric, so `training/sen2ms.py` re-anchors each chip's water level before use.

**Real AIS against real imagery** (`python training/sen2ms.py bundle`; held-out products, 141 single-vessel
chips, each ship's true AIS fix dead-reckoned across the real fix-to-shutter gap, every third ship's AIS withheld):

| Real ships the detector found | Outcome |
| :-- | :-- |
| 36 with AIS withheld (genuinely "dark" to the applet) | **35 reported DARK_VESSEL**, 1 confirmed |
| 64 broadcasting honestly | **60 CONFIRMED**, 4 flagged kinematic mismatch |
| ... of which 47 under way | 4 false accusations (8.5 %): wake heading disagreed with AIS heading by > 35 deg |

Caveat, stated plainly: the published chips have no geotransform, so each is geolocated by pinning its box to the
AIS position; the *position* match is true by construction. What this measures is the kinematic check and the
dark-vessel path on real wakes. It also caught a real bug: a day-old fix from the same berth "confirmed" a silent
ship, so AIS fixes older than `max_fix_age_hours` (3 h) can no longer identify a contact. The five full Sentinel-2
scenes have no AIS at all (no open archive covers them), so every contact there is reported dark.

**A full real scene with real AIS** (`scripts/fetch_noaa_ais.py`): Sentinel-2 over Los Angeles / Long Beach,
2024-11-08 18:45 UTC, against that day's public NOAA Marine Cadastre AIS (9 million fixes streamed from the zip,
276 broadcasters inside the footprint within 20 min of the shutter, median fix age 51 s). Here the geolocation is
independent of the AIS, so the position match is a real test:

| 43 contacts in 1.2 s | |
| :-- | :-- |
| CONFIRMED against a real MMSI | **16**, each 6-100 m from its dead-reckoned AIS position; every anchored 150-240 m ship among them |
| DARK_VESSEL | 25: almost all small craft under way with clear wakes and no transponder (legal, and exactly what a dark-vessel product should surface); one breakwater tip |
| AIS_KINEMATIC_MISMATCH | 1: a fireboat in a hard turn (curved wake vs instantaneous COG) |
| KNOWN_STRUCTURE | 1: a 184 m artificial oil island, matched from the uplinked `known_structures.json` |
| AIS broadcasters with no contact | 228, of which 226 are in port / inside the shoreline keep-out and **2** are in clear water (the only kind downlinked) |
| 32 MB raw -> downlink | 11 KB |

That scene drove four fixes, each with a regression test: piers beside berthed ships read as short wakes and
falsely accused three moored ships ("wake but AIS says stopped" now needs >= 300 m of strong wake); aircraft
appear as separated red/green/blue dots because the bands are exposed at different instants (now rejected by a
band-parallax test); "AIS not observed" is only meaningful in clear water for a vessel the sensor can resolve; and
charted structures need an uplinked list.

The first CNN, trained on synthetic chips only, **rejected 13 of 124 real ships** and accepted coastline
fragments. Retraining on synthetic + 3 608 mined real chips (train products only) fixed both: zero real ships
lost on the test products and half the false alarms. That sim-to-real gap is the single most important thing
the real data taught us.

Five full 20 km Sentinel-2 L2A scenes (Gibraltar, Suez, Long Beach, Halifax, Dover;
`scripts/fetch_sentinel2.py`) run end-to-end in 3.5 s for 21 Mpx and are selectable in the ground console. In
the Gibraltar scene 28 of the 30 reported contacts are unmistakably vessels, each placed at the head of its wake.

### Synthetic scenes (4.75 m)

Rendered in reflectance with sub-pixel hulls, turbulent wakes, Kelvin arms, transverse waves, swell, whitecaps,
sunglint, fractal cloud, coast with surf and islets (`simulation/scene_synth.py`). Verifier trained on seed 2026,
thresholds tuned on seed 777, numbers below from **seed 4242, never used in development** (60 scenes, 130 vessels):

| | Physics only | + INT8 CNN verifier |
| :-- | --: | --: |
| Precision | 0.43 | **0.90** |
| Recall | 0.78 | **0.77** |
| F1 | 0.55 | **0.83** |
| Heading error (median / within 10 deg) | | 0.5 deg / 97.6 % |
| AIS anomaly class correct (dark / mismatch / known) | | 97 % of matched vessels |

Synthetic scenes are where AIS spoofing (false course, false "at anchor", ghost transponders) can be scored,
because real spoofers do not come labelled. Misses are dominated by vessels inside the land / cloud keep-outs and wake-less small craft in gales.

### Verifier (`applet/models/model_card.json`)

| | FP32 | INT8 (flight) |
| :-- | --: | --: |
| File size | 185 KB | **59 KB** |
| Validation AUC (synthetic + real chips) | 0.982 | 0.984 |
| Latency per chip (ORT CPU, x86 dev box) | 0.14 ms | **0.09 ms** |

### Edge budget

Native x86 dev box -- *not* a Jetson timing claim; see `scripts/benchmark.py`:

| Bundle | Wall clock | Peak RAM | Raw -> downlink |
| :-- | --: | --: | --: |
| 6 x 1024^2 synthetic scenes | 0.9 s | 0.41 GB (2.9 % of 14 GB) | 54 MB -> 6.8 KB (8 000x) |
| + one 4096^2 full swath | 7.2 s | 1.43 GB (10 % of 14 GB) | 182 MB -> 14 KB (13 000x) |
| 5 x 2048^2 real Sentinel-2 scenes | 3.3 s | 1.2 GB (8.3 % of 14 GB) | 160 MB -> 54 KB incl. 180 target chips (2 900x) |

The tarball is **byte-identical across runs** (fixed mtimes, ordering and JPEG stretch) -- tested.

## Quick start

```bash
pip install -r requirements-dev.txt
python scripts/setup_data.py                         # synthetic bundles (add --all for the real datasets)
python -m applet run -i data/sample_bundle -o data/outputs -c config.example.yaml
python -m ground.server                              # GUI at http://127.0.0.1:8050
```

New to the repo? **[docs/SETUP.md](docs/SETUP.md)** walks through install, every dataset (one command, no
accounts, optional `--data-root` to keep the ~1.5 GB on another drive), reproducing the numbers, retraining and
the container.

Other tools:

```bash
python -m pytest -q                                  # 37 tests: resilience, physics, determinism
python scripts/evaluate.py -i data/sample_bundle     # precision / recall / heading / AIS accuracy
python scripts/evaluate.py --no-verifier             # ...what the CNN buys
python scripts/generate_synthetic_data.py --random 60 --seed 4242 -o data/heldout_bundle
python scripts/benchmark.py --full-swath             # 4096×4096 timing + memory
python training/train_verifier.py --scenes 500       # retrain, export, quantise (needs requirements-train.txt)
```

### Ground console

`python -m ground.server` runs the *unmodified* onboard code path and visualises every stage: true colour /
NIR / false colour / NDWI / CFAR z-score layers, cloud–land–detection masks, targets with heading arrows and
wake segments, dead-reckoned AIS positions, ground truth, CNN rejects, the downlink queue with the actual
JPEG chips, per-stage latency, RAM against the 14 GB envelope, the detection funnel and live accuracy. The
CFAR threshold, physics threshold, cloud limit and CNN on/off are sliders, so the precision/recall trade can
be shown live. No CDN or internet resources are used.

### Emulated Jetson container

```bash
./scripts/run_emulated.sh          # or .\scripts\run_emulated.ps1
```

Builds `docker/Dockerfile.arm64` (Ubuntu 22.04 / aarch64 / Python 3.10, flight requirements only) and runs it
with `--memory=14g --memory-swap=14g --cpus=6 --network none`. On Apple Silicon this is native ARM64 and is
the right place to take timing numbers; under QEMU on x86 it is a does-it-fit check only, exactly as the
organisers' prep guide says.

## Input bundle

```
bundle/
  manifest.json       gsd_meters, reflectance_scale, bands, shutter_time, scenes[{id, file, center_lat/lon | corners}]
  ais_catalog.json    vessels[{mmsi, name, timestamp, latitude, longitude, sog_knots, cog_deg}]
  known_structures.json   optional: structures[{name, latitude, longitude, radius_m}] (platforms, islands, buoys)
  *.tif | *.npy | *.png   4-band rasters, any bit depth, (H,W,C) or (C,H,W)
```

Georeferencing is either a centre point (north-up) or four `corners` (any projection or rotation; pixel→lat/lon
is a bilinear blend, so no GDAL/pyproj onboard). A `ground_truth` block may be present for scoring; the
validator strips it before the pipeline sees the scene.

Resilience (all tested): missing directory, corrupt or zero-byte files, NaN/Inf samples, dead bands, a missing
NIR band (red substitutes, the scene is flagged, detection continues), path traversal in the manifest, fully
clouded scenes, scenes with no water. One bad scene never takes down the pass.

## Output

`downlink_<pass>.tar.gz` → `tactical_intelligence.geojson` (one feature per target: class, priority, heading,
speed, hull size, wake length, physics / CNN / fused confidence, matched MMSI, plain-language note),
`scene_report.json` (quality verdicts, funnel), `manifest.json`, and `chips/*.jpg` for anomalies only, highest
priority first, until the byte budget is spent. Known, AIS-consistent traffic never earns a chip.
`edge_telemetry.json` (timings, RAM, cores) is written beside the tarball.

## Real data

```bash
python scripts/fetch_sentinel2.py -o data/real/s2_bundle      # 5 x 20 km Sentinel-2 L2A windows, ~95 MB, no login
python training/sen2ms.py mine                                # real chips from SEN2MS train products
python training/train_verifier.py --real-npz data/real/sen2ms/train_chips.npz
python training/sen2ms.py evaluate                            # scorecard on held-out products
python training/sen2ms.py bundle                              # real chips + their real AIS -> bundle + AIS scorecard
python scripts/fetch_noaa_ais.py --bundle data/real/s2_ais_bundle   # real AIS for a US scene (NOAA, ~360 MB/day)
```

`fetch_sentinel2.py` finds the least cloudy recent scene per area through the public Element84 Earth Search STAC
API and range-reads only B04/B03/B02/B08 windows from the public COGs. SEN2MS is one 565 MB zip from Zenodo
(record 15571607) placed at `data/real/sen2ms/`; it is read in place. Everything under `data/real/` is gitignored.
Contains modified Copernicus Sentinel data. MASATI (Gallego et al. 2018, Alashhab et al. 2019; research use only,
RGB, no NIR) is a candidate second source of real hard negatives.

## Honest limitations

* Real-image validation is at 10 m (Sentinel-2), not the 4.75 m target GSD, and on a dataset whose radiometry had
  to be repaired at ingest. Real false alarms (thin cloud, whitecaps, shoals, jetties) are still ~37 per 1000 km2.
* Ships moored alongside, or within 200 m of the land mask, are not reported (by design).
* No Jetson was available: TensorRT engines, DLA, shared CPU/GPU memory contention, power and thermals are
  unvalidated. The design minimises exposure (CPU-only path is the verified one; the model is 59 KB).
* The Kelvin-arm measurement has not yet fired, even on synthetic ships rendered with arms: the CFAR background
  sigma is inflated next to a bright hull, which buries the faint cusp lines (z ~ 0.5). `kelvin_arms_detected`
  is therefore always `false` today; nothing downstream depends on it.
* Wake-derived speed is reported for only a few percent of vessels (large, fast, clean wake); when reported it is
  usually within 1 kn but has produced a 2x error on one long-wake case, hence advisory-only.
* Hull length reads ~20 % short (the sub-pixel bow taper falls below the core threshold) and is a rough
  estimate for craft whose wake outshines the hull (`hull_resolved: false`).
* Wake-less vessels under ~25 m in a gale are indistinguishable from whitecaps and are dropped by design.
* Vessels within the 60 m land buffer or 40 m cloud buffer are not reported.

## Layout

```
applet/            flight code: config, cli, runner, core/ (validator, telemetry), pipelines/, packaging/, models/
simulation/        scene renderer (ground-side)
training/          verifier training, ONNX export, INT8 quantisation
ground/            FastAPI + single-page console
scripts/           setup_data (start here), bundle generator, Sentinel-2 / NOAA fetchers, evaluate, benchmark
tests/             37 tests
docker/            Dockerfile.arm64
docs/              SETUP (collaborators start here), hackathon rules, rubric, track notes, pitch template
```
