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
| **CNN only on candidate chips** | Whitecaps, cloud puffs and islets share the ship's NIR signature; shape and context separate them, which is what a CNN is good at. On held-out scenes it removes **93 % of the physics stage's false alarms** while costing 2 % recall — for 0.2 ms per chip. |
| **Wake = a line, not a "V"** | At 4.75 m GSD the Kelvin cusp arms are rarely resolved; the turbulent centreline always is. Each candidate gets a 360-ray transform centred on the hull: the ray with contiguous excess contrast is the wake, the opposite bearing is the heading. A Kelvin-arm test (paired ridges at ±19.47° cast from the bow) is wired in but experimental: see limitations. |
| **Speed from wave physics, or not at all** | Transverse wake waves obey λ = 2πV²/g. The detector reports a speed only when a clean spectral line with ≥ 8 cycles exists astern **and is absent dead ahead** (the same line through the hull, so swell cancels). Otherwise `estimated_speed_knots` is `null` — never a guess. It is the least mature measurement here, so a speed disagreement with AIS is advisory and cannot raise an anomaly on its own. |
| **Reflectance units, metres not pixels** | Every threshold is in reflectance or metres, converted per scene from the manifest GSD. The same config runs 4.75 m HyperScape100 and 10 m Sentinel-2. |
| **ONNX Runtime** | One graph: CPU provider inside the emulated container (no GPU exists there), TensorRT/CUDA providers on real Jetson hardware, selected automatically. No code change, nothing to re-validate but speed. |

## Measured results

Synthetic scenes are rendered in reflectance with sub-pixel hulls, turbulent wakes, Kelvin arms, transverse
waves, swell, whitecaps, sunglint, fractal cloud, coast with surf and islets (`simulation/scene_synth.py`).
The verifier was trained on seed 2026, thresholds were tuned on seed 777, and the numbers below are from
**seed 4242, which was never looked at during development** (60 scenes, 130 visible vessels):

| | Physics only | + INT8 CNN verifier |
| :-- | --: | --: |
| Precision | 0.44 | **0.92** |
| Recall | 0.81 | **0.79** |
| F1 | 0.57 | **0.85** |
| Heading error (median / within 10°) | | 0.4° / 98.8 % |
| AIS anomaly class correct (dark / mismatch / known) | | 100 % of matched vessels |

Misses are dominated by vessels inside the land or cloud keep-out buffers and wake-less small craft in gales.
**These are synthetic numbers.** They prove the pipeline and the measurement harness; they are not a claim
about real imagery. See *Real data* below for the path to one.

Verifier (`applet/models/model_card.json`):

| | FP32 | INT8 (flight) |
| :-- | --: | --: |
| File size | 185 KB | **59 KB** |
| Validation AUC | 0.9875 | 0.9870 |
| Latency per chip (ORT CPU, x86 dev box) | 0.98 ms | **0.20 ms** |

Edge budget (native x86 dev box — *not* a Jetson timing claim; see `scripts/benchmark.py`):

| Bundle | Wall clock | Peak RAM | Raw → downlink |
| :-- | --: | --: | --: |
| 6 × 1024² scenes | 0.9 s | 0.41 GB (2.9 % of 14 GB) | 54 MB → 6.8 KB (8 000×) |
| + one 4096² full swath | 7.2 s | 1.43 GB (10 % of 14 GB) | 182 MB → 14 KB (13 000×) |

The tarball is **byte-identical across runs** (fixed mtimes, ordering and JPEG stretch) — tested.

## Quick start

```bash
pip install -r requirements-dev.txt
python scripts/generate_synthetic_data.py            # renders data/sample_bundle (seeded)
python -m applet run -i data/sample_bundle -o data/outputs -c config.example.yaml
python -m ground.server                              # GUI at http://127.0.0.1:8050
```

Other tools:

```bash
python -m pytest -q                                  # 31 tests: resilience, physics, determinism
python scripts/evaluate.py -i data/sample_bundle     # precision / recall / heading / AIS accuracy
python scripts/evaluate.py --no-verifier             # ...what the CNN buys
python scripts/generate_synthetic_data.py --random 60 --seed 4242 -o data/heldout_bundle
python scripts/benchmark.py --full-swath             # 4096×4096 timing + memory
python training/train_verifier.py --scenes 500       # re-mine chips, retrain, export, quantise
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

The verifier is trained on synthetic chips because no labelled 4.75 m VNIR ship set exists and none is
provided. The pipeline is ready for real imagery — Sentinel-2 L2A (B04, B03, B02, B08 at 10 m, scale 0.0001,
offset −0.1) drops into a bundle as-is, and `training/train_verifier.py --real-npz` mixes real labelled chips
into training. Openly downloadable candidates: *SEN2MS Vessel BBoxes* (Zenodo 15571607, 3 681 Sentinel-2 chips
with AIS-derived boxes) and Sentinel-2 COGs via the Element84 Earth Search STAC API.

## Honest limitations

* All accuracy figures are on synthetic scenes; real sunglint, ship-like rocks, platforms and ice will add
  failure modes the simulator does not have.
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
scripts/           bundle generator, evaluate, benchmark, container runners
tests/             31 tests
docker/            Dockerfile.arm64
docs/              hackathon rules, rubric, track notes, pitch template
```
