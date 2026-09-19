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
| **CNN only on candidate chips** | Whitecaps, cloud puffs and islets share the ship's NIR signature; shape and context separate them, which is what a CNN is good at. On held-out synthetic scenes it removes 93 % of the physics stage's false alarms for 2 % recall; on four real scenes held out of its training it removes 66 % of them for 8 % of the real vessels — for ~0.09 ms per chip. It only learned to do that on real imagery after being trained on hand-adjudicated real false alarms; the synthetic-trained version scored genuine vessels 0.999 and clutter 0.985 and therefore rejected nothing. |
| **Wake = a line, not a "V"** | At 4.75 m GSD the Kelvin cusp arms are rarely resolved; the turbulent centreline always is. Each candidate gets a 360-ray transform centred on the hull: the ray with contiguous excess contrast is the wake, the opposite bearing is the heading. A Kelvin-arm test (paired ridges at ±19.47° cast from the bow) is wired in but experimental: see limitations. |
| **Speed from wave physics, or not at all** | Transverse wake waves obey λ = 2πV²/g. The detector reports a speed only when a clean spectral line with ≥ 8 cycles exists astern **and is absent dead ahead** (the same line through the hull, so swell cancels). Otherwise `estimated_speed_knots` is `null` — never a guess. It is the least mature measurement here, so a speed disagreement with AIS is advisory and cannot raise an anomaly on its own. |
| **Reflectance units, metres not pixels** | Every threshold is in reflectance or metres, converted per scene from the manifest GSD. The same config runs 4.75 m HyperScape100 and 10 m Sentinel-2. |
| **ONNX Runtime** | One graph: CPU provider inside the emulated container (no GPU exists there), TensorRT/CUDA providers on real Jetson hardware, selected automatically. No code change, nothing to re-validate but speed. |

## Measured results

### 16 real scenes against real AIS the imagery never saw

The primary benchmark. Sixteen Sentinel-2 L2A scenes across every US coast, each paired with the **same
day's public NOAA Marine Cadastre AIS** (1,049 broadcasters inside the footprints). Every scene is
georeferenced from its own COG transform, so nothing about the imagery knows where the ships are — unlike
chip datasets, where the position match is true by construction. Then **every one of the 571 contacts was
adjudicated by eye** against the pixels, because AIS can prove a contact is a ship but cannot prove one
isn't. Build it with `scripts/fetch_sentinel2.py --region us` + `scripts/fetch_noaa_ais.py`, score it with
`scripts/scorecard.py`.

| | Before retraining | **Flight model** |
| :-- | --: | --: |
| Recall vs AIS the sensor could see (n = 136–141) | 0.922 | **0.912** |
| Precision, adjudicated by hand | 0.469 | **0.693** |
| False alarms | 286 | **107** (18.9 per 1000 km²) |
| Position error vs dead-reckoned AIS | median 46 m | median **46 m**, 96.8 % within 200 m |
| Heading vs reported COG | median 5.8° | median **5.4°**, 77 % within 20° |
| Runtime / peak RAM, 5,669 km² of water | 13.3 s / 2.39 GB | **10.4 s / 2.50 GB** |

Recall is quoted against the broadcasters the sensor could **actually see** — clear water, resolvable,
not under cloud, outside the 200 m shoreline keep-out. 887 of the 1,049 are alongside quays inside
harbours, which this design drops on purpose; counting them as misses would understate a number we can
otherwise defend. Of the ships reported missing in clear water, the reviewers confirmed several had no
vessel visible at all — stale or ghost fixes, which are excluded rather than charged to the detector.

**The retraining generalised; it did not memorise.** Four scenes were held out of training entirely, and
they improved as much as the twelve that were trained on:

| | Precision before | after | False alarms |
| :-- | --: | --: | --: |
| **Held out** (Tampa, Long Beach, Puget Sound, New York) | 0.503 | **0.731** | 85 → 29 |
| Trained (the other 12) | 0.454 | **0.676** | 201 → 78 |

Tampa is the sharpest case: **0.202 → 0.467**, false alarms 67 → 16, on a scene the model never saw.
Miami — also a shallow turbid bay — stayed in training, so this measures whether one bay's clutter
teaches another's. Every scene improved or held; none got worse. The mechanism is visible in the scores:
the old verifier gave vessels 0.999 and clutter 0.985, so at its 0.80 threshold it rejected nothing. The
retrained one puts clutter at 0.606 (AUC 0.744 → 0.920 on held-out chips).

### Real chips (SEN2MS, 10 m)

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
| Precision | 0.43 | **0.93** |
| Recall | 0.78 | **0.76** |
| F1 | 0.55 | **0.84** |
| Heading error (median / within 10 deg) | | 0.5 deg / 98.8 % |
| AIS anomaly class correct (dark / mismatch / known) | | 97 % of matched vessels |

Retraining the verifier on real adjudicated clutter did not cost anything here: synthetic precision rose
from 0.90 to 0.93 for 0.01 of recall. The model learned what real false alarms look like without
unlearning the synthetic distribution it was already good at.

Synthetic scenes are where AIS spoofing (false course, false "at anchor", ghost transponders) can be scored,
because real spoofers do not come labelled. Misses are dominated by vessels inside the land / cloud keep-outs and wake-less small craft in gales.

### Verifier (`applet/models/model_card.json`)

| | FP32 | INT8 (flight) |
| :-- | --: | --: |
| File size | 185 KB | **59 KB** |
| Validation AUC (synthetic + real chips) | 0.978 | 0.979 |
| **AUC on held-out real scenes** (171 adjudicated chips) | | **0.920** (was 0.744) |
| Latency per chip (ORT CPU, x86 dev box) | 0.10 ms | **0.09 ms** |

Trained on 11,607 mined synthetic chips + 3,608 SEN2MS crops + 368 hand-adjudicated chips from 12 real
scenes (oversampled ×8, or a few hundred real hard negatives are swamped). The held-out AUC is the honest
one: in-distribution validation AUC barely moved (0.984 → 0.979) while real-scene AUC went 0.744 → 0.920,
which is exactly what fixing an out-of-distribution failure looks like. Reproduce with
`python training/eval_verifier.py --models applet/models/verifier_int8.onnx`.

### Edge budget

Measured on **native ARM64** (Apple M4, 6 threads via `OMP_NUM_THREADS` to mirror the judges'
`--cpus=6`) and on an x86-64 dev box. `scripts/run_on_macmini.sh --native` reproduces the ARM64 column
over a tailnet; `scripts/benchmark.py` the rest.

| Workload | ARM64 (M4, 6 threads) | x86-64 dev box | Peak RAM (ARM64) | Raw -> downlink |
| :-- | --: | --: | --: | --: |
| One 4096² full swath (23.1 Mpx, 19.4 km) | **1.61 s** ±0.02 | 7.2 s | 2.09 GB (14.6 % of 14 GB) | 182 MB -> 9.4 KB (19 900x) |
| 16 × 2048² real Sentinel-2 scenes, 5 669 km² | **3.03 s** | 10.4 s | 2.59 GB (18 % of 14 GB) | 316 MB -> 11 KB |
| Full test suite | **2.3 s** | 8.0 s | | |

Per stage on the full swath: detector 0.83 s, screener 0.38 s, ingest 0.27 s, **CNN 0.085 s**, AIS
0.029 s, packaging 0.004 s. Only 2.24 of 6 cores are busy, so there is headroom we have not spent.

**This is not a Jetson prediction.** An M4 is far quicker than an Orin NX's Cortex-A78AE cores, so treat
the ARM64 column as an upper bound on ARM performance and a proof that the code is correct and fast
enough on aarch64 — not as the flight figure. What it does establish is that nothing here depends on
x86: same ONNX graph, same OpenCV calls, same answers.

**Bit-exact across architectures.** The same input bundle produces a downlink tarball with the *same
SHA-256* on x86-64 Windows and ARM64 macOS — `eacda3b0dd9a64…` — including JPEG encoding and INT8 ONNX
inference, and every scorecard metric matches to the digit (recall 0.912, precision 0.693, position
error 46.3 m). Reproducibility is usually claimed across runs; this holds across instruction sets.

### Duty cycle: what a day in orbit actually asks of it

A benchmark measures one pass. A satellite runs for months, so the question is whether the processor
keeps up with the sensor and what it costs over a day. From the brief (4.75 m GSD at 500 km, 19.4 km
swath, up to 1 Tbit/day, 5–12 min ground passes at ~50 Mbps):

| | |
| :-- | --: |
| Ground track speed at 500 km | 7.06 km/s |
| One 4096² frame (376 km²) takes | **2.75 s to capture** |
| We process one in | **1.58 s** (M4) — **1.74× real time** |
| 1 Tbit/day works out to | 1,242 frames = **57 min of imaging/day (4 % duty cycle)** |
| Compute busy per day | **33 min** on an M4; 2.7 h even at 5× slower |
| Spare RAM for a backlog | 11.9 GB = 118 frames = **5.4 min** of unprocessed imaging |

The margin that matters is **2.75 s per frame**: anything slower than that cannot keep up while the
sensor is running and must buffer. We are at 1.58 s, so hardware up to 1.7× slower than an M4 still
runs in real time, and the 14 GB envelope absorbs a 5-minute backlog if it doesn't. The processor is
idle ~98 % of the day either way — this workload is bursty, not sustained, which is the right shape
for a power-limited bus.

And the bottleneck it exists to solve, in one line: **102 Gbit/day of downlink capacity against
1,000 Gbit/day of imagery — 10 %.** Our intelligence product is 11.6 MB/day, which is **1.9 seconds of
a single ground pass**, or 0.09 % of daily capacity.

**Sustained operation is tested, not assumed** (`scripts/soak.py`). Every other number here comes from
a fresh process that exits; a real applet is long-lived, which is where leaks and thermal creep hide.
120 consecutive passes over the 16 real scenes in **one process** on the M4:

| | |
| :-- | --: |
| Latency drift, first quartile → last | 2.641 s → 2.653 s (**+0.5 %**) |
| RSS | 1,619 → 1,677 MB (**+58 MB**, 3.6 %, allocator not leak) |
| Contacts per pass | 376, every pass |
| Distinct downlink tarballs | **1** — byte-identical across all 120 |

```bash
python scripts/soak.py -i data/real/s2_us_bundle -n 120 --quiet
```

It exits non-zero on a leak, on >15 % slowdown, or if the downlink stops being reproducible, so it
works as a regression test rather than a one-off demonstration.

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
python -m pytest -q                                  # 59 tests: resilience, physics, determinism, flight-image closure
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

Builds `docker/Dockerfile.arm64` (Ubuntu 22.04 / aarch64 / Python 3.10, flight requirements only, versions
pinned in `docker/constraints.txt`) and runs it
with `--memory=14g --memory-swap=14g --cpus=6 --network none`. On Apple Silicon this is native ARM64 and is
the right place to take timing numbers; under QEMU on x86 it is a does-it-fit check only, exactly as the
organisers' prep guide says. Nobody on the team has Docker on an ARM64 host, so the image is built and started
under QEMU by CI (`.github/workflows/tests.yml`), and `tests/test_flight_image.py` stages exactly the files the
Dockerfile copies and runs a pass from them, so flight code cannot import something the image does not ship.

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

Beside it, `queues/priority` (no AIS match) and `queues/nonPriority` hold a wider context crop of every
target for the file-queue downlink scheduler in `src/pyFlows` (three priority images per non-priority one).
They are not part of the tarball or its byte budget; `downlink.write_queues: false` turns them off and
`downlink.queue_dir` points them at a running scheduler, which is what the ground console does.

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

* Real-image validation is at 10 m (Sentinel-2), not the 4.75 m target GSD. Real false alarms (thin cloud,
  whitecaps, shoals, jetties) are still ~19 per 1000 km2 after retraining, down from ~50.
* **Shallow turbid water is the worst case and remains so.** Precision ranges from 0.93 in Puget Sound to 0.47
  in Tampa Bay, where sandbars and shoals dominate. Retraining more than doubled Tampa but did not fix it.
* Precision rests on human adjudication of 539 contacts, which is a judgement about what is visible in a 10 m
  pixel, not an independent survey. It is well calibrated where we can check it — of 30 AIS-confirmed contacts
  a reviewer judged, 28 were called vessels — but everyone labelled anonymously, so inter-rater agreement was
  never measured. Recall, by contrast, is scored against AIS and needs no human judgement.
* The verifier learned real clutter from 368 chips of one sensor in one season; a different sea state, sun
  angle or latitude is untested.
* About a quarter of headings are flipped 180 degrees: the wake gives the axis reliably, and the bow-versus-stern
  decision on that axis is the weak step.
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
tests/             59 tests
docker/            Dockerfile.arm64
docs/              SETUP (collaborators start here), hackathon rules, rubric, track notes, pitch template
```
