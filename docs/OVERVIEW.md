# IRIS -- how it works

Pulsar SpaceHack 2026, Track 1 (free-for-all). This is the explanation. Every measured number lives in
[`RESULTS.md`](RESULTS.md), the judge-facing summary is [`RUBRIC_SPEC_SHEET.md`](RUBRIC_SPEC_SHEET.md),
and the narrative is in [`README.md`](../README.md).

## The idea in one paragraph

A small Earth-observation satellite sees a ground station for a few minutes per orbit and images far
more than it can send down. A ship that has switched its AIS transponder off is only interesting *now*:
at 15 knots it is 14 km away half an hour later. So instead of downlinking a 182 MB image and finding
the ship on the ground hours later, the applet finds every vessel onboard, works out which way each is
heading, compares them against the AIS picture it was uplinked before the pass, and sends down about
**10 KB** that says *which ships are not broadcasting, which are lying about their course, and which
broadcasters are not where they claim to be*.

## What we assumed, and why

The organisers name no sensor, so we chose one and say so: a **Simera Sense HyperScape100** -- 4.75 m
GSD at 500 km, 19.4 km swath (4096 px), VNIR 460-860 nm, no SWIR. Galaxia Mission Systems (Halifax)
announced buying one on 3 September 2026 and names dark-vessel tracking as a mission
([`GALAXIA_ALIGNMENT.md`](GALAXIA_ALIGNMENT.md)). Two consequences run through the whole design:

* **No SWIR** means no NDSI, so sea ice has to be told from cloud inside the visible/NIR range alone.
* **Every threshold is in reflectance or metres**, never pixels or digital numbers, so the same config
  runs the 4.75 m target sensor and the 10 m Sentinel-2 imagery we actually validate on.

Compute target is an NVIDIA Jetson Orin NX 16 GB. Nobody was given one; the judging environment is an
emulated `linux/arm64` container with 14 GB, 6 CPUs, no GPU and no network, and that is what CI builds
and runs on every push.

## The three parts

```
            ONBOARD (applet/)                          GROUND (ground/, src/pyFlows/)
 input bundle                                  downlink_<pass>.tar.gz ──► console: map, queue, funnel,
 ├ manifest.json                                 (GeoJSON + report + chips)  telemetry, thermal panel
 ├ ais_catalog.json   ──► run_pass() ──►       queues/priority|nonPriority ──► timed, encrypted downlink
 ├ known_structures.json   five stages           (context crops)               scheduler ──► fleet alerts
 └ 4-band rasters                               edge_telemetry.json ──► RAM / CPU / modelled temperature
```

* **`applet/`** is the flight code and the only thing that has to run in the container. One entry
  point, `run_pass()` in [`applet/runner.py`](../applet/runner.py), is used identically by the CLI, the
  ground console, the benchmarks and every test -- there is no separate "demo path".
* **`ground/`** is a FastAPI server and a single-page console that runs that same code path and draws
  what it did. No CDN, no internet.
* **`src/pyFlows/`** models the spacecraft side of the downlink. The pass writes context crops into
  `queues/priority` and `queues/nonPriority`; a rate-limited, AES-256-GCM encrypted sender drains them
  three priority items to one non-priority, and delivered dark-vessel alerts are routed to the nearest
  response fleet on the ground.

## The cascade, stage by stage

The ordering is the design. Each stage is cheap relative to the next and hands it far less data, so the
expensive step (a neural network) only ever sees a few dozen small chips.

### 0. Ingest and validate -- `applet/core/validator.py`
Reads the manifest, loads each raster whatever its bit depth or axis order, converts to reflectance, and
builds a georeference from either a centre point or four corners (a bilinear blend -- no GDAL or pyproj
onboard). Anything wrong with one scene -- corrupt file, NaNs, a dead band, a missing NIR band, a path
that tries to escape the bundle -- is recorded against that scene and the pass carries on. Any ground
truth in the bundle is stripped here so nothing downstream can see it.

### 1. Quality screener -- `applet/pipelines/quality_screener.py`
Classifies every pixel as water, land, cloud or **sea ice** and decides whether the scene is worth
searching. Water is NDWI plus a dark-NIR test; cloud is bright and spectrally flat; land is any large
non-water region, buffered. Ice is the interesting one: it is as bright as cloud, but ice absorbs toward
865 nm while cloud droplets scatter neutrally, so ice keeps a positive NDWI and cloud sits at zero. On
four real Arctic scenes that boundary lands at NDWI +0.05 every time. Ice stays *inside* the searchable
sea -- a vessel beset in ice is the target, not an exclusion. Cloudy, empty or corrupt scenes stop here
and cost nothing further.

### 2. Vessel detector -- `applet/pipelines/vessel_detector.py`
Physics first. Water reflects about 1 % in NIR; steel and foam 10-40 %.

* **Ring-CFAR on the NIR band.** Each pixel is compared with a ring of background around it (guard
  120 m, background 400 m); it is a candidate if it exceeds the local mean by 5 robust sigmas *and* by
  an absolute 0.02 reflectance. The threshold therefore rises by itself in rough water.
* **Blobs and gates.** Bright pixels are linked into objects and gated by area, by distance from land
  (200 m shore keep-out), and -- when candidates are dense -- by being an outlier of the clutter
  population rather than a member of it. Band-to-band parallax marks aircraft.
* **Wake ray transform.** At 4.75 m the Kelvin "V" rarely resolves, but the turbulent centreline always
  does. From each hull we cast 360 rays and keep the one with contiguous excess contrast: that is the
  wake, and the opposite bearing is the heading (median error 4-5 degrees against real AIS). No wake
  means the heading is taken from hull shape and flagged as ambiguous by 180 degrees.
* **Speed, or nothing.** Transverse wake waves obey lambda = 2 pi V^2 / g. A speed is reported only when
  a clean spectral line of at least 8 cycles exists astern and is absent dead ahead; otherwise the
  field is `null`. It is advisory and cannot raise an anomaly on its own.
* **Ice regime and the lead transform.** Where a candidate's surroundings are ice, a bright blob is no
  longer evidence of a hull, so the detector asks for something ice cannot produce. A vessel under way
  in pack ice leaves an open-water channel *darker* than the floes -- the same ray transform with the
  sign reversed.
* Each candidate gets a **physics score** (contrast, shape, wake, Kelvin) and must clear 0.35.

### 3. Chip verifier -- `applet/pipelines/chip_verifier.py`
Whitecaps, cloud puffs and islets share a ship's NIR signature; shape and context separate them, which
is what a CNN is good at. A 47 k-parameter network (4 x conv-BN-ReLU-pool, global average pool, one
linear layer) scores each 64 x 64 four-band chip. It is trained from scratch on chips mined from *our
own detector's* candidates -- synthetic scenes, real SEN2MS chips, and the false alarms the team
adjudicated by hand -- then quantised to a **59 KB INT8 ONNX** file. A candidate is dropped below 0.80
unless its physics score is near-certain (0.95). It runs through ONNX Runtime: the CPU provider in the
container, TensorRT/CUDA picked up automatically on real hardware (written, never run). If the runtime
or the model is missing the stage degrades to physics-only and the pass still completes.

### 4. AIS correlator -- `applet/pipelines/ais_correlator.py`
Every broadcaster in the uplinked catalogue is dead-reckoned to the scene's shutter second, then matched
one-to-one against the detections: a tight gate (0.15 NM, growing with distance run since the fix) and a
wide gate (up to 1 NM) for broadcasters that are near but off their own track. A fix older than three
hours cannot identify anything -- on real data a day-old fix from the same berth once "confirmed" a ship
that was silent. What falls out is the product:

| Class | Meaning | Downlink priority |
| :-- | :-- | --: |
| `DARK_VESSEL` | seen, nobody broadcasting nearby | 0.70-1.00 |
| `AIS_KINEMATIC_MISMATCH` | a broadcaster is there but its course or track contradicts the wake | 0.60-0.85 |
| `CONFIRMED_KNOWN_VESSEL` | position and kinematics agree | 0.10 |
| `KNOWN_STRUCTURE` | matches an uplinked platform / island / buoy | 0.05 |
| `ICEBERG` *(opt-in)* | probable ice, see below | 0.02 |
| `AIS_NOT_OBSERVED` | AIS claims a resolvable ship in clear open water and nothing is there | reported |

### 5. Downlink packager -- `applet/packaging/downlink.py`
Writes `tactical_intelligence.geojson` (one feature per target with class, priority, heading, speed,
size, the three confidences, the matched MMSI and a plain-language reason), `scene_report.json`,
`manifest.json`, and JPEG chips for anomalies only, highest priority first, until the byte budget is
spent. The tarball is **byte-reproducible**: the same bundle gives the same SHA-256 on x86-64 Windows,
ARM64 macOS and the linux/arm64 container. `edge_telemetry.json` (timings, RAM, cores, modelled
temperature) is written *beside* it so it can never disturb that hash.

## Things that are off unless you ask

Each of these changes what a pass outputs, so each is opt-in and a default run stays a pure function of
its input bundle.

* **Ship or ice (`--arctic`)** -- [`arctic_classifier.py`](../applet/pipelines/arctic_classifier.py).
  Calved glacier ice floats in open water, so no background test sees it. In a scene the screener found
  ice in, a contact crowded by other bright objects with no wake, lead or AIS is called `ICEBERG` and
  sent to the back of the queue; everything else is `UNCERTAIN` and stays a full alert; only a
  transponder may say `SHIP`. Nothing is dropped.
* **Thermal governor (`thermal.governor_enabled`)** -- [`core/thermal.py`](../applet/core/thermal.py),
  [`core/governor.py`](../applet/core/governor.py). A two-node orbital thermal model predicts junction
  temperature from measured CPU load and looks ahead across the orbit; a four-rung ladder (FULL /
  REDUCED / SURVEY / BEACON) sheds *evidence* -- JPEG chips first, then the CNN -- before the hardware
  would shed throughput. Invariant: degrade the evidence, never the alert. Temperatures are modelled and
  labelled as such everywhere they appear.
* **Persistent-clutter memory (`ais_correlation.unknown_memory_enabled`)** --
  [`core/unknown_memory.py`](../applet/core/unknown_memory.py). Remembers no-AIS contacts that recur at
  the same place on three distinct dates (a rock, a buoy) and stops raising them. It is state that
  outlives a pass, which is why it is off by default, and what it hides is counted in the funnel.
* **Ice-retrained verifier** -- an alternative 59 KB model selected by uplinking `verifier.model_path`.

## In and out

**Input bundle** (a directory; nothing else is read, and nothing is fetched):

```
manifest.json          gsd_meters, reflectance_scale, bands, shutter_time, scenes[{id, file, georef}]
ais_catalog.json       vessels[{mmsi, name, timestamp, latitude, longitude, sog_knots, cog_deg}]
known_structures.json  optional
*.tif | *.npy | *.png  4-band rasters (R, G, B, NIR)
```

**Output**: `downlink_<pass>.tar.gz` (the product), `edge_telemetry.json`, and `queues/` of context
crops for the downlink scheduler. Everything that tunes behaviour is one YAML file
([`config.example.yaml`](../config.example.yaml), 5.8 KB); a model swap is 59 KB.

## How we know any of it works

We only quote numbers a command in this repo reproduces, and every figure in the spec sheet is tagged
measured, modelled or unvalidated.

* **Independent ground truth.** Whole Sentinel-2 scenes georeferenced from their own metadata, scored
  against the *same day's* public AIS (NOAA for 16 US scenes and the Bay of Fundy, Norway's Kystverket
  for Svalbard). Nothing about the imagery knows where the ships are.
* **Every contact adjudicated by eye.** AIS can prove a contact is a ship but never that one isn't, so
  571 US contacts and 51 Svalbard ones were labelled by the team. Those verdicts are in
  [`data/labels/`](../data/labels/) and are carried onto any new run *by position*, because detection
  ids renumber whenever the contact set changes.
* **Held out means held out.** Four US scenes never entered verifier training; synthetic seed 4242 was
  never used in development; the ice features were fitted on Alaska + US and tested frozen on Svalbard.
* **Uncorrected imagery.** The same 15 scenes were re-run on L1C (top-of-atmosphere, what a sensor
  actually sees) to check the L2A results were not an artefact of ground processing.
* **Negative results are kept.** Shadow height, object spectra, chip-AUC-as-a-proxy, RF geolocation
  gating, a wake-means-ship rule -- each was measured, failed, and is written down with its numbers.

```bash
python scripts/setup_data.py --synthetic      # offline: sample, tuning and held-out bundles
python -m pytest -q                           # the test suite
python -m applet run -i data/sample_bundle -o data/outputs
python -m ground.server                       # console at http://127.0.0.1:8050
python scripts/setup_data.py --all            # real datasets (internet, ~1.5 GB, no accounts)
python scripts/scorecard.py -i data/real/s2_us_bundle \
    --labels data/labels/us_labels.json --reference data/labels/us_reference_contacts.json
```

## Where things live

```
applet/            flight code
  runner.py          run_pass(): the single entry point
  core/              validator, telemetry, thermal model, governor, clutter memory, downlink crypto
  pipelines/         screener, detector, verifier, AIS correlator, ship-or-ice, stage registry
  packaging/         byte-reproducible downlink tarball
  models/            59 KB INT8 verifier, FP32 fallback, model card
ground/            FastAPI server, console, explainer pages, transfer viewer, fleet alerts
src/pyFlows/       rate-limited, encrypted downlink scheduler draining the pass's priority queues
simulation/        seeded scene renderer used for tuning and held-out synthetic tests
training/          chip mining, verifier training, ONNX export, INT8 quantisation, evaluation
scripts/           setup_data (start here), fetchers, scorecard, benchmark, soak, and the studies
                   behind each claim (iceberg, stationary, rf_cue, cue_geometry, orbit_pass_sim)
data/labels/       the team's hand verdicts and the reference contacts that carry them
review/            the contact sheets those verdicts were made on
tests/             corrupt input, physics vs rendered truth, sea ice, governor control law,
                   byte-identical output, and a test that the container image ships what it imports
docker/            Dockerfile.arm64 + pinned constraints
docs/              this file, SETUP, RUBRIC_SPEC_SHEET, PITCH_AND_DEMO, GALAXIA_ALIGNMENT, LABELLING
```

## What it does not do

Stated here so nobody finds it the hard way; the full list is in the spec sheet.

* Never validated on a Jetson: no TensorRT, DLA, power or thermal measurement.
* Validated at 10 m, not the 4.75 m target.
* Ships alongside a quay or within 200 m of land are dropped by design.
* A stationary vessel has no wake, so 40 % of its physics score is unavailable and it has no margin for
  any other error (LE BOREAL, 142 m, missed by 0.035).
* Dense ice defeats a VNIR detector. A person could not call 17 of 49 Svalbard contacts either; that is
  the sensor's information limit and the argument for a co-located non-optical sensor.

## Team

Jackson Chambers -- detection cascade, evaluation, thermal governor, ground console.
Megan Neville -- explainer pages and pipeline animation, persistent-clutter memory, the ship-or-ice design.
Ryan (ryan6625) -- downlink workflow, transfer viewer, fleet alerts, downlink encryption.
