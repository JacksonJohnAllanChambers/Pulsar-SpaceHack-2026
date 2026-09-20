# Tactical Edge Sentinel — rubric spec sheet

One page per judge. Every row is a claim, its evidence, and the command that reproduces it.

**The rule this project runs on:** every figure below is tagged **[M]** measured, **[Mo]** modelled, or
**[U]** unvalidated. Nothing is stated without one. If a number here is not reproducible by the command
beside it, that is a defect and we want to know.

**Scale, stated once so no row has to repeat it:** Arctic recall rests on n=5 visible broadcasters and
Bay of Fundy on n=2. They are directional, not statistical. The 16-scene US benchmark (571 contacts
adjudicated by hand) is the only result here with real weight behind it.

---

## Theme & Edge AI Relevance — 25 %

*"Does it meaningfully leverage NVIDIA Jetson capabilities, edge computing, embedded AI, CV, sensor
processing, local inference, optimization? Does RAM usage fall within actual allowed parameters for the
device? CPU? Temp? Does it take a really long time to run?"*

| Their question | Our answer | Tag | Reproduce |
| :-- | :-- | :-- | :-- |
| RAM within device limits? | **2.09 GB peak** on a full 4096² swath = **14.6 %** of the 14 GB cap | [M] | `scripts/benchmark.py` |
| CPU? | 2.85 of 6 cores busy average; headroom unspent | [M] | `edge_telemetry.json` per pass |
| **Temp?** | Two-node orbital thermal model predicts junction temperature and **drives a real decision** — sustainable SoC power **34.6 W eclipse vs 18.9 W sunlit** | [Mo] | `scripts/orbit_pass_sim.py --skip-cost` |
| Long to run? | **1.61 s** per full swath, native ARM64 (M4, 6 threads) | [M] | `scripts/run_on_macmini.sh --native` |
| In the judging container? | 16 real scenes in **5.33 s**, **2,603 MB = 18.2 %** of cap, linux/arm64, no GPU, no network | [M] | `scripts/run_emulated.sh` |
| Local inference | 59 KB INT8 ONNX, **0.09 ms/chip**, ORT CPU provider; TensorRT/CUDA selected automatically if present | [M] / [U] on accelerators | `applet/models/model_card.json` |
| Jetson hardware validation | **None. We never had a Jetson.** Every bundle carries `"validated_on_hardware": false` | [U] | `grep validated_on_hardware data/outputs/*/edge_telemetry.json` |

**The temperature answer in one sentence:** your setup guide says the container cannot show power or
thermal throttling, and your rubric asks about temperature — so we modelled it, labelled it `MODELLED`
on the same screen as the number, and used it to drive cascade depth rather than leaving it decorative.

---

## Technical Implementation — 30 %

| Claim | Evidence | Tag | Reproduce |
| :-- | :-- | :-- | :-- |
| Five-stage cascade behind one entry point | `run_pass()` — used identically by CLI, console, benchmark and every test | [M] | `applet/runner.py` |
| Test coverage | **150 tests**: corrupt input, missing bands, physics vs rendered truth, sea-ice regime, thermal control-law regressions, byte-identical output | [M] | `python -m pytest -q` |
| **Determinism across architectures** | Same bundle → **same SHA-256 tarball** on x86-64 Windows, ARM64 macOS and linux/arm64 container, across different numpy / OpenCV / ONNX builds | [M] | `tests/test_pipeline.py::test_output_is_byte_identical_across_runs` |
| Cannot crash the spacecraft | Every stage degrades rather than raises — missing band, corrupt file, absent ONNX runtime, failed model load all still produce a bundle | [M] | `tests/test_validator.py` |
| Runs in the judges' container | Dockerfile COPY set staged and executed by a test, so flight code cannot import something the image does not ship | [M] | `tests/test_flight_image.py` |
| Real-data benchmark | 16 US Sentinel-2 scenes vs same-day NOAA AIS, **571 contacts hand-adjudicated** | [M] | `scripts/scorecard.py -i data/real/s2_us_bundle` |

---

## Innovation & Problem Solving — 20 %

| Contribution | Why it is not the obvious approach | Tag |
| :-- | :-- | :-- |
| **Wake ray transform** | At 4.75 m the Kelvin cusp arms rarely resolve, so we do not chase the textbook V. 360 rays from each hull find the turbulent centreline, which always resolves. **4° median heading error vs real AIS.** | [M] |
| **Sea ice as a fourth surface class** | HyperScape100 stops at 860 nm → no SWIR → no NDSI. Ice separated from cloud *inside VNIR alone*: ice absorbs toward 865 nm, cloud scatters neutrally, so the boundary lands at NDWI +0.05 on all four Arctic scenes. Cuts Arctic false alarms **23 %** at **zero** cost to temperate recall. | [M] |
| **Lead transform** — same ray transform, sign reversed | A vessel under way in pack ice leaves an open-water channel *darker* than the floes. Independently confirmed on Landsat thermal: fires on **21.5–25.8 %** of narrow-lead candidates vs **1.0–2.0 %** control. | [M] |
| **Thermal-aware cascade depth** | Treats the power envelope as an *input*, not a limit discovered by throttling. Looks ahead, because the bus time constant (~70 min) is the same order as the orbit (~95 min). Invariant: **degrade the evidence, never the alert.** | [Mo] for degrees, [M] for the rung costs |
| **Cue-latency geometry** | A 15-knot vessel runs **13.9 km in 30 min** against a 19.4 km swath. At the only independently-assessed RF accuracy (ESA EDAP+ on Unseenlabs), the 95 % gate is **20.8 km — wider than the swath**. So the argument for onboard RF is not better geolocation; **co-location removes the need for geolocation.** | [M] geometry, [Mo] error model |
| Verifier trained on its own candidates | Chips mined from the detector's own output, not a generic ship dataset. FP32→INT8: 3.1× smaller, same AUC. | [M] |

---

## Impact & Practical Value — 15 %

| | |
| :-- | :-- |
| **Customer** | Galaxia Mission Systems (Halifax). Their MÖBIUS-1 launch release names *"maritime security, tracking dark vessels and combating illegal fishing operations."* They acquired a Simera HyperScape100 on **3 Sep 2026** — sixteen days before this event. |
| **Mission fit** | $2.5M DRDC contract for CTOS, whose stated purpose is validating **onboard edge computing**. Our applet is a Blade workload for their Onyx Edge architecture. |
| **The decision it supports** | 182 MB of raw swath → **7.3 KB** naming which ships are not broadcasting and which are lying about their course, inside the pass rather than hours later. |
| **Scale of the problem** | 243,000 km of Canadian coastline; extended Arctic navigable season. |
| **Where it is deployable today** | Temperate and light-ice water. **Not** heavy pack ice (see below), **not** polar night. |

Full sourcing with dates: `docs/GALAXIA_ALIGNMENT.md`.

---

## Presentation & Demonstration — 10 %

| | |
| :-- | :-- |
| Live demo | `python -m ground.server` — the unmodified onboard code path, visualised. Six-click path rehearsed in `docs/PITCH_AND_DEMO_TEMPLATE.md` §3. No CDN, no internet. |
| Evidence you can open | Self-contained contact sheets: every contact as true-colour + NIR at two zooms, with the verdict a human gave it. |
| Honesty on screen | The governor panel is badged **MODELLED** on the same screen as the temperature. |
| Limitations | `docs/PITCH_AND_DEMO_TEMPLATE.md` §6, volunteered before being asked. |

---

## The measured results, in full

### Temperate — the benchmark with weight behind it

| | |
| :-- | --: |
| Recall vs AIS the sensor could see | **0.912** |
| Precision, 571 contacts hand-adjudicated | **0.697** |
| False alarms | 105 (18.9 per 1000 km²) |
| Heading error vs reported COG | 4° median |

### Arctic — Svalbard vs Kystverket AIS

| | |
| :-- | --: |
| Recall vs visible AIS | **0.60** (n=5) |
| Recall vs all hand-confirmed vessels | 0.75 (6 of 8) |
| Precision (32 of 49 adjudicated) | **0.188** |
| Position error | median **20.4 m** |

**Precision tracks ice fraction — the finding we lead with:**

| Scene | Sea ice | Contacts | Precision |
| :-- | --: | --: | --: |
| Isfjorden | 9.7 % | 6 | **0.667** |
| Kongsfjorden | 25.5 % | 43 | **0.077** |

2.6× the ice → 7× the contacts → precision falls 9×.

### Atlantic Canada — Bay of Fundy vs NOAA AIS

| | |
| :-- | --: |
| Recall vs visible AIS | **1.0** (n=2) |
| Position error | median **18.5 m** |
| Cloud | 0.0 % |
| Heading error | 35.1° median, 50 % flipped — small vessels, weak wakes |

---

## What we will not claim

* **No Jetson.** No TensorRT, DLA, power or thermal validation. The governor's temperatures are a model
  calibrated to NVIDIA's published envelopes; the spacecraft bus parameters are ours, not a flown bus's.
* **Optical does not solve heavy sea ice.** Precision 0.077 at 25 % ice. We measured where it stops.
* **17 of 49 Arctic contacts could not be adjudicated by eye at all.** A trained human on 10 m pixels
  cannot separate a small stationary hull from a floe. That is an information limit of a VNIR payload,
  not a defect in the code — and it is the strongest argument in this repo for a non-optical sensor on
  the same bus. If all 17 were clutter, Arctic precision would be 0.122.
* **Stationary vessels have no error budget.** LE BOREAL (142 m, 0.1 kn) scored physics **0.315 against
  a 0.35 gate** — missed by 0.035. CFAR fired at z=16.1; the hull was under-segmented to 37 m; and with
  no wake, 40 % of the score is structurally unavailable. A *moving* ship with identical segmentation
  would have cleared the gate easily. Diagnosis: `scripts/diagnose_miss.py --mmsi 578000500`.
* **The governor rung table is measured on synthetic tuning scenes** (`data/eval_bundle`, seed 777,
  4.75 m), not on real imagery.
* **No speed number for the governor.** Our hardware could not measure it — run-to-run drift exceeded
  the effect and two careful attempts disagreed in sign. We withdrew the figure rather than defend it.
* Real validation is at 10 m Sentinel-2, not the 4.75 m target GSD.
* Ships moored alongside, or within 200 m of land, are not reported — by design.
* Hull length reads ~20 % short temperate, 59.7 % short Arctic (n=2).
* The Kelvin-arm measurement never fires today.

---

## Reproduce everything

```bash
python scripts/setup_data.py --synthetic     # bundles, no network
python -m pytest -q                          # 150 tests
python scripts/scorecard.py -i data/real/s2_us_bundle      # temperate benchmark
python scripts/scorecard.py -i data/real/svalbard_poc --labels data/outputs/svalbard_review/labels.json
python scripts/scorecard.py -i data/real/fundy_bundle      # Atlantic Canada
python scripts/orbit_pass_sim.py --input data/eval_bundle --repeats 5   # governor, synthetic
python scripts/cue_geometry.py                             # cue-latency argument
python -m ground.server                                    # live console
```
