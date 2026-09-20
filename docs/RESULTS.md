# Results — every measured number, and the command that produced it

The full data appendix. [`RUBRIC_SPEC_SHEET.md`](RUBRIC_SPEC_SHEET.md) is the one-page version for
judges; [`OVERVIEW.md`](OVERVIEW.md) explains how the thing works. This file is the evidence.

**Every figure is tagged.** **[M]** measured on hardware we had · **[Mo]** modelled, never validated ·
**[U]** unvalidated. Nothing appears untagged. If a number here does not reproduce from the command
beside it, that is a defect and we want to know.

**Provenance of this page.** Every table below was re-run on **2026-09-20** on the x86-64 Windows dev
box (22 threads, Python 3.11.9, numpy 2.4.2, OpenCV 5.0.0, ONNX Runtime 1.20.1) against commit
`83066b6`, except where a row says otherwise. Raw JSON is written under `data/outputs/` by each command.

**Read the sample sizes before the accuracy.** The 16-scene US benchmark rests on 347 hand-adjudicated
contacts and is the only result here with statistical weight. Arctic recall is **n=5 visible
broadcasters**; Bay of Fundy is **n=2**. Those are directional, and we say the n out loud every time.

---

## 1. Temperate — 16 US scenes vs same-day NOAA AIS

The primary benchmark. Sixteen Sentinel-2 L2A scenes across every US coast, each paired with the same
day's public NOAA Marine Cadastre AIS. Each scene is georeferenced from its own COG transform, so
nothing in the imagery knows where the ships are. Then every contact was adjudicated by eye, because
AIS can prove a contact *is* a ship but never that one is not.

```bash
python scripts/scorecard.py -i data/real/s2_us_bundle \
    --labels data/labels/us_labels.json --reference data/labels/us_reference_contacts.json
```

| | | Tag |
| :-- | --: | :-- |
| Scenes usable / total | 16 / 16 | [M] |
| Searched water | 5,672.9 km² | [M] |
| AIS broadcasters in footprint | 1,049 | [M] |
| …**visible to the sensor** | 136 | [M] |
| …matched by the detector | **124** | [M] |
| **Recall vs AIS the sensor could see** | **0.912** | [M] |
| Contacts raised | 375 | [M] |
| Contacts adjudicated by hand | 347 (28 unlabelled, excluded) | [M] |
| True vessels / false alarms | 242 / **105** | [M] |
| **Precision** | **0.697** | [M] |
| False-alarm density | 18.5 per 1000 km² | [M] |
| Recall vs all known vessels (AIS + human-confirmed) | 0.953 | [M] |
| Position error vs dead-reckoned AIS | median **46.3 m**, p90 98.2 m, max 563 m | [M] |
| …within 200 m | 96.8 % | [M] |
| Heading error vs reported COG (n=66 under way) | median **5.4°**, p90 89.6° | [M] |
| …within 20° / flipped 180° | 77.3 % / 22.7 % | [M] |
| Hull length error (n=85, ≥20 m ships) | median 15.1 %, p90 121.9 % | [M] |
| Wake-derived speed | **n=0 — it never fired on this bundle** | [M] |
| Runtime / peak RAM (x86-64, 16 scenes) | 14.20 s / 2,640 MB | [M] |

**Why 887 broadcasters are not counted as misses.** They are alongside quays inside harbours, which the
200 m shoreline keep-out drops on purpose. The full breakdown of the 925 unmatched:
`IN_PORT_OR_SHORE_KEEPOUT` 887 · `CLEAR_WATER_NO_TARGET` 19 · `BELOW_SENSOR_RESOLUTION` 15 ·
`UNDER_CLOUD` 4. Of the 19 in clear water, reviewers confirmed **7 had no vessel visible at all** —
stale or ghost fixes, excluded rather than charged to the detector, which is why recall is 124/136
and not 124/143.

### Per scene — where it is good and where it is not

| Scene | Water km² | Cloud % | Contacts | AIS matched | Dark | Precision |
| :-- | --: | --: | --: | --: | --: | --: |
| Puget Sound | 187.9 | 0.00 | 15 | 9 | 6 | **0.933** |
| Galveston | 376.9 | 0.00 | 38 | 31 | 7 | 0.892 |
| Long Beach | 344.9 | 0.17 | 39 | 17 | 22 | 0.833 |
| Norfolk | 341.9 | 0.00 | 19 | 8 | 11 | 0.789 |
| New York | 373.5 | 0.00 | 29 | 9 | 20 | 0.778 |
| San Diego | 355.6 | 2.07 | 29 | 8 | 21 | 0.724 |
| Miami | 377.9 | 0.43 | 60 | 7 | 53 | 0.719 |
| Charleston | 412.0 | 0.00 | 3 | 0 | 3 | 0.667 |
| Corpus Christi | 371.0 | 0.00 | 17 | 7 | 10 | 0.667 |
| San Francisco | 380.7 | 0.00 | 10 | 2 | 8 | 0.667 |
| Honolulu | 228.0 | 3.61 | 15 | 5 | 10 | 0.583 |
| Mississippi | 378.0 | 0.00 | 30 | 9 | 21 | 0.520 |
| Delaware | 419.4 | 0.00 | 6 | 2 | 4 | 0.500 |
| Savannah | 374.6 | 0.39 | 6 | 2 | 4 | 0.500 |
| **Tampa** | 388.3 | 0.72 | 30 | 3 | 27 | **0.500** |
| **Boston** | 362.2 | 0.00 | 29 | 5 | 24 | **0.423** |

Shallow, turbid and structurally busy water is the worst case — Tampa and Boston — and clean deep
water with heavy traffic is the best. Miami raises 60 contacts, the most of any scene, and still holds
0.719.

---

## 2. The same scenes without atmospheric correction (L1C)

The Track Guide warns against validating on ground-processed data as if it were raw onboard input.
Every scene above is **L2A** — atmospherically corrected on the ground with compute a satellite does
not have. So we fetched the **L1C** (top-of-atmosphere) product of the *same acquisitions*, cut from
the identical pixel window, and ran the same model with the same config. Labels carry across by
position. 15 of 16 scenes; Boston is a Sentinel-2C commissioning pass with no public L1C.

```bash
python scripts/scorecard.py -i data/real/s2_us_l1c_bundle \
    --labels data/labels/us_labels.json --reference data/labels/us_reference_contacts.json
```

| 15 scenes | L2A (corrected) | **L1C (what the sensor sees)** | Tag |
| :-- | --: | --: | :-- |
| AIS visible / matched | 136 / 124 | 132 / **119** | [M] |
| Recall vs visible AIS | 0.912 | **0.902** | [M] |
| Contacts | 375 | 373 | [M] |
| Adjudicated / unlabelled | 347 / 28 | 313 / **60** | [M] |
| Precision on adjudicated | 0.697 | **0.732** | [M] |
| False alarms | 105 | 84 | [M] |
| Position error, median | 46.3 m | **46.3 m** | [M] |
| Heading error, median | 5.4° | **5.4°** | [M] |
| Hull length error, median | 15.1 % | 14.1 % | [M] |

**Vessel detection is essentially unaffected**, which is the point: the detector is a local-contrast
test in NIR, the band the atmosphere touches least. Haze costs *clutter in shallow turbid water*, not
ships. The honest caveat: 60 L1C contacts have no labelled counterpart, so true L1C precision lies
between **0.62** (if every unlabelled one is a false alarm) and **0.73**.

---

## 3. Arctic — Svalbard vs Kystverket AIS

NOAA stops at 50.195 °N, so there is no open US Arctic AIS. Norway's Kystverket publishes open AIS
covering the Svalbard protection zone (Kystdatahuset REST, no credentials, NLOD licence). Two
Sentinel-2 scenes, 2024-06-22, 591.6 km² of searched water, 9–42 % sea ice.

```bash
python scripts/scorecard.py -i data/real/svalbard_poc \
    --labels data/labels/svalbard_labels.json --reference data/labels/svalbard_reference_contacts.json
```

| | | Tag |
| :-- | --: | :-- |
| AIS in footprint / visible / matched | 8 / 5 / **3** | [M] |
| **Recall vs visible AIS** | **0.60** — *n=5* | [M] |
| Recall vs all hand-confirmed vessels | 0.75 (6 of 8) | [M] |
| Contacts / adjudicated / uncallable | 49 / 32 / **17** | [M] |
| **Precision** | **0.188** | [M] |
| False alarms | 26 (43.9 per 1000 km²) | [M] |
| Position error | median **20.4 m**, p90 61.1 m | [M] |
| Heading error | median **2.2°**, 100 % within 20° | [M] |
| Hull length error | median **59.7 %** short — *n=2* | [M] |

Found CRYSTAL ENDEAVOR (164.5 m), SILVER WIND (155.8 m) and TEISTEN (17 kn). Both AIS misses were
adjudicated and both are **genuine**: LE BOREAL, 142 m sitting at 0.1 kn, was there and we did not
find it (diagnosed in §8).

### Precision tracks ice fraction — the finding we lead with

Same code, same day, two fjords 100 km apart:

| Scene | Sea ice | Contacts | Precision | Tag |
| :-- | --: | --: | --: | :-- |
| Isfjorden | 9.7 % | 6 | **0.667** | [M] |
| Kongsfjorden | 25.5 % | 43 | **0.077** | [M] |

2.6× the ice → 7× the contacts → precision falls 9×. We are not claiming optical solves sea ice. We
measured where it stops.

**17 of 49 contacts could not be adjudicated at all.** A trained eye on 10 m pixels cannot separate a
small stationary hull from a floe — no wake, and at 2–3 px no shape either. That is an information
limit of a VNIR payload, not a defect in the code. If all 17 were clutter, precision would be 0.122.

### Ship-or-ice, opt-in (`--arctic`)

```bash
python scripts/scorecard.py -i data/real/svalbard_poc --arctic \
    --labels data/labels/svalbard_labels.json --reference data/labels/svalbard_reference_contacts.json
```

| Svalbard | off | `--arctic` | Tag |
| :-- | --: | --: | :-- |
| Raised as dark vessels | 46 | **16** | [M] |
| Classified `ICEBERG` (demoted, still downlinked) | 0 | 30 | [M] |
| Demoted: clutter / uncallable / **real vessels** | — | 21 / 9 / **0** | [M] |
| Contacts without AIS per 1000 km² | 77.8 | **27.0** | [M] |
| **Precision of the alerts that remain** | 0.188 | **0.545** | [M] |
| Recall vs visible AIS, position, heading | 0.60 / 20.4 m / 2.2° | **unchanged** | [M] |

Read it with its limits: the neighbour threshold was read off these same two scenes and every Svalbard
vessel is large, so **0.545 is optimistic**. On the 16 US scenes the feature is contact-for-contact
identical with it on, and adds no field to the bundle. Evidence, and the three alternatives that
failed, in §7.

### Alaska probe — precision-only, no AIS exists

```bash
python scripts/scorecard.py -i data/real/arctic_probe
```

4 real scenes (Utqiagvik, Prudhoe, Kotzebue, Point Hope), 1,458.2 km² searched, **115 contacts, zero
vessels** — every one inspected by hand. This is the sea-ice module's benchmark, not a recall test.

| Configuration | Arctic contacts | per 1000 km² | US recall | US precision | Tag |
| :-- | --: | --: | --: | --: | :-- |
| Flight config before the ice module | 149 | 121.5 | 0.912 | 0.693 | [M] |
| **+ ice mask and ice regime (shipped)** | **115** | **78.9** | **0.912** | **0.697** | [M] |
| + ice-retrained verifier (uplink option) | 71 | 48.7 | 0.921 | 0.655 | [M] |

The physics module is **free**: −23 % Arctic contacts at zero cost to temperate water. The retrained
verifier buys another −38 % but costs **+26 US false alarms for +1 AIS-confirmed ship**, so it ships
beside the flight model and is selected by uplinking `verifier.model_path`, not by default.

---

## 4. Atlantic Canada — Bay of Fundy vs NOAA AIS

USCG receivers in Maine hear Canadian MMSI 316* across the Bay of Fundy, so a Canadian scene can be
AIS-scored after all. One scene, 2024-08-08, 316.5 km², 0.0 % cloud.

```bash
python scripts/scorecard.py -i data/real/fundy_bundle
```

| | | Tag |
| :-- | --: | :-- |
| AIS in footprint / visible / matched | 4 / 2 / **2** | [M] |
| Recall vs visible AIS | **1.0** — *n=2* | [M] |
| Contacts | 16 (14 without AIS, unadjudicated) | [M] |
| Position error | median **18.5 m**, 100 % within 200 m | [M] |
| **Heading error** | median **35.1°**, **50 % flipped 180°** | [M] |

Heading is **poor** here and we are not hiding it: small vessels with weak wakes give the ray transform
little to lock onto, and with no wake the hull-shape heading is ambiguous by 180°. Coverage is
propagation-dependent — probe each date with `scripts/probe_canada_ais.py` before trusting it.

---

## 5. The verifier CNN

`applet/models/model_card.json`. Trained from scratch on chips mined from **our own detector's**
candidates — no pretrained weights anywhere in this repo.

| | | Tag |
| :-- | --: | :-- |
| Architecture / parameters | 4×[conv3×3-BN-ReLU-pool] → GAP → FC · **47,041** | [M] |
| Training chips | 11,607 synthetic + 6,552 real (3,608 SEN2MS; 368 hand-adjudicated ×8) | [M] |
| Validation chips | 3,644 | [M] |
| **FP32** → size / accuracy / AUC / latency | 185.4 KB · 0.799 · 0.978 · 0.097 ms/chip | [M] |
| **INT8** → size / accuracy / AUC / latency | **59.0 KB** · 0.838 · **0.979** · **0.094 ms/chip** | [M] |
| Quantisation cost | **3.1× smaller, AUC unchanged** | [M] |

### It generalised; it did not memorise

Four scenes were held out of training entirely (Tampa, Long Beach, Puget Sound, New York). Miami — also
a shallow turbid bay — stayed in, so Tampa measures transfer to an *unseen* turbid bay.

| | Before retraining | After | Tag |
| :-- | --: | --: | :-- |
| Held-out chip AUC | 0.744 | **0.920** | [M] |
| Held-out scene precision | 0.503 | **0.731** | [M] |
| Held-out false alarms surviving threshold | 85 | **29** | [M] |
| True vessels lost | — | 7 | [M] |
| Median clutter score | 0.985 | **0.606** | [M] |

The mechanism is visible in that last row: the synthetic-only verifier scored genuine vessels 0.999
and clutter 0.985, so at its 0.80 threshold **it rejected nothing**. Training on real adjudicated
clutter is what made the stage do work.

---

## 6. Edge budget, determinism and endurance

### One full 19.4 km swath (4096², 23.1 Mpx, 182 MB)

```bash
python scripts/benchmark.py --full-swath -n 3
```

| | x86-64 Windows dev box | ARM64 (Apple M4, 6 threads) | Tag |
| :-- | --: | --: | :-- |
| Warm pass | 7.30 s ± 0.07 | **1.61 s** ± 0.02 | [M] |
| Peak RAM | 1,643 MB (**11.5 %** of 14 GB) | 2.09 GB (14.6 %) | [M] |
| Cores busy | **1.65 of 6** | 2.24 of 6 | [M] |
| Raw → downlink | 182 MB → **9.39 KB** (19,840×) | 182 MB → 9.4 KB | [M] |

Per stage, x86-64: detector 3.25 s · screener 2.39 s · ingest 1.34 s · **CNN 0.158 s** · AIS 0.041 s ·
queue router 0.075 s · packaging 0.027 s. The neural network is **2 %** of the pass — that is the
physics-first cascade paying off.

**The M4 column is an upper bound, not a Jetson prediction.** An M4 is far quicker than an Orin NX's
Cortex-A78AE cores. What it establishes is that nothing depends on x86.

### In the judges' own container

```bash
./scripts/run_emulated.sh        # --memory=14g --memory-swap=14g --cpus=6 --network none
```

| linux/arm64 under QEMU, 8 sample scenes | | Tag |
| :-- | --: | :-- |
| Scenes ingested / rejected | 7 valid, 1 rejected at ingest (truncated file) | [M] |
| Peak RAM | **587 MB — 4.1 % of the 14 GB cap** | [M] |
| Raw → downlink | 54.0 MB → **6.68 KB** (8,273×) | [M] |
| Wall clock | 57–76 s | [M], QEMU — meaningless as a speed figure |
| Cores busy | reported, then **disclaimed** — see below | [M] |

**The CPU figure inside QEMU is not ours.** User-mode QEMU leaves `utime`/`stime` at 0 in
`/proc/self/stat`, so `psutil` reads every stage as free — which is why earlier container runs printed
`0.0 of 6 cores busy`. `time.process_time()` keeps counting, but what it counts includes QEMU's own
translation threads: a **single-threaded** 2 s spin bills 2.52 CPU-seconds, and a full pass pins at
exactly the `--cpus=6` ceiling. The two clocks disagreeing *is* the emulation signature, so telemetry
now sets `cpu_time_includes_emulator` and the report says to measure natively. **The applet's real
figure is the native one: 1.65 of 6 cores.**

### Byte-for-byte determinism across architectures

Given the **same input bundle**, the downlink tarball has the same SHA-256 on both:

| | SHA-256 of `downlink_PASS_*.tar.gz` | Tag |
| :-- | :-- | :-- |
| x86-64 Windows · Python 3.11.9 · numpy 2.4.2 · OpenCV 5.0.0 | `80cc882c490f76e2…` | [M] |
| linux/arm64 QEMU · Python 3.10 · numpy 2.2.6 · OpenCV 4.13.0 | `80cc882c490f76e2…` | [M] |

Same GeoJSON, same scene report, same 11 JPEG chips — different instruction sets, different operating
systems, different numpy / OpenCV / ONNX builds.

**One caveat that matters if you reproduce this.** The *scene renderer* is **not** cross-architecture
deterministic: `generate_synthetic_data.py` run inside the aarch64 image produces different pixels from
the same script on x86-64 (`1ce987cc…` vs `30c53b2b…` for `scene_01_clear.tif`), because the rendering
path depends on the OpenCV build. It is deterministic on a given machine. So **mount one bundle into
both** rather than letting each side render its own, or you will compare two different inputs and
conclude the applet is non-deterministic when it is not.

### Endurance — 25 consecutive passes in one process

```bash
python scripts/soak.py -i data/real/s2_us_bundle -n 25 --quiet
```

| | | Tag |
| :-- | --: | :-- |
| Wall clock, first → last quartile | 12.88 s → 11.98 s (**−7.0 %**, i.e. warming up, not drifting) | [M] |
| RSS | 478 → 488 MB (**+10 MB**), peak 509 MB | [M] |
| Contacts per pass | **375, every pass** | [M] |
| Distinct downlink tarballs | **1 — byte-identical across all 25** | [M] |

It exits non-zero on a leak, on >15 % slowdown, or if the output stops being reproducible, so it works
as a regression test rather than a demonstration. (The 120-pass M4 run reported +0.5 % drift and
+58 MB RSS.)

---

## 7. Thermal governor

Two switches, deliberately separate. `report_thermal` (**on**) models and *reports* a junction
temperature beside the measured RAM and CPU, and changes no decision. `governor_enabled` (**off**) lets
the model act on the cascade — off by default because a governed pass depends on host timings, so it
would stop being a pure function of its input bundle.

```bash
python scripts/orbit_pass_sim.py --input data/eval_bundle --repeats 5
```

### The modelled half — **[Mo]**, never validated on hardware

| | | Tag |
| :-- | --: | :-- |
| Radiator | 0.09 m², ε 0.85, α 0.20 — rejects **35.21 W at 27 °C** | [Mo] |
| Direct sun adds back | **24.5 W** on the same face | [Mo] |
| **Sustainable SoC power** | **34.63 W in eclipse** vs **18.89 W sunlit** | [Mo] |
| We act at | 95 °C (Orin's own TJ_max is 105 °C) | [Mo] |

Spacecraft bus parameters (radiator area, coating, parasitic load) are **ours, not a flown bus's**. The
Orin power envelopes are NVIDIA's published ones. The CPU utilisation driving it is genuinely measured.

### Governor behaviour across two orbits — **[Mo]**

| Orbit | Peak junction | Settles on | Rungs used | Transitions | Throttled |
| :-- | --: | :-- | :-- | --: | :-- |
| Mid-beta SSO (35 % eclipse) | 83.06 °C | **FULL** | FULL, REDUCED | 4 | no |
| Dawn-dusk SSO (**no eclipse**) | 70.42 °C | **REDUCED** | REDUCED | 0 | no |

A dawn-dusk orbit never gets thermal relief, so the governor settles permanently one rung shallower —
and it gets there *while the die is still cold*, because it looks ahead. The bus time constant (~70 min)
is the same order as the orbit (~95 min), so reacting to heat would always be too late.

### What each rung actually costs — **[M]**, on 40 synthetic tuning scenes (`data/eval_bundle`, seed 777)

| Rung | Power cap | Wall s | CPU s | Cores | Peak RSS | Candidates | Dark | AIS-confirmed | Chips | Downlink |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **FULL** | 25 W | 4.92 | 12.03 | 2.53 | 1,070 MB | 3,835 | 18 | **46 / 46** | 28 | 18.4 KB |
| **REDUCED** | 15 W | 4.90 | 10.56 | 2.16 | 1,048 MB | 3,544 | 18 | **46 / 46** | 0 | **7.5 KB** |
| **SURVEY** | 15 W | 5.55 | 8.50 | 1.54 | 1,050 MB | 2,552 | 18 | 41 / 46 | 0 | 6.8 KB |
| **BEACON** | 10 W | 5.33 | 8.02 | 1.53 | 1,037 MB | 1,498 | **121** | 44 / 46 | 0 | 10.2 KB |

The invariant is **degrade the evidence, never the alert**:

* **REDUCED is free.** Same physics, same CNN, same 18 dark vessels and all 46 AIS-confirmed ships —
  it drops only the JPEG evidence crops, and the downlink falls **59 %**. It is the first rung for
  exactly that reason.
* **SURVEY** drops the wake ray transform, the most expensive per-candidate stage. That costs heading,
  speed and the wake term of the physics score: **5 AIS-confirmed ships lost**.
* **BEACON** is physics-only with a hard candidate cap and no CNN. Positions still go down and every
  contact is UNVERIFIED. *The applet never stops reporting; it only stops explaining.* On this
  synthetic bundle false alarms rise **6.7×** and the downlink goes back **up** — 121 alerts cost more
  than 18 alerts plus their chips. **On real imagery it behaves differently: see the next table.**

### The same table on real imagery — **[M]**, 16 US Sentinel-2 scenes

The synthetic table above was the honest limit of what we had measured, and it is flagged as such
everywhere it appears. So we ran the identical sweep against the real benchmark:

```bash
python scripts/orbit_pass_sim.py --input data/real/s2_us_bundle --repeats 5
```

| Rung | Power cap | Wall s | CPU s | Cores | Peak RSS | Candidates | Verified | Dark | AIS-confirmed | Mismatches | Chips | Downlink |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **FULL** | 25 W | 11.30 | 28.12 | 2.51 | 2,971 MB | 807 | 375 | 251 | 109 | **15** | 259 | 87.3 KB |
| **REDUCED** | 15 W | 12.49 | 27.16 | 2.23 | 2,909 MB | 807 | **375** | **251** | **109** | **15** | 0 | **22.5 KB** |
| **SURVEY** | 15 W | 17.06 | 23.81 | 1.41 | 2,829 MB | 612 | 310 | 193 | 116 | 1 | 0 | 18.8 KB |
| **BEACON** | 10 W | 16.70 | 23.67 | 1.43 | 2,815 MB | 379 | 279 | 175 | 104 | 0 | 0 | 16.5 KB |

**REDUCED is free on real imagery too, and that is the claim that mattered.** Byte for byte the same
807 candidates, 375 verified contacts, 251 dark vessels, 109 AIS-confirmed ships and all 15 kinematic
mismatches — for a downlink of **22.5 KB against 87.3 KB, a 74 % cut**. The rung exists to drop JPEG
evidence crops, and on 16 real scenes that is exactly and only what it does.

**Two things the synthetic bundle got wrong, which is why this table needed running.**

* **AIS-confirmed appears to *rise* at SURVEY (109 → 116). It is a loss, not a gain.** Dropping the
  wake ray transform removes the evidence that raises `AIS_KINEMATIC_MISMATCH`, so mismatches collapse
  **15 → 1** and those contacts are re-labelled as ordinary confirmed traffic. Count the total instead:
  AIS matches fall **124 → 117 → 104** down the ladder. The spoofing detector — the thing that catches
  a ship lying about its course — is effectively dead below REDUCED. The synthetic bundle produced only
  10 mismatches and hid this.
* **BEACON does not multiply false alarms on real imagery.** On synthetic scenes it raised dark-vessel
  alerts 6.7× (18 → 121), and we published that. On real scenes dark alerts *fall* 251 → 175, because
  the per-scene candidate cap binds long before the clutter does: 807 candidates → 379. BEACON's real
  failure mode is **silently dropping contacts**, not flooding the operator — the opposite of what we
  said, and worse, because a flood is visible and a silent drop is not.

Wall-clock deepens down the ladder (11.3 s → 16.7 s) rather than shrinking, which is the same
measurement problem as below; the deterministic columns are the ones to read.

### No speed number appears here, on purpose — **[U]**

The power cap and worker count are exact, because they are configuration. What that does to wall-clock
we **could not measure on this hardware**. Re-running the whole script moved our FULL median across
6.51 / 8.33 / 9.73 s on identical data, and two careful attempts **disagreed in sign** (five sequential
repeats said REDUCED used 30 % less CPU than FULL; six interleaved rounds said 11 % more). We published
"9 % slower" from a single shot and withdrew it. Lesson, for anyone extending this: interleave the
arms, repeat the whole invocation, and compare between-invocation drift before quoting a percentage.

---

## 8. Ship-or-ice: the evidence, and three things that did not work

```bash
python scripts/iceberg_study.py
```

All on 539 real contacts (Svalbard 49 hand-labelled · Alaska 115 all-clutter · US 375).

| Candidate evidence | Result | Verdict | Tag |
| :-- | :-- | :-- | :-- |
| Centre-window brightness / flatness / texture | fired on **0 of 539** — a real contact is ~12 px, so a 32×32 mean is the water around it | rejected | [M] |
| Fill-invariant spectral slope | real physics with a control — NIR/visible **0.46** glacier ice, **0.77–1.0** hulls, **1.12** *temperate* clutter (opposite sign) — but as a frozen gate caught **1 of 104** Alaska ice contacts | reported, no vote | [M] |
| "A wake or lead means SHIP" | wrong on **21 of 21** Alaska pack-ice contacts and 3 of 5 Svalbard ones — leads and brash streaks are linear too | rejected | [M] |
| **Crowding inside an icy scene** | ≥3 bright neighbours in the chip, no wake/lead/AIS → see §3 | **shipped** | [M] |

Only an AIS match may assert `SHIP`. Image evidence can demote; it cannot confirm.

**Why the existing ice module cannot see this case.** The per-candidate `ice_regime` flag fires on
**0 of 49** Svalbard contacts and 14 of 115 Alaska ones. Kongsfjorden is not pack ice — it is calved
glacier ice floating in open dark water, so the annulus around each piece genuinely *is* water.

### The stationary-vessel miss, diagnosed

```bash
python scripts/diagnose_miss.py --mmsi 578000500
```

LE BOREAL, 142 m at 0.1 kn, scored **physics 0.315 against a 0.35 gate** — missed by 0.035. CFAR fired
at z=16.1 (gate 5.0) and the linked component passed every geometry gate, but `_analyse_candidate`
resolved the hull to **37 × 23 m**, a quarter of its length, and with `wake_length_m = 0.0` the wake and
Kelvin terms are **40 % of the score, structurally unavailable**. A *moving* ship with identical bad
segmentation clears the gate on wake alone. **The lesson is the error budget:** a stationary vessel has
no margin, so any other imperfection is fatal. Next lead is hull under-segmentation, not the wake terms.

Two suspects **exonerated by measurement** (do not re-derive): the `round_and_big` cap never fires
(elongation 1.61 > 1.5) and a sweep over 5 thresholds recovered **zero** AIS vessels; and sweeping the
ice-regime gate from 0.05 to disabled changes Svalbard **not at all**, because the candidate dies at
the physics gate first.

---

## 9. How this compares to published work

**The short answer: our detection quality is in the same range as the xView3 winners, on a far smaller
and easier evaluation, with a ground-truth protocol that cannot count the vessels we missed — and at
roughly 2–3× their throughput on a CPU instead of a V100.** Every part of that sentence needs its
caveat, so here they all are.

### The benchmark that matches this task: xView3-SAR

[xView3-SAR](https://arxiv.org/abs/2206.00897) is the closest published analogue — dark-vessel
detection in medium-resolution imagery, ground-truthed by AIS correlation plus expert annotators. It
is much bigger than anything we built: 991 Sentinel-1 scenes averaging 29,400 × 24,400 px, **243,018
verified objects** over 43.2 million km², across global sea states.

| xView3 Challenge, holdout partition | Aggregate | **F1 detection** | F1 close-to-shore |
| :-- | --: | --: | --: |
| 1. BloodAxe | 0.6177 | **0.7702** | 0.5310 |
| 2. selim_sef | 0.6047 | 0.7629 | 0.4768 |
| 5. Kohei | 0.5717 | 0.7342 | 0.4527 |
| xView3 reference model (Faster-RCNN) | 0.1904 | 0.4302 | 0.1293 |
| **This applet, 16 US Sentinel-2 scenes** | — | **≤ 0.805** | **not attempted** |

Our F1 uses a consistent population: TP = 242 hand-confirmed vessels among our contacts, FP = 105
adjudicated false alarms, FN = 12 AIS-visible broadcasters missed in clear water →
F1 = 2·242 / (2·242 + 105 + 12) = **0.805**.

### Five reasons that number is not a like-for-like win

1. **Our FN is a lower bound, so our F1 is an upper bound.** We adjudicated *our own detector's
   output*. A dark vessel we never detected leaves no trace in our labels — only AIS-broadcasting
   misses are countable. xView3's annotators labelled the imagery independently of any detector, so
   their recall denominator is honest and ours is optimistic. This is the single biggest caveat.
2. **We exclude the hard case by design.** The 200 m shoreline keep-out drops everything close to
   shore. That is exactly the sub-task where the xView3 winner falls from 0.770 to **0.531** — the
   hardest part of the problem, and we simply do not attempt it.
3. **Different sensor, different difficulty.** SAR sees through cloud, at night, in any weather.
   Our VNIR payload needs daylight and a gap in the cloud; two of our eight sample scenes are rejected
   before detection for exactly that reason. xView3 models work in conditions where we produce nothing
   at all.
4. **Scale.** 243,018 objects across global waters versus our **347 adjudicated contacts** on 16 US
   coastal scenes, mostly calm and nearly cloud-free. Ours is a directional result on an easier
   distribution.
5. **They solve more of the problem.** The aggregate score also covers vessel/fishing classification
   (F1 0.939 / 0.843 for the winner) and length estimation. We classify nothing beyond
   broadcasting-or-not and our length estimate reads ~15 % short.

So: **comparable detection F1 on an easier subset, measured more permissively.** Not a win, and we
would not claim one in front of a judge.

### Where we are genuinely ahead: cost per pixel

| | Hardware | Throughput | Tag |
| :-- | :-- | --: | :-- |
| xView3 1st place | 1× Tesla V100 (~300 W), 60 GB RAM | ~13 min per 29,400 × 24,400 scene ≈ **0.9–1.8 Mpx/s** | published |
| **This applet** | x86-64 CPU, **no GPU**, 1.65 of 6 cores | 23.07 Mpx in 7.30 s = **3.16 Mpx/s** | [M] |

The challenge required inference "in under 15 minutes on a computer with one Tesla V100 GPU, 60 GB
RAM, and a server-grade CPU". We do the detection half of that work **2–3× faster with no GPU at all,
in 1.6 GB of RAM** — which is the entire argument for running onboard rather than downlinking. The
caveat: they run more tasks per pixel, and their figure is approximate and includes reading a 2.4 GB
band. This is a fair order-of-magnitude comparison, not a controlled benchmark.

### Optical Sentinel-2 work, and why those numbers look better than everyone's

Published Sentinel-2 vessel detection reports numbers well above ours — one 2024 method reports
precision 96.4 %, recall 94.3 %, F1 95.2 % against AIS. **Check the evaluation protocol before
comparing.** Most optical ship-detection work scores on pre-cropped patches centred on ships, where
the position match is true by construction and open ocean never gets the chance to produce a false
alarm. Scoring a whole scene end to end — where 5,672 km² of water can raise a contact anywhere — is a
different and much harder measurement.

**We have that effect measured inside this repo**, which is why we insist on it: our ice-retrained
verifier improved held-out *chip* AUC from 0.921 to 0.954 while end-to-end *scene* precision **fell**
from 0.697 to 0.655 (§3). Chip-level metrics moved one way and the thing we actually care about moved
the other. Our own SEN2MS chip scores (§5, AUC 0.979) are the flattering kind of number; the 0.697 is
the honest one.

### What we would need to make this a real comparison

Run this detector on the xView3 holdout split and submit against their metric. We cannot: xView3 is
Sentinel-1 SAR and this applet is a VNIR optical detector whose entire physics — NDWI water masking,
NIR contrast, sun-glint handling, the wake ray transform — has no meaning on a backscatter image.
The honest statement is the one above: same ballpark, easier problem, more permissive protocol,
much cheaper per pixel.

---

## 10. Where the numbers came from

| Input | Source | Licence |
| :-- | :-- | :-- |
| Sentinel-2 L2A / L1C | Copernicus, via Element84 Earth Search (public COGs, no login) | Copernicus open terms — *contains modified Copernicus Sentinel data* |
| US AIS, 2024 | NOAA / BOEM Marine Cadastre | US Government work, public domain |
| Svalbard AIS | Kystverket, via Kystdatahuset REST | NLOD — credit Kystverket |
| SEN2MS ship chips | Zenodo record 15571607 | CC-BY-4.0 |
| Hand labels (571 US + 51 Svalbard) | this team — `data/labels/` | ours |
| Synthetic scenes | `simulation/scene_synth.py`, seeded | ours |
| Verifier weights | trained from scratch — **no pretrained weights** | ours |

Considered and **not** used: MASATI (research-use-only licence, RGB with no NIR).

---

## 11. What we will not claim

* **No Jetson.** No TensorRT, DLA, power or thermal validation. Every bundle carries
  `"validated_on_hardware": false`.
* **No CPU figure from inside QEMU.** It counts the emulator; the flag says so on the same screen.
* Real validation is at **10 m** Sentinel-2, not the 4.75 m target GSD.
* **Arctic recall is n=5** and **Bay of Fundy is n=2.**
* **Optical does not solve dense ice.** Precision 0.077 at 25 % ice, and `--arctic` lifts the remainder
  to 0.545 only by *demoting*, on two scenes, with a threshold read off those same two scenes.
* **17 of 49 Arctic contacts were uncallable by a human.** Precision is quoted over the 32 that were not.
* **Stationary vessels have no error budget** (§8), and a small stationary hull in a growler field will
  be called ice.
* Hull length reads ~15 % short temperate, **59.7 % short Arctic (n=2)**.
* **Wake-derived speed fired on 0 of 375 US contacts.** It is advisory and cannot raise an anomaly alone.
* **The Kelvin-arm measurement has never fired**, even on synthetic ships rendered with arms.
* The governor's rung table is now measured on **both** synthetic tuning scenes and the 16 real
  scenes, and they disagree below REDUCED — trust the real one (§7).
* **We have not run this detector on xView3 or any other public benchmark**, so "comparable F1" (§9)
  is a cross-benchmark reading, not a ranked result. Our recall denominator cannot count dark vessels
  we never detected, which makes our F1 an upper bound.
* Ships moored alongside, or within 200 m of land, are not reported — by design.

---

## Reproduce all of it

```bash
python scripts/setup_data.py --all          # every dataset, public sources, no accounts
python -m pytest -q                         # 147 tests
python scripts/scorecard.py -i data/real/s2_us_bundle --labels data/labels/us_labels.json \
    --reference data/labels/us_reference_contacts.json                      # §1
python scripts/scorecard.py -i data/real/s2_us_l1c_bundle --labels data/labels/us_labels.json \
    --reference data/labels/us_reference_contacts.json                      # §2
python scripts/scorecard.py -i data/real/svalbard_poc --arctic --labels data/labels/svalbard_labels.json \
    --reference data/labels/svalbard_reference_contacts.json                # §3
python scripts/scorecard.py -i data/real/arctic_probe                       # §3
python scripts/scorecard.py -i data/real/fundy_bundle                       # §4
python training/eval_verifier.py                                            # §5
python scripts/benchmark.py --full-swath -n 3                               # §6
./scripts/run_emulated.sh                                                   # §6
python scripts/soak.py -i data/real/s2_us_bundle -n 25 --quiet              # §6
python scripts/orbit_pass_sim.py --input data/eval_bundle --repeats 5       # §7
python scripts/iceberg_study.py                                             # §8
python scripts/diagnose_miss.py --mmsi 578000500                            # §8
```
