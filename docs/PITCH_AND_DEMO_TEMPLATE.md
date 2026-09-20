# Tactical Edge Sentinel — pitch and demo guide

Pulsar SpaceHack 2026, Track 1. Rubric weights: Edge relevance 25, Technical 30, Innovation 20,
Impact 15, Presentation 10.

> Every number in this document is measured and reproducible from this repo. If a number is not
> here, we do not claim it on stage. The fastest way to lose a technical panel is to be caught
> rounding in our own favour, and the fastest way to win one is to volunteer a limitation before
> they find it.

---

## 1. The spine

> **Finding ships in open water is easy. Not finding ice is hard.**
> We built the ice case, on Galaxia's payload, and measured it against AIS the imagery never saw.

Everything else is support for that sentence. If you only get one line out, it is that one.

Three facts that make it land, all sourced (see `docs/GALAXIA_ALIGNMENT.md`):

* **Galaxia already said this is what MÖBIUS-1 is for.** Launch release, 23 June 2025: *"Its Earth
  Observation capabilities support a diverse range of applications including maritime security,
  **tracking dark vessels** and combating illegal fishing operations."*
* **We designed against the payload they bought.** Galaxia acquired a Simera Sense **HyperScape100 on
  3 September 2026** — sixteen days before this event. 4.75 m GSD, 19.4 km swath, **460–860 nm VNIR**.
  That VNIR ceiling is not a detail: no SWIR means no NDSI, the standard ice index, so ice had to be
  separated from cloud inside the VNIR range alone. **The hard part of this project exists because of
  their sensor choice.**
* **You cannot downlink the ocean.** A 19 km swath is ~180 MB of raw samples; a ground pass is minutes.
  Everything you discard, you discard forever. We downlink **7.3 KB** that says which ships are not
  broadcasting and which are lying about their course.

---

## 2. Five slides, three minutes

Cut ruthlessly. Slide 3 is the one that wins; protect its time.

### Slide 1 — The problem, and why ice is the hard version (25 s)
**Visual:** an Arctic scene with the CFAR candidates overlaid — a field of bright, ship-sized blobs.
**Say:** "Dark vessels exploit the gap between when a satellite sees and when anyone on the ground
knows. In open water this is a solved-ish problem. In ice it is not: sea ice is a field of bright,
high-contrast, ship-sized objects on dark water — the exact signature a ship detector is built to
find. Every floe looks like a hull."

### Slide 2 — The cascade: physics first, CNN second (35 s)
**Visual:** five stages, data volume shrinking at each arrow.

```
raw 4-band scene ──► screener ──► ring-CFAR + wake ray transform ──► 47k-param INT8 CNN
                     (masks)      (a few dozen 64x64 chips)          (0.09 ms/chip)
                                        └──► AIS correlation ──► 7.3 KB tarball
```

**Say:** "Water reflects ~1% in NIR, steel and foam 10–40%. A local-contrast test finds a 10-pixel hull
in a few OpenCV calls. Running a network over 16 megapixels of empty ocean costs orders of magnitude
more energy to reach the same candidates. The CNN only ever sees the chips the physics already flagged."

**The innovation line:** "At 4.75 m the Kelvin cusp arms are rarely resolved, so we don't chase the
textbook V. We cast 360 rays from each hull and find the turbulent centreline — that always resolves.
Median heading error against real AIS is **4 degrees**. And in pack ice we run the same transform with
the sign reversed: a vessel under way leaves an open-water channel that is *darker* than the floes."

### Slide 3 — Evidence: real AIS the imagery never saw (60 s — PROTECT THIS)
**Visual:** the two scorecards side by side.

| | Temperate (16 US scenes, NOAA AIS) | **Arctic (Svalbard, Kystverket AIS)** |
| :-- | --: | --: |
| Recall vs AIS the sensor could see | **0.912** | 0.60 *(n=5 visible)* |
| Precision, hand-adjudicated | **0.697** | **0.188** |
| Position error vs dead-reckoned AIS | — | **median 20.4 m** |
| Heading error vs reported COG | 4° median | **2.2° median** |

**And the finding the whole pitch turns on** — precision against ice fraction, same code, same day,
two fjords 100 km apart:

| Scene | Sea ice (% of searched water) | Contacts | **Precision** |
| :-- | --: | --: | --: |
| Isfjorden | 9.7 % | 6 | **0.667** |
| Kongsfjorden | 25.5 % | 43 | **0.077** |

**Say:** "Sixteen Sentinel-2 scenes across every US coast, each paired with that day's public NOAA AIS —
a source the imagery knows nothing about. Every one of 571 contacts was then adjudicated by eye, because
AIS can prove a contact *is* a ship but cannot prove one isn't. Then we went north. NOAA stops at 50
degrees, so we used Norway's open Kystverket feed, which covers the Svalbard protection zone. We found
CRYSTAL ENDEAVOR, SILVER WIND and TEISTEN in 9-to-42-percent sea ice, positioned to **20 metres**."

**The ice result:** "Our sea-ice module cuts Arctic false alarms **23%** — 149 contacts to 115 — and
costs *nothing* in temperate water: recall is unchanged at 0.912."

**Say this, and do not soften it:** "Two-and-a-half times the ice, seven times the contacts, and
precision falls by a factor of nine. We are not going to stand here and tell you optical solves ice.
We measured exactly where it stops."

**The line that earns the room:** "We then hand-adjudicated all 49 contacts — and **17 of them, a third,
we could not call either way.** Not the detector failing: a trained eye on 10-metre pixels genuinely
cannot tell a small stationary hull from a floe. That is an information limit of a VNIR payload, not a
bug in our code. Which is exactly why the next spacecraft needs something that isn't a camera."

**What the applet does about it (10 s -- the first thing to cut if you are running long):** "So it stops
pretending. In a scene with ice it makes a three-way call -- ship, iceberg, or *uncertain* -- and sends
probable ice to the back of the downlink queue instead of raising it as a dark vessel. On Svalbard that
is **46 alerts down to 16, with zero real ships demoted**, and the bundle gets 40% smaller. Nothing is
ever deleted, and only a transponder is allowed to say 'ship'."


### Slide 4 — It fits the device (30 s)
**Visual:** the budget table.

| Workload | ARM64 (M4, 6 threads) | Peak RAM | Raw → downlink |
| :-- | --: | --: | --: |
| One 4096² full swath (19.4 km) | **1.61 s** | 2.09 GB (14.6% of 14 GB) | 182 MB → 9.4 KB |
| 16 real Sentinel-2 scenes, 5,669 km² | 3.03 s | 2.59 GB (18%) | 316 MB → 11 KB |

**Say:** "We had no Jetson. That's an M4 running the same code natively — faster than an Orin NX, so read
it as an upper bound. The same bundle produces a **byte-identical tarball** on x86-64 Windows, ARM64 macOS
and inside your linux/arm64 container, across different numpy and ONNX builds."

**Then the governor, in one breath:** "It also models its own thermal budget — in orbit there's no
convection, so 25 W is affordable in eclipse and not in sunlight — and sheds *evidence* before the
hardware sheds *throughput*. The invariant is **degrade the evidence, never the alert**: the first rung
keeps all 46 AIS-confirmed ships and cuts the downlink 60% — that rung table is measured on our **synthetic** tuning bundle, not on the real scenes. The degrees are **modelled**, we never had
hardware, and every bundle says `validated_on_hardware: false` inside it."

*Only open the orbit panel if they ask. Details in Appendix A.*

### Slide 5 — Impact and close (20 s)
* Sovereign-waters surveillance, illegal fishing, sanctions evasion, SAR cueing — 243,000 km of Canadian
  coastline, and a DRDC contract whose stated purpose is validating onboard edge compute.
* **Close:** "Galaxia said MÖBIUS-1 is for tracking dark vessels, and that the future of space is how fast
  you turn data into intelligence. We took a 182 megabyte swath down to a 7.3 kilobyte decision, onboard,
  on the payload you bought three weeks ago — in sea ice, against AIS the imagery never saw. And we can
  tell you exactly which of our numbers are measurements and which are models."

---

## 2b. Appendices — only if asked

### A. The thermal governor in full
Sustainable SoC power **34.6 W in eclipse vs 18.9 W sunlit** (direct sun puts 24.5 W back onto the same
radiating face). Predicted junction at 25 W: 53 °C eclipse, **94 °C sunlit against a 95 °C throttle**. A
dawn-dusk SSO never gets an eclipse, and the console shows it settling permanently on a shallower rung.
Four-rung ladder, measured: FULL 18 dark / 46 AIS / 18.0 KB · **REDUCED 18 / 46 / 7.3 KB (free)** ·
SURVEY 18 / 41 / 6.6 KB · BEACON 121 / 44 / 9.9 KB.

**If asked about speed, do not invent one:** the power cap and worker count are exact configuration; what
that does to wall-clock we could not measure — spread within one rung exceeded the gap between rungs, and
re-running moved the median across 6.51 / 8.33 / 9.73 s. We withdrew a "9% slower" line rather than defend
noise.

### B. What their next satellite should carry
`python scripts/cue_geometry.py`. HyperScape100 is a *confirm* sensor — 19.4 km is not a search width. A
15-knot vessel runs **13.9 km in 30 minutes**, most of the swath, so a cross-platform cue that must go
down, be correlated and come back up is on a stopwatch it usually loses.

And the measured kicker: at the only independently-assessed RF geolocation accuracy (ESA EDAP+ on
Unseenlabs — median 2.5 km, mean 5.4 km), the 95% gate is **20.8 km — wider than the entire swath**. So
the argument is *not* better geolocation. **Co-locating RF on the same bus removes the need for
geolocation**: "is anything transmitting in this frame?" is a far easier question than a 2.5 km fix — and
in our Arctic scenes, 115 ice contacts against zero emitters means that single bit rejects all of them.

Full sourcing in `docs/GALAXIA_ALIGNMENT.md`.

---

## 3. Demo script (rehearse this exact path)

```bash
python -m ground.server            # http://127.0.0.1:8050 — no CDN, no internet
```

**Six clicks. Everything else is an appendix.** Demo one or two scenes, not all 16 — the console holds
rasters in memory for the layer views and the full set buys you nothing on stage.

1. **Run `data/real/svalbard_poc`.** Lead with the hard case, not the easy one. Narrate while it works:
   screener → detector → CNN → AIS.
2. **True colour → NIR.** "This is why we detect in NIR: water goes black, the hull does not — and so
   does the open-water channel behind a ship in ice."
3. **CFAR z-score layer.** "Every bright blob is a candidate. In ice, most of them are floes. This is
   the problem in one image."
4. **Turn the CNN off, then on.** False alarms disappear. *Still the money shot* — the precision/recall
   trade, live, on a slider.
5. **Click CRYSTAL ENDEAVOR.** Confirmed against Kystverket AIS the imagery never saw — show the
   dead-reckoned position it matched and the distance. **20 metres.**
6. **The downlink queue.** "Seven kilobytes. The scene was hundreds of megabytes. Anomalies got chips;
   honest traffic did not — it never needs to be seen."

**Optional seventh click, if slide 3's ice line landed:** tick `Ship or ice`, re-run. Pass summary goes
**46 dark -> 16 dark + 30 "Probable ice (demoted)"**, the queue re-sorts with ice-blue contacts at the
bottom, and CRYSTAL ENDEAVOR is untouched. Click one ice contact and read its reason aloud: *"13 other
bright objects within the chip, 15% scene ice, no wake, lead or AIS."* `Ship or ice?` in the header is
Megan's explainer page if someone wants the logic.

**If they ask about the Jetson / thermals**, then and only then: tick `Thermal governor`, set Orbit to
`Dawn-dusk SSO (no eclipse)`, re-run. The card drops to **REDUCED while the die is still cold** — *"it
isn't reacting to heat, it's looking ahead; in a dawn-dusk orbit no eclipse is coming."* Downlink card:
**18.0 → 7.3 KB, same 18 dark vessels, same 46 AIS-confirmed ships**, targets now reading "chip shed by
governor". That is the invariant on screen. "Show across an orbit →" has the two traces; read the
`MODELLED` badge out loud before anyone asks.

**Backup:** `data/outputs/` holds a pre-built tarball and telemetry from a previous run. Keep a contact
sheet open in a second tab (`data/outputs/svalbard_review/contact_sheet.html`) — one self-contained file,
and it makes the evidence tangible.

---

## 4. The numbers (keep this slide honest)

**Temperate benchmark — 16 US scenes vs same-day NOAA AIS** (`scripts/scorecard.py -i data/real/s2_us_bundle`).
571 contacts hand-adjudicated. **Recall 0.912 · precision 0.697 · 105 false alarms** (18.9 per 1000 km²).

> Recall is reported against the broadcasters the sensor could actually see: a ship alongside a quay is
> dropped on purpose by the 200 m shoreline keep-out, and counting it as a miss would understate a number
> we can otherwise defend.

**Arctic benchmark — Svalbard vs Kystverket AIS** (`scripts/scorecard.py -i data/real/svalbard_poc`).
Norway's open feed covers the Svalbard protection zone, where NOAA stops at 50.195 N. Two scenes,
592 km² of searched water, 9–42 % sea ice:

| | |
| :-- | --: |
| AIS broadcasters in footprint / visible to sensor / detected | 8 / 5 / **3** |
| Recall vs visible AIS | **0.60** *(n=5 — a real number, not a big one)* |
| Position error vs dead-reckoned AIS | **median 20.4 m**, p90 61.1 m, 100 % within 200 m |
| Heading error vs reported COG | **median 2.2°** |
| Contacts with no AIS | 46 (77.8 /1000 km², upper bound) |
| **Adjudicated** (32 of 49 reviewed, 17 uncallable) | **precision 0.188**, 26 true false alarms (43.9 /1000 km²) |
| Recall vs all hand-confirmed vessels | **0.75** (6 of 8) |

Per-scene, and this is the result to lead with: **Isfjorden 9.7 % ice → precision 0.667** ·
**Kongsfjorden 25.5 % ice → precision 0.077**.

Found: CRYSTAL ENDEAVOR (164.5 m), SILVER WIND (155.8 m), TEISTEN (17 kn). One genuine miss worth owning
if asked: LE BOREAL, 142 m at 0.1 kn, flagged `CLEAR_WATER_NO_TARGET`.

**Sea-ice module** — Arctic contacts 149 → **115** (−23 %), US recall **unchanged** at 0.912, precision
0.693 → 0.697. The physics module is free: it costs nothing in temperate water.

**Ship or ice (`--arctic`, opt-in; Megan's three-way design)** -- `scripts/scorecard.py -i
data/real/svalbard_poc --arctic --labels data/outputs/svalbard_review/labels.json`:

| Svalbard, hand labels | off | `--arctic` |
| :-- | --: | --: |
| Raised as dark vessels | 46 | **16** |
| Demoted as ice: clutter / could-not-tell / **vessels** | -- | 21 / 9 / **0** |
| Precision of the alerts that remain | 0.188 | **0.545** |
| Tarball | 14.3 KB | **8.7 KB** |

16 US scenes with it switched on: **contact-for-contact identical**, no field added. The evidence is
crowding inside a scene the screener found ice in -- calved glacier ice arrives as a field -- with any
wake, lead or AIS match protecting the contact. Three alternatives were measured and rejected
(`scripts/iceberg_study.py`): centre-window brightness/texture fired on **0 of 539** real contacts; a
fill-invariant spectral slope is real physics (NIR/visible 0.46 ice vs 0.77-1.0 hulls) but caught **1 of
104** as a frozen gate; and "a wake means ship" was wrong on **21 of 21** Alaska contacts.
**Say the caveat with the number:** the threshold was read off these same two scenes and every Svalbard
vessel is large, so 0.545 is optimistic.

**Held-out real Sentinel-2 chips (SEN2MS, 10 m):** open-water recall 0.82, under-way recall 0.83,
heading median 4° vs AIS, 37 false alarms per 1000 km² with the CNN (73 without).

**Held-out synthetic (seed 4242, never used in development):** precision 0.90, recall 0.77, F1 0.83,
heading median 0.5°, AIS anomaly class correct 97%.

**Verifier:** 59 KB INT8 ONNX, validation AUC 0.984, 0.09 ms per chip on ORT CPU.

**Thermal governor — the exact half** (`python scripts/orbit_pass_sim.py --input data/eval_bundle
--repeats 5`; **40 synthetic tuning scenes at 4.75 m, seed 777**, NOT the real-scene benchmark above;
same code, same bundle, only the cascade profile varies; deterministic given a profile):

| Rung | Power cap | Workers | Candidates | Dark | AIS-confirmed | Chips | Downlink |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| FULL | 25 W | 3 | 3 835 | 18 | 46 / 46 | 28 | 18.0 KB |
| REDUCED | 15 W | 2 | 3 544 | 18 | 46 / 46 | 0 | 7.3 KB |
| SURVEY | 15 W | 1 | 2 552 | 18 | 41 / 46 | 0 | 6.6 KB |
| BEACON | 10 W | 1 | 1 498 | 121 | 44 / 46 | 0 | 9.9 KB |

**No timing number appears here on purpose.** Re-running the measurement script moved our FULL
wall-clock median across 6.51 / 8.33 / 9.73 s — same machine, same data. Drift between runs is bigger than
the gap between rungs, and two careful attempts disagreed in sign (five sequential repeats said REDUCED
used 30% less CPU than FULL; six interleaved rounds said 11% more). We withdrew the "9% slower" line an
earlier draft carried rather than defend either. If a judge presses, this is a good answer to have ready:
*we know which of our numbers are measurements and which are not.*

**Thermal governor — the modelled half:** sustainable SoC power 34.6 W in eclipse vs 18.9 W sunlit;
predicted steady-state junction at 25 W is 53 °C in eclipse and 94 °C sunlit against a 95 °C throttle
point. Inferred from measured CPU utilisation against NVIDIA's published nvpmodel envelopes.
**Never validated on hardware**, and every bundle says so in its own telemetry.

---

## 5. Judges' questions, answered honestly

**"Why not downlink the image and process on the ground?"**
"You only see a ground station for minutes per orbit. Downlinking raw imagery backs up the queue for
hours, and by then the contact has moved. An 11 KB alert arrives during the current pass. The
compression ratio here is 2 900–13 000×, and it is lossless *with respect to the decision* — we are
not sending a smaller picture, we are sending the answer."

**"What about whitecaps and rough seas?"**
"Ring-CFAR adapts its threshold to the local sea state, so a rising clutter floor raises the bar
automatically. What CFAR cannot do is tell a whitecap from a small hull by brightness alone — that is
what the CNN is for, and on real Sentinel-2 it halved our false alarms without losing a single real
ship. Wake-less vessels under about 25 m in a gale are indistinguishable from whitecaps, and we drop
them by design rather than flood the operator."

**"How does the satellite know about AIS with no internet?"**
"Ground control uplinks a small JSON catalogue before the pass — kilobytes. Onboard we dead-reckon
each broadcaster to the exact shutter second. Real data taught us a lesson here: a day-old fix from
the same berth once 'confirmed' a ship that was actually silent, so fixes older than three hours can
no longer identify a contact."

**"Your false-alarm number — how do you know those are false?"**
"We don't, from AIS alone. 'No transponder' is not 'no vessel' — most of ours are small craft under
way with clear wakes, which is exactly what a dark-vessel product should surface. So we say *upper
bound*, and then we adjudicate every contact by eye against the pixels and publish the true
precision separately. The contact sheet is in the repo."

**"Did you run on a Jetson?"**
"No. Teams weren't given hardware, so TensorRT engines, DLA, memory contention, power and thermals
are unvalidated, and we say so in the README. We minimised the exposure: the CPU path is the verified
one, the model is 59 KB, and ONNX Runtime selects the provider at load time, so moving to TensorRT
changes speed and nothing else we'd have to re-validate."

**"Where is the GPU in all this? It's a Jetson."**
"Mostly idle, on purpose. The expensive step in ship detection is looking at 16 megapixels of empty
ocean, and physics does that in a few OpenCV calls; the network only ever sees a few dozen 64x64 chips,
0.09 ms each. Our own thermal model says a sunlit radiator sustains about 19 W, so an applet that needs
the GPU flat-out is one that gets throttled. What we leave free is the point: the GPU and DLA stay
available to whatever else the payload runs, and we share the bus with ADCS and comms at 15-18% of RAM.
The CNN goes through ONNX Runtime, so on real hardware the TensorRT provider is picked up at load time --
written, never run, and we say so."

**"So can you tell a ship from an iceberg?"**
"Not from the object, and we measured why. At 10 metres a growler and a small hull are both a dozen
pixels; a person couldn't call 17 of 49 either. Spectrally ice *is* different -- bluer, darker in NIR --
and that survives sub-pixel mixing, but white superstructure and wake foam look the same, so as a gate it
caught 1 contact in 104. What does work is context: glacier ice arrives as a field, ships at sea don't.
So in an icy scene, a contact crowded by other bright objects with no wake, no lead and no transponder
is demoted -- never deleted -- and everything else stays a full-priority alert. A small boat stopped
inside a growler field will be called ice. That's the limit of the sensor, and it's why the demotion
keeps the position and the reason in the bundle. It's also not wasted output: an iceberg list is what
the Canadian Ice Service and the International Ice Patrol publish."

**"Then how can you claim anything about temperature?"**
"We don't claim it — we model it and label it. The setup guide told us the container can't show power or
thermal behaviour, and the rubric asks about temperature anyway, so the only honest move is to build the
model, state its assumptions, and use it for something real. The spacecraft bus parameters are ours, not a
flown bus's; the Orin power envelopes are NVIDIA's published ones; the CPU utilisation it's driven from is
genuinely measured in the container. The console badges it `MODELLED` and every bundle carries
`validated_on_hardware: false`. If you hand us an Orin NX for an hour, `JetsonThermalSource` already reads
`/sys/.../thermal_zone*/temp` and the model steps aside — that path is written and has never been run."

**"Does the governor make the pass slower?"**
"We don't know, and we won't guess. The power cap and the worker count are exact because they're
configuration; what that does to wall-clock we could not measure — the run-to-run spread on our machine
is wider than the effect, and two honest attempts gave opposite signs. So we report the deterministic
columns and leave the timing blank. What we can say is the comparison that matters: the alternative to
degrading deliberately is the hardware throttling, which costs you wall-clock anyway and gives you no say
in *what* degrades. Ours chooses — the first rung drops only the JPEG evidence crops and keeps all 46
AIS-confirmed ships and all 18 dark-vessel alerts while cutting the downlink 60%."

**"What's the weakest part?"**
"Wake-derived speed. It fires on only a few percent of vessels and has produced a 2× error on one
long-wake case, so it is advisory and cannot raise an anomaly on its own. Second weakest: the Kelvin
arm test has never fired even on synthetic ships rendered with arms — the CFAR background sigma is
inflated next to a bright hull and buries the cusp lines. Nothing downstream depends on it."

---

## 6. What we will not claim

Volunteer these before you're asked. It costs one sentence and buys the whole panel's trust.

* Real validation is at 10 m (Sentinel-2), not the 4.75 m target GSD.
* **Arctic recall is n=5.** Two Svalbard scenes, five visible broadcasters, three found. Directionally
  real, statistically thin, and we say the n out loud rather than quoting 0.60 bare.
* **Arctic precision is 0.188**, against 0.697 temperate. In 25 % ice it is 0.077. We do not dress
  this up: dense ice defeats a VNIR detector, ours included.
* **Ship-or-ice lifts that to 0.545 only by demoting, on two scenes, with a threshold read off those same
  scenes.** Every Svalbard vessel is large. A small stationary hull inside a growler field will be called
  ice -- demoted with its position and reason kept, its JPEG chip not sent. It is off by default.
* **17 of 49 Arctic contacts could not be adjudicated by eye at all.** Precision is quoted over the
  32 that could be. If the uncallable ones were all clutter, precision would be 0.122.
* Arctic hull length error is 59.7 % median (n=2) against ~20 % temperate -- likely ice fragments
  merging into the hull blob, but n=2, so we are not explaining it yet either.
* Ships moored alongside, or within 200 m of land, are not reported — by design.
* No Jetson: no TensorRT, DLA, power or thermal validation. The thermal governor's temperatures are a
  model calibrated to NVIDIA's published power envelopes, never a measurement — and the spacecraft bus
  parameters it assumes (radiator area, coating, parasitic load) are ours, not a real bus's.
* The governor is **off by default**: a governed pass depends on host timings, so it stops being a pure
  function of its input bundle. We turn it on in the demo and say so.
* We cannot quote a speed effect for the governor. Our hardware is too noisy to measure it, and we
  say so rather than publish a number we cannot reproduce.
* Hull length reads about 20% short; the sub-pixel bow taper falls below the core threshold.
* The Kelvin-arm measurement never fires today.
* Vessels inside the land or cloud buffers are not reported.
