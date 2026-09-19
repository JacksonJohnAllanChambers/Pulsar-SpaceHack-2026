# Tactical Edge Sentinel — pitch and demo guide

Pulsar SpaceHack 2026, Track 1. Rubric weights: Edge relevance 25, Technical 30, Innovation 20,
Impact 15, Presentation 10.

> Every number in this document is measured and reproducible from this repo. If a number is not
> here, we do not claim it on stage. The fastest way to lose a technical panel is to be caught
> rounding in our own favour, and the fastest way to win one is to volunteer a limitation before
> they find it.

---

## 1. The narrative

* **Company thesis (Galaxia):** *"Data is not the mission. Intelligence is."*
* **The bottleneck:** a Simera HyperScape100 generates far more imagery per day than a
  LEO downlink can carry. A ground pass is minutes long. A 19 km swath is ~180 MB of raw samples.
* **The consequence:** you cannot downlink the ocean. You must decide *on orbit* what is worth
  sending, and everything you discard, you discard forever.
* **Our answer:** find every vessel in the scene onboard, measure its heading from its wake,
  check it against the uplinked AIS picture, and downlink **~11 KB** that says which ships are not
  broadcasting, which are lying about their course, and which broadcasters are not where they
  claim to be. A full 32 MB real scene becomes an 11 KB tactical bundle — **2 900–13 000× less data.**

---

## 2. Six slides, three minutes

### Slide 1 — The downlink bottleneck
**Visual:** 180 MB raw swath vs. the minutes-long pass window; a map of the area a single
nanosatellite covers per day.
**Say:** "Dark vessels exploit the gap between when a satellite *sees* and when anyone on the ground
*knows*. We close it inside the pass."

### Slide 2 — The cascade: physics first, CNN second
**Visual:** the five stages, with the data volume shrinking at each arrow.

```
raw 4-band scene ──► screener ──► ring-CFAR + wake ray transform ──► 47k-param INT8 CNN
                     (masks)      (a few dozen 64x64 chips)          (0.09 ms/chip)
                                        └──► AIS correlation ──► ~11 KB tarball
```

**Say:** "Water reflects about 1% in NIR; steel and foam 10–40%. A local-contrast test finds a
10-pixel hull in a few OpenCV calls. Running a detector network over 16 megapixels of empty ocean
would cost orders of magnitude more energy to reach the same candidates. The CNN only ever sees
the few dozen chips the physics already flagged."

**The innovation line:** "At 4.75 m the Kelvin cusp arms are rarely resolved, so we do not chase the
textbook V. We cast 360 rays from each hull and find the turbulent centreline — that always resolves,
and the opposite bearing is the heading. Median heading error against real AIS is 4 degrees."

### Slide 3 — Hero demo: the ground console
**Visual:** `python -m ground.server`, live. See §3 for the exact click path.
**Say:** "This is the unmodified onboard code path. Same functions, same thresholds — the console
only visualises what the satellite computed."

### Slide 4 — Edge relevance (rubric 25%)
**Visual:** the measured budget table.

| Workload | Wall clock | Peak RAM | Raw → downlink |
| :-- | --: | --: | --: |
| 6 × 1024² synthetic scenes | 0.9 s | 0.41 GB (2.9% of 14 GB) | 54 MB → 6.8 KB |
| + one 4096² full swath | 7.2 s | 1.43 GB (10%) | 182 MB → 14 KB |
| 16 × 2048² real Sentinel-2 scenes | 12.2 s | 2.44 GB (17%) | 5,669 km² of water searched |

**Say plainly:** "We had no Jetson. These are x86 dev-box numbers; ARM64 numbers come from the same
container running natively on Apple Silicon. TensorRT and DLA are wired but unvalidated, and we will
not pretend otherwise. What *is* validated is the CPU path — the one that runs in your emulated
container with no GPU."

### Slide 5 — Evidence on real data (rubric 30%)
**Visual:** the real-scene scorecard (§4). Lead with the independent test.
**Say:** "Sixteen real Sentinel-2 scenes across every US coast, each paired with that day's public
NOAA AIS — a source the imagery knows nothing about. The geolocation comes from the satellite's own
transform, so when a contact lands on top of a broadcasting ship, that is evidence, not bookkeeping."

### Slide 6 — Impact and summary
* Sovereign-waters surveillance, illegal fishing, sanctions evasion, search and rescue cueing.
* **Outcome:** actionable intelligence inside the current pass instead of hours later.
* **Close:** "We turned a passive sensor into a tactical sentinel. Data is not the mission."

---

## 3. Demo script (rehearse this exact path)

```bash
python -m ground.server            # http://127.0.0.1:8050 — no CDN, no internet
```

1. **Pick `data/real/s2_us_bundle`** and run. Narrate while it works: screener → detector → CNN → AIS.
   *(Demo one or two scenes if the machine is tight: the console keeps rasters in memory for the
   layer views, so all 16 at once is a memory hog that buys you nothing on stage.)*
2. **Layer toggles** — true colour → NIR. "This is why we detect in NIR: the water goes black and
   the hull does not."
3. **CFAR z-score layer.** "This is the physics stage's view. Every bright blob is a candidate."
4. **Turn the CNN off, then on.** Watch the false alarms disappear. *This is the money shot* —
   the precision/recall trade, live, on a slider.
5. **Click a DARK_VESSEL.** Show the heading arrow along the wake, the chip, the plain-language note.
6. **Click a CONFIRMED vessel.** Show the dead-reckoned AIS position it matched and the distance.
7. **The downlink queue.** "Eleven kilobytes. The image was thirty-two megabytes. The anomalies got
   chips; the honest traffic did not — it never needs to be seen."

**Backup:** if the network or the room's machine misbehaves, `data/outputs/` holds a pre-built
tarball and telemetry from a previous run. Have the contact sheet
(`data/outputs/review/contact_sheet.html`) open in a second tab — it is a single self-contained file
and it makes the evidence tangible.

---

## 4. The numbers (keep this slide honest)

**Real scenes, real AIS, independent geolocation** — `python scripts/scorecard.py -i data/real/s2_us_bundle`

> Filled in from `data/outputs/us_scorecard/scorecard.json`. Report recall against the broadcasters
> the sensor could actually see: a ship alongside a quay is dropped on purpose by the 200 m shoreline
> keep-out, and counting it as a miss would understate a number we can otherwise defend.

**Held-out real Sentinel-2 chips (SEN2MS, 10 m):** open-water recall 0.82, under-way recall 0.83,
heading median 4° vs AIS, 37 false alarms per 1000 km² with the CNN (73 without).

**Held-out synthetic (seed 4242, never used in development):** precision 0.90, recall 0.77, F1 0.83,
heading median 0.5°, AIS anomaly class correct 97%.

**Verifier:** 59 KB INT8 ONNX, validation AUC 0.984, 0.09 ms per chip on ORT CPU.

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

**"What's the weakest part?"**
"Wake-derived speed. It fires on only a few percent of vessels and has produced a 2× error on one
long-wake case, so it is advisory and cannot raise an anomaly on its own. Second weakest: the Kelvin
arm test has never fired even on synthetic ships rendered with arms — the CFAR background sigma is
inflated next to a bright hull and buries the cusp lines. Nothing downstream depends on it."

---

## 6. What we will not claim

Volunteer these before you're asked. It costs one sentence and buys the whole panel's trust.

* Real validation is at 10 m (Sentinel-2), not the 4.75 m target GSD.
* Ships moored alongside, or within 200 m of land, are not reported — by design.
* No Jetson: no TensorRT, DLA, power or thermal validation.
* Hull length reads about 20% short; the sub-pixel bow taper falls below the core threshold.
* The Kelvin-arm measurement never fires today.
* Vessels inside the land or cloud buffers are not reported.
