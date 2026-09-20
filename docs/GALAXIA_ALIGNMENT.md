# Tactical Edge Sentinel × Galaxia Mission Systems

**What this document is.** A technical argument that this applet belongs on Galaxia's spacecraft,
written for someone who works there. Every claim about Galaxia is sourced to a dated public
announcement and linked at the bottom. Every claim about our own system is either measured in this
repo or explicitly labelled as a model.

**What it is not.** A pitch deck. Where our system cannot do something, §7 says so plainly rather
than leaving it for someone to find.

---

## 1. Why this project, for this company

Galaxia's own launch release for MÖBIUS-1 states that its Earth-observation capabilities

> "support a diverse range of applications including **maritime security, tracking dark vessels and
> combating illegal fishing operations**."
> — [Galaxia, MÖBIUS-1 launch release, 23 June 2025][mobius]

That is this project, stated in Galaxia's words before we started. Three further facts fix the design
target precisely:

| Fact | Date | Source |
| :-- | :-- | :-- |
| MÖBIUS-1 launched on Falcon 9 / Transporter-14 — first commercial satellite built and launched in Atlantic Canada | 23 Jun 2025 | [newswire][mobius] |
| **Onyx Edge** announced: "Core and Blades", flight computer physically separated from the AI compute cluster; hardware foundation for **VSaS** | 24 Jul 2026 | [SpaceQ][onyx] |
| **HyperScape100 acquired from Simera Sense** — hyperspectral, maritime surveillance named as a target application | 3 Sep 2026 | [SpaceQ][hyper-buy], [SpaceNews][hyper-sn] |
| $2.5M DRDC contract for the Canadian Tactical Operations Satellite (CTOS), to validate data relay and **edge computing** in orbit | Sep 2025 | [SpaceQ][ctos] |

And the thesis, from the Onyx Edge announcement:

> "The future of space will not be defined by how much data satellites collect, but by how quickly
> they can turn it into intelligence."
> — Arad Gharagozli, CEO, [24 July 2026][onyx]

Our applet turns a 182 MB swath into a **7.3 KB** decision packet onboard, and never downlinks the
ocean it discarded. That is the same sentence, implemented.

> **One honest note on provenance.** The hackathon's own two PDFs (`Hackathon_Project_Track_Guide.pdf`,
> `Hackathon_Jetson_Setup_Guide.pdf`) specify **no sensor, no company and no dataset** — they say
> explicitly that teams source their own inputs. We chose the HyperScape100 as our reference payload
> because Galaxia announced buying one three weeks before this event, not because anyone told us to.

---

## 2. Where this applet sits in the Onyx Edge architecture

Onyx Edge splits the spacecraft into a **Core** (flight computer, ADCS, telemetry, fault recovery —
radiation-resilient, must never be at risk) and **Blades** (the AI compute cluster). That split is
exactly the deployment contract our applet was written against, by coincidence of good practice
rather than design:

```
   HyperScape100 ──LVDS/SpaceWire──►  [ BLADE ]  Tactical Edge Sentinel
                                          │      screener → CFAR + wake ray transform
                                          │      → 59 KB INT8 CNN → AIS correlation
                                          ▼
                                     7.3 KB packet ──► [ CORE ] ──► downlink
```

Concretely, what makes it Blade-shaped:

* **One entry point.** Everything runs through `run_pass()`. A VSaS tenant does not need our CLI,
  our console or our tests — it needs one function and a directory of bands.
* **It cannot take the spacecraft down.** Every stage degrades rather than raises: a missing band, a
  corrupt file, an absent ONNX runtime or a failed model load all fall back to a reduced result and
  the pass still produces a bundle. 150 tests, many of which exist only to prove that.
* **It already runs in the flight container.** `linux/arm64`, `--memory=14g --cpus=6 --network none`,
  no GPU, no internet at runtime. Peak RSS ~1.0–2.6 GB depending on swath size.
* **No accelerator dependency.** ONNX Runtime selects its provider at load time: CPU in the container,
  TensorRT/CUDA if a Blade offers them. Moving to an accelerator changes speed and nothing we would
  have to re-validate.
* **Deterministic.** The same bundle produces a byte-identical tarball (same SHA-256) on x86-64
  Windows, native ARM64 macOS and inside the linux/arm64 container, across different numpy / OpenCV /
  ONNX Runtime builds. For a VSaS tenant whose output is evidence, that is the property that matters.

---

## 3. Payload fit: HyperScape100

Manufacturer figures we designed against ([Simera Sense datasheet][simera]):

| | |
| :-- | :-- |
| GSD / swath at 500 km | 4.75 m / 19.4 km |
| Spectral range | 460–860 nm (VNIR) |
| Bands | **32 simultaneous, selected from a library of 400**, reconfigurable in orbit |
| Depth, dTDI | 12 bit; 1–8 stages, set **per band** |
| Mass / power | 1.26 kg; 4.0 W idle, 7.75 W imaging |
| Interface | LVDS or SpaceWire |

**What we run on today.** Real Sentinel-2 at 10 m, four bands (R/G/B/NIR), as a VNIR proxy — plus
synthetic scenes rendered at 4.75 m. Every threshold in the applet is expressed in **reflectance and
metres**, never in DN or pixels, and pixel sizes are derived per scene from the manifest GSD. That is
why the same configuration runs unchanged on 10 m Sentinel-2 and 4.75 m HyperScape100 imagery.

**What the 460–860 nm ceiling already forced us to solve.** No SWIR means no NDSI, the standard
snow/ice index. Sea ice is a field of bright, high-contrast, ship-sized objects on dark water — the
exact signature a ship detector is built to find — so it is the dominant Arctic false-alarm source and
we could not use the textbook index to remove it. We separated ice from cloud *inside the VNIR range
alone*: ice absorbs toward 865 nm while cloud droplets scatter almost neutrally, so at equal brightness
ice keeps a positive NDWI and cloud sits near zero. Measured on four real Arctic scenes the boundary
lands at +0.05 in every one. **This work only exists because we targeted a VNIR-only payload.**

**What we would do with 32-of-400 selectable bands — and have not done yet.** The reconfigurability is
the most interesting unexploited property of this imager. The honest status: our four-band data cannot
answer which 32 bands are optimal, and we will not pretend otherwise. What we can say is what the
question is worth asking:

* Which bands maximise separability of **steel hull vs. sea ice vs. open water vs. cloud**? Our ice/cloud
  discriminant currently rests on two bands (green, NIR). With a 400-band library that is a
  band-selection optimisation, not a hand-tuned threshold.
* The imager sets **dTDI per band** (1–8 stages). At high latitudes the sun is low and SNR is the binding
  constraint; spending TDI stages on the two or three bands the discriminant actually depends on, and
  fewer elsewhere, is a free SNR gain at fixed data rate.
* Bands are **re-selectable between passes**. An Arctic pass and a temperate pass want different sets. A
  payload that reconfigures per pass wants an applet that reconfigures with it — see §8.

---

## 4. The thermal governor answers a constraint Galaxia has publicly named

Sustained onboard AI across consecutive polar orbits is power- and heat-limited: in LEO the only heat
sink is conduction to structure and radiation from an external face, and polar inclinations add
radiation dose. This is a real constraint on any "run neural networks on the bus" architecture.

We built the applet to treat that as an input rather than discover it by throttling
(`applet/core/thermal.py`, `applet/core/governor.py`):

* A two-node lumped-capacitance model (SoC junction → chassis → deep space) predicts junction
  temperature from **measured** CPU utilisation against NVIDIA's published nvpmodel envelopes.
* The result that matters, for our assumed bus: **sustainable SoC power is 34.6 W in eclipse and 18.9 W
  in sunlight**, because direct sun puts 24.5 W back onto the same face that has to do the rejecting.
  A 25 W mode is affordable in eclipse and is not affordable in sunlight — and a **dawn-dusk
  sun-synchronous orbit never gets an eclipse at all.**
* So the applet degrades on one invariant: **degrade the evidence, never the alert.** Measured on the
  40 synthetic tuning scenes at 4.75 m, seed 777 (`data/eval_bundle`) — not the real-scene benchmark in §5 —
  the first rung is free: all 46 AIS-confirmed ships and all 18 dark-vessel alerts are kept while the
  downlink drops 18.0 KB → 7.3 KB. It only stops sending the JPEG evidence crops.

**Everything thermal here is modelled, never measured.** We had no Jetson. The spacecraft parameters
(radiator area, coating, parasitic load) are ours, not a flown bus's. `JetsonThermalSource` reads
`/sys/.../thermal_zone*/temp` when it exists and the model steps aside — that path is written and has
never been executed. Every bundle we produce carries `"validated_on_hardware": false` inside it.

*For Galaxia this is the part that would change on contact with reality first, and the part where an
hour on a real Orin NX — or one Onyx Edge thermal model — would settle in an afternoon what we could
only predict.*

---

## 5. The Arctic case, measured

Four real Sentinel-2 Arctic scenes (Utqiagvik, Prudhoe Bay, Kotzebue, Point Hope), reproducible via
`scripts/fetch_sentinel2.py --region arctic --min-ice 5`.

| | Contacts | per 1000 km² | US recall | US precision |
| :-- | --: | --: | --: | --: |
| Baseline detector | 149 | 121.5 | 0.912 | 0.693 |
| **+ sea-ice module (shipped)** | **115** | **78.9** | **0.912** | **0.697** |

A 23 % cut in Arctic false alarms with **no loss of recall** on the temperate benchmark — the ice work
did not cost anything elsewhere.

Two things we will volunteer before anyone asks:

1. **There were no ships to find.** We hand-adjudicated all 149 baseline contacts across the four
   scenes: zero vessels. So this is a false-alarm result, not a detection result. We report precision
   only and do not claim Arctic recall.
2. **There is no open Arctic AIS.** NOAA Marine Cadastre stops at **50.195 °N** — no Alaska coverage at
   all. Our AIS correlation stage cannot be validated north of that line with public data. This is a
   genuine gap in the open-data ecosystem and it is worth knowing before anyone plans an Arctic
   validation campaign around it.

---

## 6. What the next spacecraft should carry, and why — from geometry, not opinion

The standard architecture is cue-and-confirm: wide-area SAR or RF finds a candidate, a
high-resolution imager confirms it. HyperScape100 is firmly a **confirm-tier** instrument — 4.75 m is
excellent, 19.4 km is not a search width. So the question is where the cue comes from.

That question has a hard geometric answer, and it is the main technical contribution of this document.
**A cue is only useful while the target is still inside the swath you point at it.** Run
`python scripts/cue_geometry.py`:

| Vessel speed | Max cue age (perfect pointing) | Max cue age (70 % usable swath) |
| --: | --: | --: |
| 10 kn | 31.4 min | 22.0 min |
| **15 kn** | **21.0 min** | **14.7 min** |
| 20 kn | 15.7 min | 11.0 min |

A 15-knot vessel runs **13.9 km in 30 minutes** — most of the 19.4 km swath. So:

* **Cross-platform cueing is on a stopwatch.** Another operator's SAR pass, downlinked, correlated,
  turned into a tasking order and uplinked, rarely closes inside 15 minutes. For slow or stationary
  targets it works fine; for a vessel actually running from something, it frequently will not.
* **A cue on the same bus has zero age.** This is the argument for a **passive RF payload beside the
  imager** rather than a bigger imager: the binding constraint is cue *latency*, not pixels.

A co-located RF receiver would also close the loop our applet currently has to leave open. Today we
detect a hull and find no AIS, and we call that a dark vessel. A vessel in ice cannot run without its
navigation radar — it would hit something. So an X-band (9.3–9.5 GHz) or S-band navigation-radar
intercept on the same spacecraft converts our strongest inference into a direct measurement:

> *optical hull, no AIS, but an active navigation radar at the same position* — that is not an
> ambiguous contact any more.

**Ranked for a next-mission payload trade:**

| Addition | What it buys | Honest cost |
| :-- | :-- | :-- |
| **Passive RF (marine radar X/S-band)** | Zero-latency onboard cue; confirms "under way and not broadcasting"; works in polar night and through cloud | Antenna accommodation and a second digitiser chain; regulatory/ITAR care on intercept |
| **SWIR extension past 1600 nm** | Restores NDSI, the standard ice index we currently have to work around; hydrocarbon/bilge discharge detection | Different detector technology; cooling; cost |
| **Thermal IR** | Sees hulls in polar night; engine-heat signature separates under-way from drifting | Bigger break from the current VNIR bus |
| Higher GSD | Marginal — 4.75 m already resolves hull geometry and wake | Does not touch the latency constraint at all |

Our ranking puts RF first, and that is a conclusion the swath geometry forces rather than a preference.

---

## 7. What this system cannot do

* **Polar night.** HyperScape100 is 460–860 nm. For months of the Arctic winter there is no signal to
  process. Nothing in our applet changes that; we are a lit-season and cued-confirmation asset, and an
  Arctic capability claim that ignores this is not honest.
* **Cloud and fog.** VNIR is blinded by low stratus and advection fog, which are common in the melt
  season. Our screener detects and reports the condition rather than guessing through it.
* **No SAR, no RF, today.** Everything above is optical-only. Cross-sensor fusion with Sentinel-1 was
  considered and rejected for *moving* targets on the geometry in §6.
* **No hardware validation.** No Jetson was ever available. TensorRT, DLA, GPU/CPU memory contention,
  real power draw and real thermal behaviour are all unvalidated.
* **Arctic recall is unmeasured** (§5), and real-data validation is at 10 m Sentinel-2, not 4.75 m.
* **Ships alongside a quay, or within 200 m of land, are not reported** — by design, as port clutter.

---

## 8. What we would build next, given access

Roughly in order of how much we think each is worth:

1. **Wavelength-aware band contract.** Replace the hardcoded R/G/B/NIR with an N-band input carrying a
   wavelength manifest, so the applet consumes whatever 32 bands are loaded that pass. This is the
   direct enabler for everything else and is mostly refactoring.
2. **Band-selection study.** With hyperspectral training data (PRISMA, EnMAP, DESIS), compute the band
   subset that maximises hull/ice/water separability and hand Galaxia a recommended 32-band
   configuration for Arctic maritime passes — plus a per-band dTDI allocation for low-sun SNR.
3. **Validate the thermal model.** One hour on a real Orin NX, or one conversation with whoever owns
   the Onyx Edge thermal budget, replaces our biggest assumption with a measurement.
4. **RF-cued mode.** Accept an onboard cue as a pipeline input — a position, a time and an uncertainty
   ellipse — and use it to prioritise both the search and the downlink queue.

---

## Sources

| Ref | Link |
| :-- | :-- |
| [mobius] | [Galaxia — MÖBIUS-1 launch release, 23 Jun 2025 (newswire.ca)](https://www.newswire.ca/news-releases/galaxia-takes-another-leap-forward-in-canadian-space-exploration-with-the-launch-of-mobius-1-832435468.html) |
| [onyx] | [SpaceQ — Galaxia unveils Onyx Edge, 24 Jul 2026](https://spaceq.ca/galaxia-unveils-onyx-edge-compute-system-for-low-earth-orbit-intelligence/) |
| [hyper-buy] | [SpaceQ — Galaxia acquires hyperspectral imager from Simera Sense, 3 Sep 2026](https://spaceq.ca/galaxia-acquires-hyperspectral-imager-from-simera-sense/) |
| [hyper-sn] | [SpaceNews — Galaxia takes next step in Earth observation](https://spacenews.com/galaxia-takes-next-step-in-earth-observation-with-purchase-of-simera-sense-hyperspectral-imager/) |
| [ctos] | [SpaceQ — Galaxia lands $2.5M DRDC contract for a tactical LEO satellite](https://spaceq.ca/galaxia-lands-2-5m-drdc-contract-to-build-tactical-leo-satellite/) |
| [simera] | [Simera Sense — HyperScape100 product page](https://simera-sense.com/products/hyperscape100/) |

[mobius]: https://www.newswire.ca/news-releases/galaxia-takes-another-leap-forward-in-canadian-space-exploration-with-the-launch-of-mobius-1-832435468.html
[onyx]: https://spaceq.ca/galaxia-unveils-onyx-edge-compute-system-for-low-earth-orbit-intelligence/
[hyper-buy]: https://spaceq.ca/galaxia-acquires-hyperspectral-imager-from-simera-sense/
[hyper-sn]: https://spacenews.com/galaxia-takes-next-step-in-earth-observation-with-purchase-of-simera-sense-hyperspectral-imager/
[ctos]: https://spaceq.ca/galaxia-lands-2-5m-drdc-contract-to-build-tactical-leo-satellite/
[simera]: https://simera-sense.com/products/hyperscape100/
