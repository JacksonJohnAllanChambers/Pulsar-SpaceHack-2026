# 👥 Team Work Breakdown & Interface Contracts (5-Person Team)

This guide assigns clear ownership to avoid merge conflicts and overlapping blockers during the 2-day hackathon.

---

## 1. Role Matrix & File Ownership

| Member | Title | Key Ownership Files | Primary Objectives |
| :--- | :--- | :--- | :--- |
| **Person 1** | **Systems & Edge Architect** *(Integrator)* | `docker/Dockerfile.arm64`<br>`applet/runner.py`, `applet/cli.py`<br>`applet/core/telemetry.py`<br>`scripts/benchmark.py` | - Owns the ARM64 container run (14 GB, 6 CPUs, `--network none`) and takes the timing numbers on a native ARM64 host.<br>- Keeps `run_pass()` the single code path for CLI, GUI, tests and benchmark. |
| **Person 2** | **Detection Physics Lead** | `applet/pipelines/vessel_detector.py` | - CFAR statistics, hull/wake separation, wake ray transform, Kelvin-arm and transverse-wave measurements.<br>- Drives misses and false alarms down using `scripts/evaluate.py`, never by eye. |
| **Person 3** | **Multispectral & Data Lead** | `applet/pipelines/quality_screener.py`<br>`applet/utils/image_io.py`<br>`simulation/`, `data/` | - Water / cloud / land masks and the usability verdict.<br>- Sources real Sentinel-2 / PlanetScope scenes into bundles and labels them; this is the biggest lever left. |
| **Person 4** | **AIS, Verifier & Downlink Engineer** | `applet/pipelines/ais_correlator.py`<br>`applet/pipelines/chip_verifier.py`, `training/`<br>`applet/packaging/downlink.py` | - Dead reckoning, two-tier gating, anomaly classes.<br>- Retrains / re-quantises the chip CNN as real chips arrive; owns `model_card.json`.<br>- Byte-reproducible, budget-aware tarball. |
| **Person 5** | **Product, Demo & Pitch Lead** | `ground/` (console)<br>`docs/PITCH_AND_DEMO_TEMPLATE.md` | - Drives the live demo from the ground console (sliders, layers, funnel, accuracy panel).<br>- Pitch tied to Galaxia's DRDC CTOS contract; records a backup video of the run. |

---

## 2. Interface Contracts (How Modules Connect)

All modules communicate via a shared `context` dictionary passed through `process(context)`:

```
[Validator]
   │  context["valid_scenes"] = [ { "id": "S1", "array": float32 reflectance (H,W,4) in R,G,B,NIR order, "gsd_m", "georef", ... } ]
   │  context["ais_catalog"] = [ { "mmsi": 123, "latitude": 44.5, "cog_deg": 310, ... } ]
   ▼
[QualityScreener]
   │  context["screened_scenes"] = [ { ..., "quality_metrics": { "is_usable": True/False } } ]
   ▼
[VesselDetector]
   │  context["detected_vessels"] = [ { "apex_px": (x,y), "heading_deg": 315, "physics_score": 0.9, "chip_tensor": ..., "chip_jpeg": b"..." } ]
   ▼
[ChipVerifier]
   │  adds "verifier_prob" and fused "confidence"; drops candidates the CNN rejects
   ▼
[AISKinematicCorrelator]
   │  context["classified_targets"] = [ { ..., "classification": "DARK_VESSEL" / "CONFIRMED_KNOWN" } ]
   ▼
[DownlinkPackager]
      Outputs: `tactical_intelligence.geojson`, `chips/*.jpg`, `downlink_PASS_001.tar.gz` (<50 KB)
```

---

## 3. Two-Day Hackathon Schedule

* **Day 1 (Morning):** Setup check. All team members clone repo, run `python scripts/generate_synthetic_data.py`, and run `python -m applet run --input data/sample_bundle --output data/outputs`.
* **Day 1 (Afternoon):** Persons 2, 3, and 4 work in their dedicated pipeline files. Person 5 drafts pitch outline and demo UI.
* **Day 1 (Evening):** Person 1 merges modules and runs the first end-to-end pass inside Docker ARM64.
* **Day 2 (Morning):** Person 1 & 3 stress-test memory, latency, and bad inputs. Person 5 finalizes slides and visual map dashboard.
* **Day 2 (Afternoon):** Pitch rehearsals, record backup demo video, final submission!
