# 📋 Pulsar SpaceHack 2026: Official Rules & Judging Rubric

This document synthesizes the official competition guidelines, judging criteria weights, and technical constraints for **Pulsar SpaceHack 2026**.

---

## 1. Judging Rubric & Weights

| Category | Weight | Key Evaluation Criteria | How This Scaffold Scores Top Marks |
| :--- | :--- | :--- | :--- |
| **Theme & Edge AI Relevance** | **25%** | - Meaningfully leverages NVIDIA Jetson capabilities, edge computing, embedded AI, CV, or sensor processing.<br>- Does RAM usage fall within allowed device limits? CPU? Thermal load?<br>- Does it take an acceptable time to run? | - Cascade sized for the device: vectorised CV screens megapixels, a 59 KB INT8 CNN sees only candidate chips (0.2 ms each).<br>- Measured, not asserted: `edge_telemetry.json` records per-stage latency, peak RSS (1.4 GB for a full 4096x4096 swath = 10 % of the 14 GB envelope) and cores busy.<br>- ONNX Runtime: CPU provider in the emulated container, TensorRT/CUDA providers picked up automatically on hardware.<br>- **Temperature is in this criterion and we answer it.** A two-node orbital thermal model (`applet/core/thermal.py`) predicts junction temperature from measured CPU utilisation against NVIDIA's published 10/15/25 W nvpmodel envelopes, and an `EdgeGovernor` re-profiles the cascade from it. Modelled, badged `MODELLED` everywhere it appears, and every bundle carries `validated_on_hardware: false`. |
| **Technical Implementation** | **30%** | - Quality and completeness of implementation.<br>- Are core features functional? Is architecture sound?<br>- Code quality, integration, performance, and technical difficulty. | - Five-stage pipeline behind one `run_pass()` used by CLI, GUI, benchmark and tests.<br>- 150 tests covering corrupt inputs, missing bands, physics accuracy against rendered truth, sea-ice regime behaviour, thermal/control-law regressions, and byte-identical output.<br>- Scored on held-out *real* Sentinel-2 (SEN2MS): 0.82 open-water recall, 4 deg median heading error vs AIS; plus held-out synthetic scenes for the AIS logic. |
| **Innovation & Problem Solving** | **20%** | - Originality of approach and effectiveness in solving the problem.<br>- Novel technique, workflow improvement, model optimization, or unique UX. | - Wake ray transform instead of polygon fitting; speed from the transverse-wave dispersion relation with a swell-cancelling reference ray; object-level CFAR for whitecap fields.<br>- Verifier trained on chips *mined from the detector's own candidates*.<br>- FP32 -> INT8: 3.1x smaller, faster, same AUC; retrained on real Sentinel-2 chips after the synthetic-only model was caught rejecting real ships.<br>- **Thermal-aware cascade depth.** The applet treats the Orin's power envelope as an input, not a limit it discovers by throttling: it looks ahead (the bus time constant is ~70 min, the orbit ~95 min, so reacting to the die is far too late) and degrades on one invariant -- *degrade the evidence, never the alert*. Measured on 40 synthetic tuning scenes at 4.75 m, seed 777 (`data/eval_bundle`), not on real imagery: the first rung is free (46/46 AIS-confirmed ships and all 18 dark-vessel alerts kept, downlink 18.0 -> 7.3 KB). |
| **Impact & Practical Value** | **15%** | - Potential real-world usefulness and business or operational value.<br>- Clear customer, application, or measurable outcome. | - Fits Galaxia's DRDC CTOS contract (Sept 2025), a nanosatellite for validating onboard edge compute -- the class of payload software CTOS exists to prove. (CTOS is not advertised as a maritime mission; we do not claim it is.)<br>- 182 MB of raw swath -> 14 KB of ranked intelligence (13 000x); anomalies get chips, known traffic does not. |
| **Presentation & Demonstration** | **10%** | - Clarity of pitch, demo effectiveness, communication of technical details, ability to answer questions. | - Offline ground console showing every stage, with live threshold sliders and an accuracy panel.<br>- README states the limits plainly (no Jetson was ever available; the thermal half is a model and says so).<br>- The console's governor panel shows the orbit trace, the ladder, and what each rung costs -- with the `MODELLED` badge on the same screen as the number. |

---

## 2. Strict Competition Rules & Constraints

1. **Emulated Container Environment**:
   * Must run inside an emulated `linux/arm64` Docker container matching the NVIDIA Jetson Orin NX environment (Ubuntu 22.04 LTS).
   * Resource limits must be strictly respected:
     ```bash
     --memory=14g --memory-swap=14g --cpus=6
     ```
2. **Zero Internet at Runtime**:
   * The applet must operate with **no internet access** during execution (`--network none`).
   * All models, weights, libraries, and lookup tables must be pre-packaged in the container or input bundle.
3. **Defined Input Bundle**:
   * Must ingest a self-contained bundle (e.g. imagery files + `manifest.json` + uplinked parameters/catalogs).
4. **Defined Output Artifact**:
   * Must generate a concrete downlink artifact (e.g. compressed tactical bundle containing GeoJSON, summary telemetry, and micro-thumbnails).
   * The output artifact must be **significantly smaller than raw imagery** to justify onboard compute.
5. **Resilience & Determinism**:
   * The applet **must not crash** on corrupted images, missing spectral bands, or sensor noise.
   * Results must be deterministic across repeated runs on identical inputs.
