# 📋 Pulsar SpaceHack 2026: Official Rules & Judging Rubric

This document synthesizes the official competition guidelines, judging criteria weights, and technical constraints for **Pulsar SpaceHack 2026**.

---

## 1. Judging Rubric & Weights

| Category | Weight | Key Evaluation Criteria | How This Scaffold Scores Top Marks |
| :--- | :--- | :--- | :--- |
| **Theme & Edge AI Relevance** | **25%** | - Meaningfully leverages NVIDIA Jetson capabilities, edge computing, embedded AI, CV, or sensor processing.<br>- Does RAM usage fall within allowed device limits? CPU? Thermal load?<br>- Does it take an acceptable time to run? | - Cascade sized for the device: vectorised CV screens megapixels, a 59 KB INT8 CNN sees only candidate chips (0.2 ms each).<br>- Measured, not asserted: `edge_telemetry.json` records per-stage latency, peak RSS (1.4 GB for a full 4096x4096 swath = 10 % of the 14 GB envelope) and cores busy.<br>- ONNX Runtime: CPU provider in the emulated container, TensorRT/CUDA providers picked up automatically on hardware. |
| **Technical Implementation** | **30%** | - Quality and completeness of implementation.<br>- Are core features functional? Is architecture sound?<br>- Code quality, integration, performance, and technical difficulty. | - Five-stage pipeline behind one `run_pass()` used by CLI, GUI, benchmark and tests.<br>- 31 tests covering corrupt inputs, missing bands, physics accuracy against rendered truth, and byte-identical output.<br>- `scripts/evaluate.py` reports precision / recall / heading / AIS-class accuracy on held-out scenes. |
| **Innovation & Problem Solving** | **20%** | - Originality of approach and effectiveness in solving the problem.<br>- Novel technique, workflow improvement, model optimization, or unique UX. | - Wake ray transform instead of polygon fitting; speed from the transverse-wave dispersion relation with a swell-cancelling reference ray; object-level CFAR for whitecap fields.<br>- Verifier trained on chips *mined from the detector's own candidates*.<br>- FP32 -> INT8: 3.1x smaller, ~5x faster, same AUC. |
| **Impact & Practical Value** | **15%** | - Potential real-world usefulness and business or operational value.<br>- Clear customer, application, or measurable outcome. | - Targets Galaxia's DRDC CTOS maritime-domain-awareness mission.<br>- 182 MB of raw swath -> 14 KB of ranked intelligence (13 000x); anomalies get chips, known traffic does not. |
| **Presentation & Demonstration** | **10%** | - Clarity of pitch, demo effectiveness, communication of technical details, ability to answer questions. | - Offline ground console showing every stage, with live threshold sliders and an accuracy panel.<br>- README states the limits plainly (synthetic-only accuracy, no hardware validation). |

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
