# 🛰️ Pulsar SpaceHack 2026: Tracks & Dataset Strategy Guide

This guide details the 4 competition tracks and maps out datasets for the reference payload **we chose** (see below). The organisers' own Track Guide specifies no sensor and no dataset -- it says explicitly that teams source their own inputs -- so everything in section 1 is our assumption, not a requirement.

---

## 1. Our chosen reference payload

> **Provenance, stated plainly.** The organisers' `Hackathon_Project_Track_Guide.pdf` names no sensor, no
> spectral range, no GSD and no company. It says: *"we are not handing out a standard dataset or a standard
> baseline model. Part of the challenge is going and finding (or building) the inputs you need yourselves."*
> Track 1 is a free-for-all, so the payload below is **a design assumption we made and can defend**, not a
> spec we were issued. We picked it for a specific, checkable reason: **Galaxia announced buying a
> HyperScape100 on 3 September 2026**, sixteen days before this event, naming maritime surveillance as a
> target application ([SpaceQ](https://spaceq.ca/galaxia-acquires-hyperspectral-imager-from-simera-sense/)).
> Every figure below is from Simera Sense's published datasheet. See `docs/GALAXIA_ALIGNMENT.md`.

Design against a Simera Sense HyperScape100 (all figures per the manufacturer's datasheet):
* **Sensor:** Simera Sense HyperScape100
* **Spatial Resolution (GSD):** **4.75 meters / pixel** (at 500 km LEO orbit)
* **Spectral Coverage:** **460 nm – 860 nm (VNIR - Visible & Near-Infrared)**
* **Swath Width:** 19.4 km (4096 px across)
* **Downlink Bottleneck:** Sensor captures up to 1 Terabit/day, but ground passes last only 5–12 minutes at ~50 Mbps.

---

## 2. Track Breakdown & Strategy

### Track 1: Free-for-All (Recommended: Tactical Maritime ISR)
* **Concept:** *Tactical Edge Sentinel — Dark Vessel & Wake Detection with AIS Correlation*.
* **Why it Wins:** Galaxia Mission Systems won a **$2.5M DRDC contract (Sept 2025)** to build the Canadian Tactical Operations Satellite (CTOS), a nanosatellite whose stated purpose is validating *data relay and edge-computing* technologies in orbit. CTOS is not advertised as a maritime mission, so we do not claim it is one -- what we claim is that an onboard applet which turns a swath into a 7 KB decision is precisely the class of payload software CTOS exists to prove, and Canada has 243,000 km of coastline where that matters.
* **Onboard Value:** Reduces a 400 MB raw scene to a 25 KB tactical GeoJSON alert with 64x64 cropped target chips, delivered during a single pass instead of hours later.

### Track 2: ML Optimization
* **Concept:** Take an existing satellite model (e.g. YOLOv8-nano for ship detection or EuroSAT ResNet-18) and optimize for Jetson Orin NX.
* **Key Techniques:** FP32 $\to$ FP16 / INT8 quantization, ONNX Runtime conversion, structured channel pruning, operator fusion.
* **Evaluation:** Document clear "Before vs. After" tables (Model size: 45 MB $\to$ 8 MB, Latency: 320 ms $\to$ 35 ms, Memory: 1.2 GB $\to$ 310 MB).

### Track 3: Downlink Prioritization
* **Concept:** Intelligent batch reviewer that scores and prioritizes captured scenes under a strict RF byte budget.
* **Key Techniques:** Multi-factor scoring (usable cloud-free percentage $\times$ target density $\times$ military/commercial importance), dynamically filling the pass budget.

### Track 4: Image Quality and Usability Screening
* **Concept:** Fast front-line screening applet detecting cloud cover, haze, motion blur, saturation, or corrupted bands.
* **Key Techniques:** Multi-spectral whiteness tests, Laplacian variance sharpness metrics, no-data border detection.

---

## 3. Recommended Open Datasets & Proxies

| Dataset / Source | Resolution (GSD) | Bands | Best Fit | Access Link |
| :--- | :--- | :--- | :--- | :--- |
| **Sentinel-2 VNIR (Bands 2,3,4,8)** | 10 m (RGB/NIR) | 4–12 bands (440–865 nm) | 100% free open proxy for VNIR satellite data | [Copernicus Data Space](https://dataspace.copernicus.eu/) |
| **PlanetScope / RapidEye** | 3.0 m – 5.0 m | 4–8 VNIR bands | **Exact 1:1 physical GSD match** to our 4.75 m reference imager | [Planet Open California](https://www.planet.com/open-california/) |
| **xView Dataset** | ~0.3 m (downsample 4x) | Optical RGB | Defense & maritime target classes (ships, cargo, airstrips) | [xView DoD Challenge](http://dx.doi.org/10.21227/s7gd-1e82) |
| **Airbus Ship Detection** | ~1.5 m – 3.0 m | RGB | Maritime Domain Awareness & ship masks | Available on Kaggle |
| **CloudSEN12+** | 10 m – 60 m | Sentinel-2 | Gold standard for cloud and haze screening | [CloudSEN12+ on HuggingFace](https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus) |
| **Synthetic Mock Bundle** | 4.75 m | 4-band (RGB+NIR) | Included in this repo (`scripts/generate_synthetic_data.py`) | Ready to run locally! |
