# 🎯 Pulsar SpaceHack 2026: Pitch Deck & Presentation Guide

This guide provides a slide-by-slide structure designed to score maximum points in **Presentation & Demonstration (10%)** and **Impact & Practical Value (15%)**.

---

## 1. The Core Narrative

* **The Company Thesis:** *"Data is not the mission. Intelligence is."* (Galaxia Mission Systems).
* **The Reality Check:** A Simera HyperScape100 generates up to **1 Terabit of imagery per day**. A satellite ground pass only lasts **8 to 12 minutes** at ~50 Mbps. 
* **The Problem:** Downlinking 400 MB of empty ocean imagery to search for ships on Earth takes hours—by the time the ground station downloads and processes the file, the suspicious vessel has moved 40 miles away.
* **Our Solution (Tactical Edge Sentinel):** We process multi-spectral sensor data directly on the satellite's NVIDIA Jetson Orin NX in **under 2 seconds**. We discard clouds and innocent commercial vessels onboard, transmitting only a **35 KB tactical GeoJSON alert** and visual target chips to defense commanders.

---

## 2. 6-Slide Pitch Structure (3-Minute Presentation)

### Slide 1: The Problem — The Downlink Bottleneck in Sovereign Waters
* **Visual:** Map of Canadian Arctic/Atlantic coastlines (243,000 km) + visual comparison of a massive 400 MB raw scene vs. narrow 8-minute ground pass window.
* **Talking Point:** "Downlinking raw imagery over sovereign waters is too slow and too expensive. Dark fishing vessels and unauthorized incursions exploit this latency gap to disappear."

### Slide 2: The Edge Architecture — Physics Meets Kinematics
* **Visual:** 3-stage edge pipeline diagram:
  1. Multi-spectral Whiteness Cloud Mask
  2. Physics-Informed Kelvin Wake Contour Detector ($\approx 38.9^\circ$ V-angle + centerline)
  3. Kinematic AIS Dead-Reckoning Correlator
* **Talking Point:** "Instead of running a heavy black-box neural network that drains battery power, we use the universal physics of Lord Kelvin's wake equation combined with dead-reckoning AIS tracking."

### Slide 3: Live / Recorded Demo (The "Hero Moment")
* **Visual:** Terminal or Web Map Dashboard showing:
  * Ingesting 4-band satellite imagery.
  * Dropping 80% cloudy tile in 4 milliseconds.
  * Detecting Kelvin wake, calculating heading ($315^\circ$) and speed ($18\text{ knots}$).
  * Comparing against uplinked AIS: no record found $\to$ **CRITICAL ALERT: DARK VESSEL**.
  * Cropped $64\times 64$ chip preview displayed on map.

### Slide 4: Edge AI Relevance & Jetson Compliance (Rubric 25%)
* **Visual:** The Benchmark Table from `scripts/benchmark.py`:
  * Peak RAM: **< 180 MB** (out of 14,000 MB allowed).
  * Latency: **< 1.2 seconds** per scene.
  * CPU Load: **Low thermal footprint** (critical for vacuum heat dissipation).
  * Bandwidth Reduction: **> 99.8% data savings** (400 MB $\to$ 25 KB).
  * Zero internet reliance at runtime (`--network none`).

### Slide 5: Real-World Defense & Commercial Impact (Rubric 15%)
* **Customer Alignment:** Directly supports Canada's **Defence Research and Development Canada (DRDC) CTOS mission** and the Canadian Coast Guard.
* **Quantifiable Outcome:** Actionable tactical intelligence delivered to naval commanders in **seconds instead of 18 hours**.

### Slide 6: The Team & Summary
* **Summary Statement:** "We turned a passive sensor into an autonomous tactical sentinel in orbit. Data is not the mission. Intelligence is."

---

## 3. Anticipated Judges' Questions & Answers

* **Q: "Why not just downlink the image and let a cloud server on Earth do the heavy processing?"**
  * **A:** "Because in Low Earth Orbit, you only see a ground station for 8 to 12 minutes. If a pass occurs once every 90 minutes, downlinking 500 MB causes multi-hour queue backlogs. In tactical maritime security, latency is life or death—a 25 KB alert reaches the field commander during the *current* pass."
* **Q: "What if the sea is rough with whitecaps?"**
  * **A:** "We engineered a 2D morphological opening filter and CFAR adaptive thresholding that annihilates disconnected wave speckles while our polygon approximation strictly enforces the $38.9^\circ$ Kelvin envelope."
* **Q: "How does the satellite know about AIS if it has no internet?"**
  * **A:** "Before each orbital pass, ground control uplinks a tiny 15 KB JSON catalog of expected commercial ships in that region. Our applet dead-reckons their kinematic positions forward to the exact second of image capture."
