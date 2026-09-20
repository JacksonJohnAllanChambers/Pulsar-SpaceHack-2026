# Power and Resource Validation

## Executive Summary

**Status: conditionally meets the project requirements.**

The applet stays well within the repository's measured CPU and memory envelope. Its thermal governor also behaves correctly in the project's orbital model. Actual Jetson power draw and spacecraft thermal performance remain unvalidated because no Jetson hardware was available.

| Area | Status | Evidence |
|---|---|---|
| Memory | **PASS [M]** | 940 MB peak, 6.56% of the 14 GB limit |
| CPU | **PASS [M]** | 2.65 average cores busy out of 6 |
| Runtime | **MEASURED [M]** | 7.901 s warm mean on Windows AMD64; not a Jetson timing claim |
| Thermal behavior | **PASS [Mo]** | Modeled peak below 95 C in both simulated orbit cases |
| Electrical power | **UNVALIDATED [U]** | Watts are inferred by the thermal model, not measured on hardware |
| Power-control tests | **PASS [M]** | 39 focused tests passed |

## Requirements Being Checked

The repository defines these runtime constraints:

- **Memory:** 14 GB, with swap also capped at 14 GB
- **CPU:** 6 cores
- **Runtime:** no internet access inside the container
- **Thermal target:** the governor acts before the modeled 95 C threshold
- **Power profiles:** 25 W, 15 W, and 10 W modeled Orin operating envelopes

The track guide does not specify a standalone electrical wattage pass/fail limit. The project's power claim is therefore a combination of measured resource usage and a clearly labeled thermal-power model.

## Measured Resource Benchmark [M]

**Command:**

```powershell
.venv\Scripts\python.exe scripts\benchmark.py --input data\eval_bundle --iterations 3
```

**Execution environment:** Windows AMD64, project virtual environment, `data/eval_bundle`.

| Measurement | Result | Requirement or interpretation |
|---|---:|---|
| Cold-start runtime | 9.747 s | Measured locally; not a Jetson claim |
| Warm runtime mean | 7.901 s | Measured locally; not a Jetson claim |
| Warm runtime standard deviation | 0.135 s | Stable across the three runs |
| Peak RSS | 940 MB | 6.56% of the 14 GB budget |
| Average CPU use | 2.65 cores | 44.2% of the 6-core budget |
| Raw sensor input | 180.0 MB | Input workload |
| Downlink output | 17.88 KB | Approximately 10,311x reduction |
| Output size stability | `True` | Same output size across runs |

The benchmark demonstrates substantial memory and CPU headroom. It does **not** establish Jetson wall-clock performance because it was run on Windows AMD64 rather than inside the ARM64 emulated container or on Jetson hardware.

## Thermal and Power Simulation [Mo]

**Command:**

```powershell
.venv\Scripts\python.exe scripts\orbit_pass_sim.py --input data\eval_bundle --repeats 5
```

The model estimates sustainable SoC power for the assumed spacecraft thermal design:

| Orbital condition | Sustainable modeled SoC power |
|---|---:|
| Eclipse | 34.63 W |
| Sunlight | 18.89 W |

The cascade profiles are configured as follows:

| Profile | Modeled power envelope | Behavior |
|---|---:|---|
| `FULL` | 25 W | Full detection, verification, wake analysis, and evidence chips |
| `REDUCED` | 15 W | Keeps detection and verification; removes evidence chips |
| `SURVEY` | 15 W | Removes wake analysis and reduces concurrency |
| `BEACON` | 10 W | Physics-only fallback with a strict candidate cap |

Continuous-processing simulation results:

| Orbit case | Peak modeled junction temperature | Final profile | Result |
|---|---:|---|---|
| Mid-beta SSO, 35% eclipse | 83.06 C | `FULL` after governor transitions | Below the 95 C action threshold |
| Dawn-dusk SSO, 0% eclipse | 70.42 C | `REDUCED` | Below the 95 C action threshold |

These temperatures and wattage values are **model outputs**, not measurements. The model uses assumed spacecraft values such as radiator area, emissivity, parasitic load, and thermal resistance. Those assumptions must be replaced with mission-specific values before making a flight-readiness claim.

## Focused Automated Tests [M]

**Command:**

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_governor.py tests\test_pipeline.py tests\test_telemetry.py
```

**Result:** `39 passed in 14.33s`

These tests cover the relevant control and evidence paths, including:

- Thermal governor profile selection and transitions
- Telemetry collection for CPU, memory, timing, and I/O
- Deterministic pipeline output
- Pipeline behavior under the configured cascade profiles

## What This Proves

The current evidence supports these statements:

1. The applet fits comfortably within the repository's 14 GB memory and 6-core CPU envelope in the measured local run.
2. The governor changes the selected cascade profile according to the modeled power and thermal budget.
3. The simulated operating profiles remain below the modeled 95 C action threshold.
4. The power-control and telemetry code paths pass the focused automated tests.

## What This Does Not Prove

The current evidence does not prove:

- Actual watts consumed by a Jetson Orin NX
- Actual Jetson junction temperature under sustained load
- Actual spacecraft bus thermal performance
- Native ARM64 or Jetson runtime latency
- That the assumed radiator, parasitic-load, and thermal-resistance values match the final spacecraft

The defensible presentation is therefore:

> The applet meets the measured CPU and memory constraints and has a modeled thermal governor that stays within its assumed power envelope. Electrical power and flight thermal compliance are not yet hardware-validated.

## Recommended Final Validation

For a hardware-backed claim, repeat the workload on the target Jetson or an equivalent ARM64 test system while recording:

- Input voltage and current at the board or carrier power rail
- Jetson power mode and clocks
- Junction temperature over the full workload
- Peak and sustained CPU utilization
- Wall-clock runtime under the same input bundle

Until those measurements exist, keep the `[Mo]` and `[U]` labels attached to the thermal and power claims.

## Source Files

- [Track and rubric requirements](RULES_AND_RUBRIC.md)
- [Rubric evidence sheet](RUBRIC_SPEC_SHEET.md)
- [Docker emulation constraints](DOCKER_EMULATION_GUIDE.md)
- [Telemetry implementation](../applet/core/telemetry.py)
- [Thermal model](../applet/core/thermal.py)
- [Power governor](../applet/core/governor.py)
- [Benchmark harness](../scripts/benchmark.py)
- [Orbit and governor simulation](../scripts/orbit_pass_sim.py)
