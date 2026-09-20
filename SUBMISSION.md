# Submission -- Tactical Edge Sentinel

Pulsar SpaceHack 2026 · Track 1 (free-for-all) · onboard dark-vessel detection for a small VNIR satellite.

**Team:** Jackson Chambers, Megan Neville, Ryan (ryan6625).

## Read these, in this order

| | |
| :-- | :-- |
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | how it works, one page per idea |
| [`docs/RUBRIC_SPEC_SHEET.md`](docs/RUBRIC_SPEC_SHEET.md) | every rubric question, our answer, whether it is measured / modelled / unvalidated, and the command that reproduces it |
| [`README.md`](README.md) | the full results, including the ones that went against us |
| [`docs/PITCH_AND_DEMO.md`](docs/PITCH_AND_DEMO.md) | the three-minute pitch, the demo path, and the questions we expect |
| [`docs/GALAXIA_ALIGNMENT.md`](docs/GALAXIA_ALIGNMENT.md) | who would use it, with sources and dates |

## Run it

In the judging environment (linux/arm64, 14 GB, 6 CPUs, **no network**, no GPU):

```bash
./scripts/run_emulated.sh          # Windows: .\scripts\run_emulated.ps1
```

That builds `docker/Dockerfile.arm64`, renders the sample bundle inside the image, and runs the applet
with `--memory=14g --memory-swap=14g --cpus=6 --network none`. Output lands in `data/outputs/`. CI does
the same build and run under QEMU on every push (`.github/workflows/tests.yml`, job `image`).

Without Docker, any Python 3.9+:

```bash
pip install -r requirements.txt
python scripts/setup_data.py --synthetic            # no network
python -m applet run -i data/sample_bundle -o data/outputs
pip install -r requirements-dev.txt && python -m pytest -q
python -m ground.server                             # console, http://127.0.0.1:8050
```

## The baseline rules

| Rule | How it is met | Evidence |
| :-- | :-- | :-- |
| Runs inside the emulated container | `docker/Dockerfile.arm64`, versions pinned in `docker/constraints.txt`, every dependency an aarch64 wheel | CI job `image`; `tests/test_flight_image.py` stages the image's COPY set and runs a pass from it |
| No internet at runtime | the applet opens no sockets; the container runs with `--network none`; the console uses no CDN | `scripts/run_emulated.sh` |
| Consumes a defined input bundle | `manifest.json` + `ais_catalog.json` + optional `known_structures.json` + 4-band rasters | README "Input bundle"; `applet/core/validator.py` |
| Produces a clear output artifact | `downlink_<pass>.tar.gz`: GeoJSON of classified contacts with a reason each, scene report, priority-ordered chips -- ~10 KB against a 182 MB swath | README "Output" |

And the Track 1 question -- *why onboard rather than on the ground?* -- a 15-knot vessel moves 14 km
in the half hour a downlink-process-uplink loop costs, against a 19.4 km swath; and the product is
1.9 seconds of one ground pass per day instead of 10x the daily downlink capacity.

## What is and is not in the repo

* **In:** all code, the 59 KB flight model and its FP32 fallback, the team's hand verdicts
  (`data/labels/`), the contact sheets they were made on (`review/`), tests, container, docs.
* **Not in:** imagery and AIS (~1.5 GB). `python scripts/setup_data.py --all` rebuilds every dataset
  from public sources with no account or API key; licences are in the spec sheet's provenance table.
  The synthetic bundles need no network at all.

## Build the archive

```bash
python scripts/package_submission.py
```

Archives the commit with `git archive`, unpacks it somewhere empty, and from that copy alone renders the
sample bundle, runs the applet twice and checks the two downlink tarballs are byte-identical. The
archive and `SUBMISSION_MANIFEST.json` (commit, SHA-256s, measured run) land in `dist/`.

## Before we hand it in

- [ ] `Jack` merged into `main` (the default branch is what a judge clones)
- [ ] CI green on the submitted commit, all three jobs
- [ ] `python scripts/package_submission.py` passes on that commit
- [ ] Slide deck built from `docs/PITCH_AND_DEMO.md`
- [ ] Backup recording of the six-click demo
- [ ] Demo rehearsed once on the laptop that will present it, offline
