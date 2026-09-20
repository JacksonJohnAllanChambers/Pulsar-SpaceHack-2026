# Setup — install, datasets, and reproducing every number

For judges and collaborators alike: this is the path from a clean clone to a running applet, and from there
to each figure quoted in the README.

Nothing large is in git. The code, the two flight models (59 KB + 185 KB) and the docs are; every dataset is
rebuilt or downloaded by one script. Python 3.10 or 3.11.

## 1. Clone and install

```bash
git clone <repo-url> && cd Pulsar-SpaceHack-2026
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows PowerShell use `.venv\Scripts\Activate.ps1` instead of the line above.

```bash
pip install -r requirements-dev.txt
```

`requirements.txt` is the flight image only (what goes in the container). `requirements-dev.txt` adds tests, the
GUI and the data tools. `requirements-train.txt` adds PyTorch + ONNX and is only needed to retrain the verifier.

## 2. Get the data

```bash
python scripts/setup_data.py
```

That renders the three synthetic bundles (about a minute, no downloads) and is enough for the tests, the CLI and
the GUI. For the real imagery:

```bash
python scripts/setup_data.py --all
```

| Step | Output | Download | Source / licence |
| :-- | :-- | --: | :-- |
| `--synthetic` | `data/sample_bundle`, `data/eval_bundle` (seed 777, tuning), `data/heldout_bundle` (seed 4242, test) | none | rendered locally, seeded |
| `--sentinel2` | `data/real/s2_bundle` - Gibraltar, Suez, Long Beach, Halifax, Dover | ~95 MB | Sentinel-2 L2A via Element84 Earth Search; Copernicus open licence |
| `--sen2ms` | `data/real/sen2ms/` zip + `ais_bundle` (real chips with their real AIS) | 565 MB | Zenodo 15571607, CC-BY-4.0 |
| `--noaa` | `data/real/s2_ais_bundle` - Long Beach 2024-11-08 + that day's AIS | ~400 MB | NOAA / BOEM Marine Cadastre, public domain |

Steps are independent, resumable and skipped when their output exists (`--force` redoes one). No account or API
key is needed for any of them. Disk: about 1.5 GB for everything.

**Short on space on your system drive?** Put the big folders elsewhere; they are linked back so all paths still work:

```bash
python scripts/setup_data.py --all --data-root D:\spacehack-data
```

(`/Volumes/External/spacehack-data` or any path on macOS / Linux. Windows uses a directory junction, which needs no
admin rights.)

Notes: NOAA only serves July-December 2024 at the moment (earlier 2024 days are listed but 404), which is why the
real-AIS scene is pinned to 2024-11-08. The least-cloudy Sentinel-2 scene per area is chosen at fetch time, so
`s2_bundle` may contain newer dates than the ones quoted in the README; `s2_ais_bundle` is pinned.

## 3. Check it works

```bash
python -m pytest -q
```

```bash
python -m applet run -i data/sample_bundle -o data/outputs -c config.example.yaml
```

```bash
python -m ground.server
```

Then open http://127.0.0.1:8050, pick a bundle, press **Run pass**. Bundles to try: `data/sample_bundle`
(synthetic, has ground truth and spoofers), `data/real/s2_ais_bundle` (real scene, real AIS, real MMSIs),
`data/real/s2_bundle` (five real scenes, no AIS so everything is dark).

## 4. Reproduce the numbers in the README

```bash
python scripts/evaluate.py -i data/heldout_bundle
```

```bash
python scripts/evaluate.py -i data/heldout_bundle --no-verifier
```

```bash
python training/sen2ms.py evaluate
```

```bash
python training/sen2ms.py bundle
```

```bash
python scripts/benchmark.py --full-swath
```

Tuning discipline: change thresholds using `data/eval_bundle` and the SEN2MS *train* products only
(`crc32(product) % 4 != 0`); `data/heldout_bundle` and the SEN2MS test products are for reporting.

## 5. Retrain the verifier (optional)

```bash
pip install -r requirements-train.txt
```

```bash
python training/sen2ms.py mine
```

```bash
python training/train_verifier.py --scenes 500 --epochs 16 --real-npz data/real/sen2ms/train_chips.npz
```

Writes `applet/models/verifier_{fp32,int8}.onnx` and `model_card.json` (about 6 minutes on a laptop CPU). Re-pick
`verifier.reject_below` on tuning data afterwards, then re-run step 4.

## 6. Container (Jetson envelope)

```bash
./scripts/run_emulated.sh
```

`.\scripts\run_emulated.ps1` on Windows. The image renders its own sample bundle at build time and runs with
`--memory=14g --memory-swap=14g --cpus=6 --network none`. Take timing numbers on an ARM64 host (Apple Silicon runs
it natively); under QEMU on x86 it only proves that it fits. Versions are pinned in `docker/constraints.txt`.

Nobody on the team has Docker on an ARM64 host, so two things stand in for a local build: CI builds and starts
the image under QEMU on every push (`.github/workflows/tests.yml`, job `image`), and `tests/test_flight_image.py`
stages exactly the files the Dockerfile copies and runs a pass from them. If you add an import to flight code
from outside `applet/`, that test tells you to add a `COPY` line.

## Windows gotchas

* If `python` resolves to an MSYS / Git-for-Windows interpreter, `import cv2` fails with a DLL error. Use the
  python.org or Microsoft Store interpreter (check with `python -c "import sys; print(sys.executable)"`).
* Consoles default to cp1252; the CLI already replaces unprintable characters instead of crashing.
