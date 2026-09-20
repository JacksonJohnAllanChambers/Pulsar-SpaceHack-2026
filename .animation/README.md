# Arctic Ship Reveal

This workspace produces a 10-second realistic-style cinematic animation. It begins in a near-vertical Arctic Ocean view where a low-contrast vessel blends into textured fragmented ice, then the camera descends and pitches to a foggy landscape view that clearly reveals the same ship.

Everything in this directory is ignored by Git, including the rendered video.

## Files

- `render_arctic_ship_reveal.py`: texture-driven procedural renderer and animation settings.
- `arctic_ship_reveal.mp4`: generated output; created after rendering.

## Prerequisites

- Python 3.10 or later.
- OpenCV with video-writing support. `opencv-python-headless` is sufficient and is already declared in the repository's `requirements.txt`.

## Install

From the repository root, create and activate a virtual environment, then install the renderer dependency:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the agent needs only the animation renderer rather than the complete project environment:

```powershell
python -m pip install "opencv-python-headless>=4.7.0.72"
```

On Linux or macOS, use `python3 -m venv .venv`, then activate with `source .venv/bin/activate`.

## Configure

Set these constants near the top of `render_arctic_ship_reveal.py` before rendering:

```python
WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SECONDS = 10
OUTPUT = Path(__file__).with_name("arctic_ship_reveal.mp4")
```

The default configuration produces a 300-frame, 16:9 MP4. To make a longer or higher-resolution version, adjust these constants. `FRAME_COUNT` is calculated automatically from the frame rate and duration.

## Render

Run this from the repository root:

```powershell
python .animation/render_arctic_ship_reveal.py
```

Expected output:

```text
Wrote ...\.animation\arctic_ship_reveal.mp4 (10s, 1280x720, 30 fps)
```

## Verify

Open the MP4 and inspect the start, midpoint, and end:

1. Start: a near-overhead field of fragmented ice; the ship is difficult to distinguish from ice.
2. Midpoint: camera motion transitions continuously from top-down to landscape.
3. End: a low landscape composition with a clearly visible ship, hull, superstructure, and wake.

When FFmpeg is available, confirm media metadata with:

```powershell
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate -of default=noprint_wrappers=1 .animation\arctic_ship_reveal.mp4
```

The expected duration is approximately 10 seconds, with one 1280x720 video stream at 30 fps.

## Troubleshooting

- `ModuleNotFoundError: No module named 'cv2'`: activate the virtual environment and install `opencv-python-headless`.
- `OpenCV could not open an MP4 writer`: install an OpenCV build with FFmpeg support, or install system FFmpeg and rerun from an environment whose OpenCV can access it.
- `py -3 -m venv` fails: install a full CPython distribution with the `venv` module, then repeat the setup steps.