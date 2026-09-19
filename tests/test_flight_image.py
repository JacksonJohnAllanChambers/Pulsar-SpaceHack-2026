"""
The judges run docker/Dockerfile.arm64, not this checkout. Stage exactly what its COPY lines ship
and run a pass from there, so flight code that imports something the image lacks fails here
instead of as a ModuleNotFoundError in a container nobody on the team can build locally.
"""

import os
import re
import sys
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE = os.path.join(ROOT, "docker", "Dockerfile.arm64")


def _stage_image(dest):
    with open(DOCKERFILE, "r", encoding="utf-8") as f:
        copies = re.findall(r"^COPY\s+(.+?)\s+(/workspace/\S*)\s*$", f.read(), flags=re.MULTILINE)
    assert copies, "no COPY lines parsed from the Dockerfile"
    for sources, target in copies:
        for source in sources.split():
            src = os.path.join(ROOT, source)
            rel = target[len("/workspace/"):]
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dest, rel), ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, os.path.join(dest, rel, os.path.basename(source)))


def test_pass_runs_from_the_files_the_image_ships(bundle_dir, tmp_path):
    image = tmp_path / "workspace"
    image.mkdir()
    _stage_image(str(image))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    where = subprocess.run(
        [sys.executable, "-c", "import applet, applet.runner; print(applet.__file__)"],
        cwd=str(image), env=env, capture_output=True, text=True,
    )
    assert where.returncode == 0, where.stderr[-2000:]
    # Guards the guard: an editable install of the repo would make every import succeed
    assert os.path.realpath(where.stdout.strip()).startswith(os.path.realpath(str(image)))

    out = tmp_path / "out"
    run = subprocess.run(
        [sys.executable, "-m", "applet", "run", "--input", bundle_dir, "--output", str(out),
         "--config", "config.example.yaml"],
        cwd=str(image), env=env, capture_output=True, text=True,
    )
    assert run.returncode == 0, (run.stdout + run.stderr)[-2000:]
    assert any(name.endswith(".tar.gz") for name in os.listdir(out))
    # Nothing is written into the source tree: crops go with the rest of the pass
    assert not (image / "src" / "downlink").exists()
    assert (out / "queues").is_dir()
