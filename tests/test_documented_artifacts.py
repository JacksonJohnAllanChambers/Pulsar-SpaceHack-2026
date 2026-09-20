"""
What the docs promise, the archive has to contain.

These exist because of a defect the rest of the suite could not see. README.md pointed a reader at
`training/arctic_ice_chips.py` and `applet/models/verifier_ice_int8.onnx`; both files were sitting on
the author's disk, untracked, so every test passed and CI was green while the archive a judge clones
documented an uplinkable second model it did not ship. A test that asked the filesystem would have
passed too -- the files were right there. The question has to be whether git is carrying them.
"""

import os
import re
import subprocess

import numpy as np
import pytest

from applet.config import AppletConfig
from applet.pipelines.chip_verifier import ChipVerifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where a reader is sent. Everything else in the repo is code, and the compiler checks that.
DOC_FILES = ["README.md", "SUBMISSION.md"] + [
    os.path.join("docs", f) for f in sorted(os.listdir(os.path.join(ROOT, "docs"))) if f.endswith(".md")
]

# A token is only treated as a repo path if it starts with one of these. Anything else in backticks
# is prose, a shell flag or a config key, and guessing at those is how this kind of test gets flaky.
SHIPPED_DIRS = ("applet/", "ground/", "src/", "simulation/", "training/", "scripts/",
                "tests/", "docker/", "docs/", "data/labels/", "review/", ".github/")

# Rebuilt by scripts/setup_data.py, or created by a run (the pyFlows queues mkdir themselves on first
# use) -- deliberately not in git, and the docs say so where they name them.
GENERATED = ("data/outputs", "data/sample_bundle", "data/eval_bundle", "data/heldout_bundle",
             "data/real", "data/cache", "dist/",
             "src/downlink", "src/sent", "src/fleet_alerts")

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE = re.compile(r"`([^`\n]+)`")


def tracked_paths():
    """The set git would hand a judge, or -- unpacked from the archive, where there is no git -- the disk."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.stdout.split("\n"))


def referenced_paths():
    for doc in DOC_FILES:
        with open(os.path.join(ROOT, doc), encoding="utf-8") as handle:
            text = handle.read()
        for target in LINK.findall(text):
            target = target.split("#")[0].strip()
            if target and not target.startswith(("http", "mailto:", "#")):
                yield doc, os.path.normpath(os.path.join(os.path.dirname(doc), target)).replace("\\", "/")
        for token in CODE.findall(text):
            token = token.strip().rstrip(".,;:").strip()
            token = token.split("::")[0]                    # a pytest node id names a file and a test
            if "{" in token or "}" in token or "*" in token:
                continue                                    # `verifier_{fp32,int8}.onnx` is prose, not a path
            if token.startswith(SHIPPED_DIRS) and " " not in token:
                yield doc, token


def test_every_repo_path_the_docs_quote_is_actually_shipped():
    shipped = tracked_paths()
    if shipped is None:                                    # unpacked archive: the disk IS the archive
        shipped = set()
        for base, _, files in os.walk(ROOT):
            for name in files:
                rel = os.path.relpath(os.path.join(base, name), ROOT).replace("\\", "/")
                shipped.add(rel)

    missing = []
    for doc, path in referenced_paths():
        if path.startswith(GENERATED):
            continue
        # A directory is referenced by naming anything under it.
        if path in shipped or any(s.startswith(path.rstrip("/") + "/") for s in shipped):
            continue
        missing.append(f"{doc} -> {path}")

    assert not missing, (
        "the docs send a reader to files the archive does not contain:\n  " + "\n  ".join(sorted(set(missing)))
    )


def test_the_documented_ice_model_loads_and_scores():
    """
    README calls the ice verifier an uplink option selected by `verifier.model_path`. That claim is only
    worth making if the file it names is a working graph, so load it the way an operator's config would.
    """
    config = AppletConfig()
    config.verifier.model_path = "applet/models/verifier_ice_int8.onnx"
    config.verifier.fallback_model_path = ""               # no silent fall back to the flight model
    verifier = ChipVerifier(config)
    if verifier.status.startswith("RUNTIME_UNAVAILABLE"):
        pytest.skip(f"onnxruntime unavailable here: {verifier.status}")

    assert verifier.status == "READY", verifier.status
    assert verifier.model_path.endswith("verifier_ice_int8.onnx")

    chips = np.random.default_rng(2026).random((5, 64, 64, 4)).astype(np.float32) * 0.3
    scores = verifier.predict(chips)
    assert scores.shape == (5,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_the_ice_model_is_not_the_flight_model():
    """The Arctic model costs +26 temperate false alarms. It is opt-in, and a default must never drift to it."""
    assert AppletConfig().verifier.model_path.endswith("verifier_int8.onnx")
    with open(os.path.join(ROOT, "config.example.yaml"), encoding="utf-8") as handle:
        assert "verifier_ice" not in handle.read()
