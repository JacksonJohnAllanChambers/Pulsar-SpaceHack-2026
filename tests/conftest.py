"""Shared fixtures: one small synthetic bundle rendered once per test session."""

import os
import sys
import subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle")
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "generate_synthetic_data.py"), "--output", str(out), "--size", "512"],
        check=True, capture_output=True,
    )
    return str(out)
