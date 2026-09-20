"""
Build the submission archive from a commit, then prove the ARCHIVE works -- not the working tree.

A repo that runs on the author's machine proves little: untracked files, local datasets and stale
bytecode all quietly help. This script archives exactly what git has (`git archive`), unpacks it
somewhere empty, and from inside that copy alone:

  1. renders the synthetic sample bundle            (no network, seeded)
  2. runs the applet on it twice                     (the flight entry point, default config)
  3. checks the two downlink tarballs are byte-identical

and records the result next to the archive in dist/SUBMISSION_MANIFEST.json.

    python scripts/package_submission.py             # archive HEAD and verify it
    python scripts/package_submission.py --no-verify # archive only
    python scripts/package_submission.py --allow-dirty
"""

import os
import sys
import json
import glob
import shutil
import hashlib
import zipfile
import argparse
import tempfile
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "tactical-edge-sentinel"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_in(cwd: str, *cmd: str) -> None:
    # PYTHONDONTWRITEBYTECODE keeps the unpacked copy identical to the archive while we test it
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=cwd)
    done = subprocess.run([sys.executable, *cmd], cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stdout[-2000:] + done.stderr[-2000:])
        raise SystemExit(f"[package] FAILED inside the archive: {' '.join(cmd)}")


def verify(archive: str) -> dict:
    work = tempfile.mkdtemp(prefix="submission_verify_")
    try:
        with zipfile.ZipFile(archive) as z:
            z.extractall(work)
        tree = os.path.join(work, NAME)
        bundle = os.path.join(tree, "data", "sample_bundle")
        print("[package] rendering the sample bundle inside the archive copy ...")
        run_in(tree, "scripts/generate_synthetic_data.py", "--output", bundle)
        hashes, telemetry = [], {}
        for attempt in ("a", "b"):
            out = os.path.join(work, f"out_{attempt}")
            print(f"[package] applet run {attempt} ...")
            run_in(tree, "-m", "applet", "run", "--input", bundle, "--output", out)
            tarballs = glob.glob(os.path.join(out, "downlink_*.tar.gz"))
            if len(tarballs) != 1:
                raise SystemExit(f"[package] expected one downlink tarball, found {len(tarballs)}")
            hashes.append(sha256(tarballs[0]))
            with open(os.path.join(out, "edge_telemetry.json"), "r", encoding="utf-8") as f:
                telemetry = json.load(f)
            with open(os.path.join(out, "downlink_bundle", "manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
        if hashes[0] != hashes[1]:
            raise SystemExit("[package] FAILED: two runs of the archive produced different tarballs")
        return {
            "sample_bundle_scenes": len(glob.glob(os.path.join(bundle, "*.tif"))),
            "targets_detected": manifest.get("targets_detected"),
            "dark_vessels": manifest.get("dark_vessels"),
            "downlink_tarball_sha256": hashes[0],
            "downlink_tarball_bytes": os.path.getsize(tarballs[0]),
            "byte_identical_across_runs": True,
            "wall_clock_s": telemetry.get("wall_clock_time_s"),
            "peak_memory_mb": telemetry.get("peak_memory_mb"),
            "verified_on": f"{sys.platform} / python {sys.version.split()[0]}",
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="HEAD", help="commit, branch or tag to package (default HEAD)")
    ap.add_argument("--output", "-o", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="package even though tracked files have uncommitted changes (they are NOT included)")
    args = ap.parse_args()

    dirty = [line for line in git("status", "--porcelain").splitlines() if not line.startswith("??")]
    if dirty and not args.allow_dirty:
        print("[package] tracked files have uncommitted changes, and an archive is built from a commit:")
        print("\n".join("    " + d for d in dirty[:20]))
        print("[package] commit them, or pass --allow-dirty to package the commit as it stands.")
        return 1

    commit = git("rev-parse", args.ref)
    os.makedirs(args.output, exist_ok=True)
    archive = os.path.join(args.output, f"{NAME}-{commit[:7]}.zip")
    git("archive", "--format=zip", f"--prefix={NAME}/", "-o", archive, commit)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    print(f"[package] {archive}: {len(names)} files, {os.path.getsize(archive) / 1e6:.1f} MB")

    record = {
        "name": NAME,
        "commit": commit,
        # The ref asked for, not the checked-out branch: packaging origin/main from a Jack
        # checkout used to record "Jack", which is the one field nobody would think to doubt.
        "packaged_ref": args.ref,
        "contains_ref": sorted(git("branch", "-a", "--contains", commit).replace("*", " ").split()),
        "packaged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": os.path.basename(archive),
        "archive_sha256": sha256(archive),
        "archive_bytes": os.path.getsize(archive),
        "files": len(names),
        "uncommitted_tracked_changes_excluded": len(dirty),
        "verification": None if args.no_verify else verify(archive),
    }
    with open(os.path.join(args.output, "SUBMISSION_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
