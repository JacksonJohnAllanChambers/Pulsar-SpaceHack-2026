#!/usr/bin/env bash
# Runs the applet on an Apple Silicon Mac over the tailnet, inside the judges' container.
#
# Why this exists: docker/Dockerfile.arm64 builds linux/arm64, but on an x86 dev box that runs
# under QEMU, which is a does-it-fit check and nothing more -- any timing taken there is fiction.
# An M-series Mac runs the same image natively, so the numbers it produces are the closest thing
# to a Jetson Orin NX measurement available without the hardware. Same image, same
# --memory=14g --memory-swap=14g --cpus=6 --network none as the judging environment.
#
#   scripts/run_on_macmini.sh                     # ship, build, test, benchmark
#   scripts/run_on_macmini.sh --bundle data/real/s2_us_bundle   # ...also score real scenes
#   HOST=mac scripts/run_on_macmini.sh --skip-build
#
# Results land in data/outputs/macmini/.

set -euo pipefail

HOST="${HOST:-mac}"
REMOTE_DIR="${REMOTE_DIR:-~/pulsar-spacehack}"
IMAGE="jetson-spacehack"
BUNDLE=""
SKIP_BUILD=0
SKIP_SHIP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --bundle) BUNDLE="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-ship) SKIP_SHIP=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
LOCAL_OUT="data/outputs/macmini"
mkdir -p "$LOCAL_OUT"

say() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$1" "$2"; }

say "1/6" "checking $HOST over the tailnet"
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" 'true' 2>/dev/null; then
    cat >&2 <<EOF
Cannot reach '$HOST' with key auth. On the Mac, either authorise this key:

  mkdir -p ~/.ssh && echo '$(cat ~/.ssh/id_ed25519.pub 2>/dev/null)' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

or enable Tailscale SSH:  sudo tailscale set --ssh
EOF
    exit 1
fi
ssh "$HOST" 'echo "    $(uname -m) $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"'

say "2/6" "checking Docker on $HOST"
if ! ssh "$HOST" 'docker info >/dev/null 2>&1'; then
    echo "Docker is not running on $HOST. Start Docker Desktop there and re-run." >&2
    exit 1
fi
ARCH="$(ssh "$HOST" 'docker info --format "{{.Architecture}}"')"
echo "    docker architecture: $ARCH"
case "$ARCH" in
    aarch64|arm64) echo "    native ARM64 -- timings from this host are meaningful" ;;
    *) echo "    WARNING: $ARCH, so linux/arm64 runs under emulation; treat timings as fit-checks only" ;;
esac

if [ "$SKIP_SHIP" -eq 0 ]; then
    say "3/6" "shipping the working tree to $HOST:$REMOTE_DIR"
    ssh "$HOST" "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR"
    # Tracked files plus new untracked ones, minus anything gitignored: the same set a clone
    # would get, but including work that has not been committed yet.
    git ls-files -c -o --exclude-standard -z \
        | tar --null -czf - -T - \
        | ssh "$HOST" "tar xzf - -C $REMOTE_DIR"
    echo "    $(git ls-files -c -o --exclude-standard | wc -l | tr -d ' ') files"
fi

if [ -n "$BUNDLE" ]; then
    say "3b/6" "shipping $BUNDLE ($(du -sh "$BUNDLE" | cut -f1))"
    ssh "$HOST" "mkdir -p $REMOTE_DIR/data/real"
    tar -czf - "$BUNDLE" | ssh "$HOST" "tar xzf - -C $REMOTE_DIR"
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
    say "4/6" "building $IMAGE for linux/arm64 on $HOST"
    ssh "$HOST" "cd $REMOTE_DIR && docker buildx build --platform linux/arm64 -t $IMAGE -f docker/Dockerfile.arm64 --load ."
fi

say "5/6" "running under the judging constraints (14 GB, 6 CPUs, no network)"
MOUNT=""
if [ -n "$BUNDLE" ]; then
    MOUNT="-v $REMOTE_DIR/$BUNDLE:/workspace/data/real_bundle:ro"
fi

# One container, several measurements: the flight path on synthetic scenes, the full-swath
# benchmark, and -- when a real bundle is mounted -- the real-scene scorecard.
ssh "$HOST" "cd $REMOTE_DIR && mkdir -p data/outputs && docker run --platform linux/arm64 \
    --memory=14g --memory-swap=14g --cpus=6 --network none --rm \
    -v $REMOTE_DIR/data/outputs:/workspace/data/outputs $MOUNT \
    --entrypoint bash $IMAGE -lc '
set -e
echo \"=== host: \$(uname -m), \$(nproc) cores visible ===\"
python3 -c \"import numpy, cv2, onnxruntime as ort; print(\\\"numpy\\\", numpy.__version__, \\\"cv2\\\", cv2.__version__, \\\"ort\\\", ort.__version__, ort.get_available_providers())\"
echo
echo \"=== synthetic pass ===\"
python3 -m applet run -i data/sample_bundle -o data/outputs/macmini_sample -c config.example.yaml
echo
echo \"=== benchmark (full swath) ===\"
python3 scripts/benchmark.py --full-swath -o data/outputs/macmini_benchmark || python3 scripts/benchmark.py --full-swath
if [ -d /workspace/data/real_bundle ]; then
  echo
  echo \"=== real Sentinel-2 scenes + real AIS ===\"
  python3 scripts/scorecard.py -i /workspace/data/real_bundle -o data/outputs/macmini_real
fi
' " 2>&1 | tee "$LOCAL_OUT/run.log"

say "6/6" "copying results back to $LOCAL_OUT"
scp -q -r "$HOST:$REMOTE_DIR/data/outputs/macmini_*" "$LOCAL_OUT/" 2>/dev/null || true
ls -la "$LOCAL_OUT"
say "done" "log: $LOCAL_OUT/run.log"
