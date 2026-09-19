#!/usr/bin/env bash
# Runs the applet on an Apple Silicon Mac over the tailnet.
#
# Why this exists: docker/Dockerfile.arm64 builds linux/arm64, but on an x86 dev box that runs
# under QEMU, which is a does-it-fit check and nothing more -- any timing taken there is fiction.
# An M-series Mac executes aarch64 natively, so numbers from it mean something.
#
#   scripts/run_on_macmini.sh --native                          # no Docker needed
#   scripts/run_on_macmini.sh --native --bundle data/real/s2_us_bundle
#   scripts/run_on_macmini.sh                                   # judges' container, needs Docker
#
# --native runs the flight code directly with OMP_NUM_THREADS=6 to mirror --cpus=6. There is no
# cgroup cap, so peak RAM is measured against the 14 GB envelope rather than enforced by it.
# Neither mode is a Jetson prediction: an M4 is much quicker than an Orin NX. What they establish
# is that the code is correct and fast on aarch64, and that nothing depends on x86.
#
# Results land in data/outputs/macmini/.

set -euo pipefail

HOST="${HOST:-mac}"
REMOTE_DIR="${REMOTE_DIR:-pulsar-spacehack}"   # relative to the remote home; no ~ expansion games
IMAGE="jetson-spacehack"
BUNDLE=""
SKIP_BUILD=0
SKIP_SHIP=0
NATIVE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --bundle) BUNDLE="$2"; shift 2 ;;
        --native) NATIVE=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-ship) SKIP_SHIP=1; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
LOCAL_OUT="data/outputs/macmini"
mkdir -p "$LOCAL_OUT"

say() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$1" "$2"; }

say "1/5" "checking $HOST over the tailnet"
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" 'true' 2>/dev/null; then
    cat >&2 <<EOF
Cannot reach '$HOST' with key auth. On the Mac, either authorise this key:

  mkdir -p ~/.ssh && echo '$(cat ~/.ssh/id_ed25519.pub 2>/dev/null)' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

or enable Tailscale SSH:  sudo tailscale set --ssh
EOF
    exit 1
fi
ssh "$HOST" 'echo "    $(uname -m) $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"'

if [ "$NATIVE" -eq 0 ]; then
    say "2/5" "checking Docker on $HOST"
    if ! ssh "$HOST" 'command -v docker >/dev/null 2>&1'; then
        cat >&2 <<EOF

Docker is not installed on $HOST, so the judges' container cannot be built there.
Re-run with --native to measure on the host directly (still a real ARM64 number, which is
the reason for using this machine), or install Docker Desktop / OrbStack on $HOST.
EOF
        exit 1
    fi
    ssh "$HOST" 'docker info >/dev/null 2>&1' || { echo "Docker installed but not running on $HOST." >&2; exit 1; }
    ARCH="$(ssh "$HOST" 'docker info --format "{{.Architecture}}"')"
    echo "    docker architecture: $ARCH"
    case "$ARCH" in
        aarch64|arm64) echo "    native ARM64 -- timings from this host are meaningful" ;;
        *) echo "    WARNING: $ARCH, so linux/arm64 runs emulated; treat timings as fit-checks only" ;;
    esac
fi

if [ "$SKIP_SHIP" -eq 0 ]; then
    say "3/5" "shipping the working tree to $HOST:~/$REMOTE_DIR"
    ssh "$HOST" "rm -rf ~/$REMOTE_DIR && mkdir -p ~/$REMOTE_DIR"
    # Tracked files plus new untracked ones, minus anything gitignored: what a clone would get,
    # including work not committed yet. Binary-safe because this is bash, not PowerShell.
    git ls-files -c -o --exclude-standard -z \
        | tar --null -czf - -T - \
        | ssh "$HOST" "tar xzf - -C ~/$REMOTE_DIR"
    echo "    $(git ls-files -c -o --exclude-standard | wc -l | tr -d ' ') files"
fi

if [ -n "$BUNDLE" ]; then
    say "3b/5" "shipping $BUNDLE ($(du -sh "$BUNDLE" | cut -f1))"
    ssh "$HOST" "mkdir -p ~/$REMOTE_DIR/$(dirname "$BUNDLE")"
    scp -q -r "$BUNDLE" "$HOST:~/$REMOTE_DIR/$(dirname "$BUNDLE")/"
fi

if [ "$NATIVE" -eq 1 ]; then
    say "4/5" "installing flight dependencies in a venv on $HOST"
    # A remote script rather than an inline heredoc: quoting a multi-line payload through ssh,
    # the local shell and zsh is how this script broke the first time.
    REMOTE_SCRIPT="$(mktemp)"
    cat > "$REMOTE_SCRIPT" <<'REMOTE'
set -e
cd ~/REMOTE_DIR_PLACEHOLDER
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt pytest
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=.venv/bin/python

echo
echo "=== host ==="
sysctl -n machdep.cpu.brand_string
echo "cores $(sysctl -n hw.ncpu), mem $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, OMP_NUM_THREADS=$OMP_NUM_THREADS"
$PY -c "import platform,numpy,cv2,onnxruntime as ort; print('python',platform.python_version(),platform.machine()); print('numpy',numpy.__version__,'cv2',cv2.__version__,'ort',ort.__version__); print('providers',ort.get_available_providers())"

echo
echo "=== tests (flight dependencies only) ==="
$PY -m pytest -q 2>&1 | tail -3

echo
echo "=== benchmark: full swath ==="
$PY scripts/generate_synthetic_data.py --output data/sample_bundle --size 1024 >/dev/null 2>&1 || true
$PY scripts/benchmark.py --full-swath 2>&1 | tail -22

if [ -d data/real/s2_us_bundle ]; then
  echo
  echo "=== 16 real Sentinel-2 scenes + real AIS ==="
  LBL=""
  [ -f labels_team.json ] && LBL="--labels labels_team.json"
  $PY scripts/scorecard.py -i data/real/s2_us_bundle -o data/outputs/macmini_real $LBL 2>&1 | tail -18
  echo
  echo "downlink sha256 (compare with the x86 run):"
  shasum -a 256 data/outputs/macmini_real/downlink_*.tar.gz
fi
REMOTE
    sed -i.bak "s|REMOTE_DIR_PLACEHOLDER|$REMOTE_DIR|" "$REMOTE_SCRIPT" && rm -f "$REMOTE_SCRIPT.bak"
    scp -q "$REMOTE_SCRIPT" "$HOST:~/.pulsar_run.sh"
    rm -f "$REMOTE_SCRIPT"

    say "5/5" "running natively (6 threads, no cgroup cap)"
    ssh "$HOST" 'bash ~/.pulsar_run.sh' 2>&1 | tee "$LOCAL_OUT/run.log"
else
    if [ "$SKIP_BUILD" -eq 0 ]; then
        say "4/5" "building $IMAGE for linux/arm64 on $HOST"
        ssh "$HOST" "cd ~/$REMOTE_DIR && docker buildx build --platform linux/arm64 -t $IMAGE -f docker/Dockerfile.arm64 --load ."
    fi
    say "5/5" "running under the judging constraints (14 GB, 6 CPUs, no network)"
    MOUNT=""
    [ -n "$BUNDLE" ] && MOUNT="-v \$HOME/$REMOTE_DIR/$BUNDLE:/workspace/data/real_bundle:ro"
    ssh "$HOST" "cd ~/$REMOTE_DIR && mkdir -p data/outputs && docker run --platform linux/arm64 \
        --memory=14g --memory-swap=14g --cpus=6 --network none --rm \
        -v \$HOME/$REMOTE_DIR/data/outputs:/workspace/data/outputs $MOUNT \
        $IMAGE" 2>&1 | tee "$LOCAL_OUT/run.log"
fi

say "done" "copying results back to $LOCAL_OUT"
scp -q -r "$HOST:~/$REMOTE_DIR/data/outputs/macmini_*" "$LOCAL_OUT/" 2>/dev/null || true
ls -la "$LOCAL_OUT"
echo "log: $LOCAL_OUT/run.log"
