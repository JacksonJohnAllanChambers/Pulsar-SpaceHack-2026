#!/usr/bin/env bash
# Linux/macOS runner for Jetson Orin NX Emulation
# Enforces exact hackathon constraints: ARM64, 14GB RAM, 6 CPUs, zero network

set -e

echo "========================================================"
echo "🛰️  BUILDING & RUNNING JETSON EMULATION CONTAINER (ARM64)"
echo "========================================================"

# 1. Build ARM64 Docker Image
echo -e "\n[STEP 1] Building ARM64 Docker image (linux/arm64)..."
docker buildx build --platform linux/arm64 -t jetson-spacehack -f docker/Dockerfile.arm64 .

# 2. Ensure output directory exists locally
OUTPUT_DIR="$(pwd)/data/outputs"
mkdir -p "$OUTPUT_DIR"

# 3. Run container with strict constraints
echo -e "\n[STEP 2] Running applet inside constrained Jetson environment:"
echo "  • Platform:    linux/arm64 (NVIDIA Jetson Orin NX)"
echo "  • Memory Cap:  14 GB (--memory=14g --memory-swap=14g)"
echo "  • CPU Cores:   6 Cores (--cpus=6)"
echo "  • Network:     DISCONNECTED (--network none)"
echo "========================================================\n"

docker run --platform linux/arm64 \
    --memory=14g \
    --memory-swap=14g \
    --cpus=6 \
    --network none \
    --rm \
    -v "$OUTPUT_DIR:/workspace/data/outputs" \
    jetson-spacehack

echo -e "\n========================================================"
echo "✅ RUN COMPLETE! Output artifacts saved to: $OUTPUT_DIR"
echo "========================================================"
