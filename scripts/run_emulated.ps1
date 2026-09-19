# Windows PowerShell runner for Jetson Orin NX Emulation
# Enforces exact hackathon constraints: ARM64, 14GB RAM, 6 CPUs, zero network

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "🛰️  BUILDING & RUNNING JETSON EMULATION CONTAINER (ARM64)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Build ARM64 Docker Image
Write-Host "`n[STEP 1] Building ARM64 Docker image (linux/arm64)..." -ForegroundColor Yellow
docker buildx build --platform linux/arm64 -t jetson-spacehack -f docker/Dockerfile.arm64 .

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Docker build failed. Ensure Docker Desktop is running and QEMU ARM64 is enabled." -ForegroundColor Red
    exit 1
}

# 2. Ensure output directory exists locally
$OutputDir = Join-Path $PSScriptRoot "..\data\outputs"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# 3. Run container with strict constraints
Write-Host "`n[STEP 2] Running applet inside constrained Jetson environment:" -ForegroundColor Yellow
Write-Host "  • Platform:    linux/arm64 (NVIDIA Jetson Orin NX)"
Write-Host "  • Memory Cap:  14 GB (--memory=14g --memory-swap=14g)"
Write-Host "  • CPU Cores:   6 Cores (--cpus=6)"
Write-Host "  • Network:     DISCONNECTED (--network none)"
Write-Host "========================================================`n"

docker run --platform linux/arm64 `
    --memory=14g `
    --memory-swap=14g `
    --cpus=6 `
    --network none `
    --rm `
    -v "${OutputDir}:/workspace/data/outputs" `
    jetson-spacehack

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "✅ RUN COMPLETE! Output artifacts saved to: $OutputDir" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
