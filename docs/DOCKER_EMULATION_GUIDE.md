# 🐳 Jetson Orin NX Docker Emulation Guide

This guide details how to configure your laptop to run the emulated **NVIDIA Jetson Orin NX (ARM64)** environment required by the hackathon organizers.

---

## 1. System Requirements

* **OS:** Windows 10/11 (with WSL2), macOS (Apple Silicon or Intel), or Linux (Ubuntu/Debian).
* **RAM:** At least 8 GB free (16 GB recommended).
* **Disk Space:** ~15 GB free disk space.
* **Software:** Docker Desktop (latest version).

---

## 2. Platform Setup Instructions

### Windows (WSL2)
1. Open PowerShell as Administrator and run:
   ```powershell
   wsl --install
   ```
2. Restart your machine if prompted.
3. Install **Docker Desktop** from [docker.com](https://www.docker.com/).
4. In Docker Desktop Settings:
   * Navigate to **General** $\to$ check **Use the WSL 2 based engine**.
   * Under **Resources** $\to$ **WSL Integration**, enable your default distro.
5. Verify ARM64 emulation:
   ```powershell
   docker run --rm --platform linux/arm64 arm64v8/ubuntu:22.04 uname -m
   # Should print: aarch64
   ```

### macOS (Apple Silicon: M1/M2/M3)
1. Download and install Docker Desktop for Apple Silicon.
2. Apple Silicon natively runs ARM64! Docker will run `--platform linux/arm64` at near-native speed.
3. Verify:
   ```bash
   docker run --rm --platform linux/arm64 arm64v8/ubuntu:22.04 uname -m
   # Prints: aarch64
   ```

### Linux (Ubuntu / Debian)
1. Install Docker Engine and add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```
2. Register the QEMU ARM64 multi-architecture binfmt translator:
   ```bash
   docker run --privileged --rm tonistiigi/binfmt --install arm64
   ```
3. Verify:
   ```bash
   docker run --rm --platform linux/arm64 arm64v8/ubuntu:22.04 uname -m
   # Prints: aarch64
   ```

---

## 3. Running the Applet Inside the Constrained Jetson Container

The hackathon enforces 4 hardware constraints:
1. **`--platform linux/arm64`**: Emulates the Jetson Orin NX 64-bit ARM architecture.
2. **`--memory=14g` & `--memory-swap=14g`**: Emulates the 14 GB unified memory budget (stopping Docker from spilling into disk swap).
3. **`--cpus=6`**: Limits processing to 6 CPU cores.
4. **`--network none`**: Disconnects all internet access at runtime.

### One-Click Execution

#### On Windows (PowerShell):
```powershell
.\scripts\run_emulated.ps1
```

#### On Linux / macOS:
```bash
chmod +x scripts/run_emulated.sh
./scripts/run_emulated.sh
```

---

## 4. Troubleshooting

* **Error: `failed to solve with frontend dockerfile.v0`**:
  * Ensure Docker Desktop is running.
  * Check that QEMU binfmt is enabled.
* **Killed / Out of Memory (OOM)**:
  * If your laptop has less than 16 GB physical RAM, Docker may fail to allocate the full 14 GB. You can temporarily test with `--memory=8g --memory-swap=8g` locally, but ensure memory usage in `edge_telemetry.json` stays under 1 GB so it easily passes on the 14 GB competition test rig.
