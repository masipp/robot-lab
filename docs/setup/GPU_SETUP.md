# GPU Setup Guide for robot-lab

## Quick Verification

**Before reading this guide, run the verification script:**

```bash
.venv/bin/python verify_gpu_setup.py
```

This will check your Python version, PyTorch installation, and CUDA compatibility. If all checks pass, you're ready to train with GPU acceleration!

---

## Overview

This guide helps you set up PyTorch with the correct CUDA version for your GPU. Getting this right is critical - mismatched versions will cause PyTorch to either fail or fall back to CPU.

## Prerequisites Check

### 1. Check Your GPU Model and Compute Capability

```bash
# Check GPU model
lspci | grep -i nvidia

# Or if drivers are installed
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**Critical: GPU Compute Capability determines PyTorch compatibility**

| GPU Model | Compute Capability | PyTorch Version Support |
|-----------|-------------------|-------------------------|
| GTX 1080, 1080 Ti | 6.1 (sm_61) | PyTorch ≤ 2.9 |
| RTX 20xx series | 7.5 (sm_75) | All PyTorch versions |
| RTX 30xx series | 8.6 (sm_86) | All PyTorch versions |
| RTX 40xx series | 8.9 (sm_89) | All PyTorch versions |

**⚠️ GTX 1080 Warning**: PyTorch 2.10+ dropped support for sm_61. You **must** use PyTorch 2.9 or earlier!

### 2. Check NVIDIA Driver and CUDA Version

```bash
nvidia-smi
```

Look at the top right corner: `CUDA Version: XX.X`

Example output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
```

This shows the **maximum** CUDA version your driver supports.

### 3. Check Python Version

```bash
python3 --version
python3.12 --version 2>/dev/null || echo "Python 3.12 not installed"
```

**Requirement**: Python 3.12+ needed for cu126 wheels with PyTorch 2.9

## Installation Decision Matrix

### For GTX 1080 (Compute Capability 6.1)

**System Requirements:**
- NVIDIA Driver: 530+ (supports CUDA 12.x)
- Python: 3.12+
- PyTorch: 2.9.x (last version supporting sm_61)
- CUDA: 12.6 (cu126)

**Installation Steps:**

1. **Pin Python version to 3.12:**
```bash
uv python pin 3.12
```

2. **Update pyproject.toml:**
```toml
requires-python = ">=3.12"
```

3. **Recreate virtual environment:**
```bash
rm -rf .venv
uv sync --python 3.12
```

4. **Install PyTorch 2.9 with CUDA 12.6:**
```bash
uv pip install --python .venv/bin/python torch==2.9.1+cu126 torchvision==0.24.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126 --reinstall
```

5. **Verify installation:**
```bash
.venv/bin/python verify_gpu_setup.py
```

Or manually check:
```bash
.venv/bin/python -c "import torch; \
  print(f'PyTorch: {torch.__version__}'); \
  print(f'CUDA available: {torch.cuda.is_available()}'); \
  print(f'CUDA version: {torch.version.cuda}'); \
  print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
PyTorch: 2.9.1+cu126
CUDA available: True
CUDA version: 12.6
GPU: NVIDIA GeForce GTX 1080
```

**NO WARNINGS should appear!** If you see compute capability warnings, you have the wrong PyTorch version.

### For RTX 20xx/30xx/40xx (Compute Capability ≥ 7.0)

You can use the latest PyTorch:

```bash
# Check your driver CUDA version first
nvidia-smi  # Look for "CUDA Version: XX.X"

# For CUDA 12.x drivers (most recent)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 drivers
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Verification Checklist

Run these checks **in order**:

### ✅ 1. Driver Check
```bash
nvidia-smi
```
Should show your GPU with no errors.

### ✅ 2. Python Version Check
```bash
.venv/bin/python --version
```
Should show Python 3.12.x for cu126 wheels.

### ✅ 3. PyTorch Version Check
```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```
For GTX 1080: Should show `2.9.1+cu126` (NOT 2.10+)

### ✅ 4. CUDA Availability Check
```bash
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```
Should show: `CUDA: True, GPU: NVIDIA GeForce GTX 1080`

### ✅ 5. Compute Capability Check
```bash
.venv/bin/python -c "import torch; print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0)}')"
```
GTX 1080 should show: `GPU Compute Capability: (6, 1)`

### ✅ 6. No Compatibility Warnings
```bash
.venv/bin/python -c "import torch; torch.cuda.is_available()"
```
Should produce **NO** warnings about compute capability or sm_XX.

## Troubleshooting

### Warning: "CUDA capability sm_61 is not compatible"

**Cause**: You have PyTorch 2.10+ but GTX 1080 only supports up to PyTorch 2.9

**Fix**:
```bash
uv pip install --python .venv/bin/python torch==2.9.1+cu126 torchvision==0.24.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126 --reinstall
```

### Error: "CUDA driver version is insufficient"

**Cause**: Your driver doesn't support the CUDA version PyTorch needs

**Fix**:
```bash
# Update NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

### PyTorch shows wrong version after install

**Cause**: `uv run` re-syncs packages from pyproject.toml

**Fix**: Use `.venv/bin/python` directly instead of `uv run python`:
```bash
# ✓ Correct
.venv/bin/python your_script.py

# ✓ Also works (robot-lab CLI uses .venv automatically)
robot-lab train --env Walker2d-v5 --algo SAC

# ✗ Wrong (may reinstall different version)
uv run python your_script.py
```

### PyTorch wheels not found for Python version

**Cause**: cu126 wheels require Python 3.12+, cu118 requires Python 3.8-3.11

**Fix**:
```bash
# Check available Python versions
ls /usr/bin/python3*

# Pin to correct version
uv python pin 3.12

# Recreate environment
rm -rf .venv && uv sync
```

## Driver Installation (if needed)

### Ubuntu/Debian

```bash
# Auto-detect and install
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify after reboot
nvidia-smi
```

### Manual driver installation

```bash
# Remove old drivers
sudo apt purge nvidia-* -y
sudo apt autoremove -y

# Install specific version
sudo apt install nvidia-driver-550

# Reboot
sudo reboot
```

## Performance Notes

**GTX 1080 with robot-lab:**
- Vectorized environments: 8 parallel workers recommended
- Expected FPS: 2000-3000 (Walker2d-v5 with SAC)
- VRAM usage: ~2-4GB typical for RL training
- CPU vs GPU speedup: 5-7x faster

## Quick Reference

### GTX 1080 Setup (Complete)

```bash
# 1. Pin Python
uv python pin 3.12

# 2. Recreate environment
rm -rf .venv && uv sync

# 3. Install PyTorch 2.9 + CUDA 12.6
uv pip install --python .venv/bin/python \
  torch==2.9.1+cu126 \
  torchvision==0.24.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126 \
  --reinstall

# 4. Verify (comprehensive check)
.venv/bin/python verify_gpu_setup.py

# Or quick manual check (should show NO warnings)
.venv/bin/python -c "import torch; \
  assert torch.cuda.is_available(), 'CUDA not available'; \
  assert '2.9' in torch.__version__, 'Wrong PyTorch version'; \
  print('✓ Setup correct!')"
```

### Finding Available PyTorch Versions

```bash
# List available CUDA variants
curl -s https://download.pytorch.org/whl/ | grep -o 'cu[0-9]*' | sort -u

# List PyTorch versions for specific CUDA
curl -s https://download.pytorch.org/whl/cu126/torch/ | grep -o 'torch-2\.[0-9]\.[0-9]*+cu126' | sort -u
```

## Key Takeaways

1. **GPU Compute Capability determines PyTorch version** - GTX 1080 (sm_61) needs PyTorch ≤ 2.9
2. **Driver CUDA version must be ≥ PyTorch CUDA version** - CUDA 13.0 driver supports cu126 wheels
3. **Python version matters for wheel availability** - cu126 needs Python 3.12+
4. **Use `.venv/bin/python` directly** - Avoid `uv run` reinstalling wrong versions
5. **Verify with no warnings** - Compatibility warnings mean something is wrong

## References

- [PyTorch GTX 1080 Compatibility Discussion](https://discuss.pytorch.org/t/what-version-of-pytorch-is-compatible-with-nvidia-geforce-gtx-1080/222056/6)
- [PyTorch CUDA Wheel Downloads](https://download.pytorch.org/whl/)
- [NVIDIA GPU Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
