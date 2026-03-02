#!/usr/bin/env python3
"""Verify GPU and PyTorch setup for robot-lab.

Run this script to check if your GPU/CUDA/PyTorch configuration is correct.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("=" * 70)
    print("PYTHON VERSION CHECK")
    print("=" * 70)
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        print("✓ Python 3.12+ detected (required for PyTorch cu126)")
        return True
    else:
        print("✗ Python 3.12+ required for CUDA 12.6 wheels")
        print("  Run: uv python pin 3.12 && rm -rf .venv && uv sync")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print("\n" + "=" * 70)
    print("PYTORCH CONFIGURATION")
    print("=" * 70)
    
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Run: uv pip install --python .venv/bin/python torch==2.9.1+cu126 \\")
        print("         torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126")
        return False
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Check version for GTX 1080 compatibility
    version_parts = torch.__version__.split('+')[0].split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    if major > 2 or (major == 2 and minor >= 10):
        print("⚠ WARNING: PyTorch 2.10+ does NOT support GTX 1080 (sm_61)")
        print("  GTX 1080 requires PyTorch ≤ 2.9")
        print("  Run: uv pip install --python .venv/bin/python torch==2.9.1+cu126 \\")
        print("         torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126 --reinstall")
        has_version_warning = True
    else:
        print("✓ PyTorch version compatible with GTX 1080")
        has_version_warning = False
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("✗ CUDA not available")
        print("  Check:")
        print("  1. NVIDIA drivers installed: nvidia-smi")
        print("  2. PyTorch CUDA version matches driver")
        return False
    
    print(f"✓ CUDA detected")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    
    # Check GPU details
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f"\nGPU {i}: {gpu_name}")
        print(f"  Compute Capability: {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")
        
        # Check GTX 1080 specifically
        if "GTX 1080" in gpu_name:
            if capability == (6, 1):
                print("  ✓ GTX 1080 detected (sm_61)")
                if not has_version_warning:
                    print("  ✓ PyTorch version compatible")
                else:
                    print("  ✗ PyTorch version NOT compatible (see warning above)")
            else:
                print(f"  ⚠ Unexpected compute capability for GTX 1080: {capability}")
        
        # Check if capability is supported
        if capability[0] < 7 and has_version_warning:
            print("  ✗ Compute capability < 7.0 not supported by PyTorch 2.10+")
    
    # Check for warnings by attempting to use CUDA
    print("\n" + "-" * 70)
    print("Testing CUDA operations...")
    try:
        # This will trigger any compatibility warnings
        x = torch.tensor([1.0], device='cuda')
        print("✓ CUDA operations work without errors")
        return not has_version_warning
    except Exception as e:
        print(f"✗ CUDA operation failed: {e}")
        return False

def check_environment():
    """Check if running in correct virtual environment."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    venv_path = Path(sys.executable).parent.parent
    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment: {venv_path}")
    
    # Check if in project venv
    expected_venv = Path.cwd() / ".venv"
    if venv_path.resolve() == expected_venv.resolve():
        print("✓ Running in project virtual environment")
        return True
    else:
        print("⚠ Not running in project .venv")
        print(f"  Use: .venv/bin/python {__file__}")
        return True  # Not critical

def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("robot-lab GPU Setup Verification")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Virtual Environment", check_environment()),
        ("PyTorch & CUDA", check_pytorch()),
    ]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        all_passed = all_passed and passed
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All checks passed! GPU setup is correct.")
        print("\nYou can now run training with GPU acceleration:")
        print("  robot-lab train --env Walker2d-v5 --algo SAC")
        return 0
    else:
        print("\n✗ Some checks failed. See above for fixes.")
        print("\nRefer to docs/GPU_SETUP.md for detailed setup instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
