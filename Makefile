# Makefile for robot_lab development tasks

.PHONY: help test test-smoke test-fast test-slow test-training test-coverage lint format clean install install-torch-gtx1080 install-torch-cuda126 install-torch-cpu verify-gpu

# pytest command with plugin isolation (avoids conflicts with system ROS pytest plugins)
PYTEST := PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
PYTEST_OPTS := -p no:ament_xmllint -p no:ament_flake8 -p no:ament_copyright \
               -p no:launch_testing -p no:launch_testing_ros -p no:ament_lint \
               -p no:ament_pep257 -p no:colcon_core

# Default target
help:
	@echo "Robot Lab Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-smoke     - Run smoke tests only (imports, dependencies)"
	@echo "  make test-fast      - Run fast tests (skip slow training tests)"
	@echo "  make test-slow      - Run only slow training tests"
	@echo "  make test-training  - Run training tests only"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with ruff"
	@echo ""
	@echo "Setup & Cleanup:"
	@echo "  make install        - Install dependencies with uv"
	@echo "  make clean          - Clean up generated files"
	@echo ""
	@echo "PyTorch Installation (GPU-specific):"
	@echo "  make install-torch-gtx1080  - Install PyTorch 2.9.1+cu126 (GTX 1080)"
	@echo "  make install-torch-cuda126  - Install latest PyTorch with CUDA 12.6"
	@echo "  make install-torch-cpu      - Install CPU-only PyTorch"
	@echo "  make verify-gpu             - Verify GPU setup and PyTorch"

# Run all tests
test:
	@echo "Running all tests..."
	@$(PYTEST) tests/ $(PYTEST_OPTS)

# Run smoke tests only
test-smoke:
	@echo "Running smoke tests only..."
	@$(PYTEST) tests/test_smoke.py $(PYTEST_OPTS)

# Run fast tests (exclude slow training tests)
test-fast:
	@echo "Running fast tests (excluding slow training tests)..."
	@$(PYTEST) tests/ -m "not slow" $(PYTEST_OPTS)

# Run only slow training tests
test-slow:
	@echo "Running only slow training tests..."
	@$(PYTEST) tests/ -m "slow" $(PYTEST_OPTS)

# Run training tests
test-training:
	@echo "Running training tests..."
	@$(PYTEST) tests/test_training.py $(PYTEST_OPTS)

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage report..."
	@if ! .venv/bin/python -c "import pytest_cov" 2>/dev/null; then \
		echo "Installing pytest-cov..."; \
		uv pip install pytest-cov; \
	fi
	@$(PYTEST) tests/ --cov=robot_lab --cov-report=html --cov-report=term $(PYTEST_OPTS)
	@echo "Coverage report saved to htmlcov/index.html"

# Lint code
lint:
	@echo "Running ruff linter..."
	@ruff check robot_lab/ tests/

# Format code
format:
	@echo "Formatting code with ruff..."
	@ruff format robot_lab/ tests/
	@ruff check --fix robot_lab/ tests/

# Install dependencies
install:
	@echo "Installing dependencies with uv..."
	@uv sync
	@echo ""
	@echo "✓ Installation complete!"
	@echo "PyTorch 2.9.1+cu126 has been installed from dependencies."
	@echo ""
	@echo "Run 'make verify-gpu' to verify GPU setup."
	@echo ""
	@echo "To change PyTorch version for different GPU:"
	@echo "  make install-torch-cuda126  (RTX series - latest PyTorch)"
	@echo "  make install-torch-cpu      (CPU only)"

# Install PyTorch for GTX 1080 (Compute Capability 6.1)
install-torch-gtx1080:
	@echo "Installing PyTorch 2.9.1+cu126 for GTX 1080..."
	@echo "This version is required for Compute Capability 6.1 (sm_61)"
	@uv pip install --python .venv/bin/python \
		torch==2.9.1+cu126 torchvision==0.24.1+cu126 \
		--index-url https://download.pytorch.org/whl/cu126 --reinstall
	@echo ""
	@echo "✓ PyTorch 2.9.1+cu126 installed!"
	@echo "Run 'make verify-gpu' to verify the installation."

# Install PyTorch for modern GPUs (RTX 20xx/30xx/40xx)
install-torch-cuda126:
	@echo "Installing latest PyTorch with CUDA 12.6..."
	@uv pip install --python .venv/bin/python \
		torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cu126
	@echo ""
	@echo "✓ PyTorch with CUDA 12.6 installed!"
	@echo "Run 'make verify-gpu' to verify the installation."

# Install CPU-only PyTorch
install-torch-cpu:
	@echo "Installing CPU-only PyTorch..."
	@uv pip install --python .venv/bin/python \
		torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo "✓ CPU-only PyTorch installed!"

# Verify GPU setup
verify-gpu:
	@echo "Verifying GPU setup..."
	@if [ ! -f docs/verify_gpu_setup.py ]; then \
		echo "Error: docs/verify_gpu_setup.py not found"; \
		exit 1; \
	fi
	@.venv/bin/python docs/verify_gpu_setup.py

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"
