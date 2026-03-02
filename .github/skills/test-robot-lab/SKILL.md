---
name: test-robot-lab
description: Testing Guide for robot_lab. Includes test structure, running instructions, and troubleshooting.
user-invokable: true
---
# Robot Lab Testing Guide

This document describes the testing infrastructure for the robot_lab project.

## Test Suite Overview

The test suite consists of two main categories:

### 1. Smoke Tests (`tests/test_smoke.py`)
Fast tests that verify basic functionality:
- **Module Imports**: All core modules can be imported without errors
- **Dependencies**: Required packages (Gymnasium, Stable-Baselines3, PyTorch, etc.) are available
- **Configuration**: Config files are accessible and loadable
- **Environments**: Custom environments are registered properly

### 2. Training Tests (`tests/test_training.py`)
Integration tests that verify the RL training pipeline:
- **Basic Training**: Training runs complete and generate expected outputs
- **Output Structure**: Correct directory structure and files are created
- **Short Training**: Quick validation that training completes without errors
- **Learning Verification**: MountainCarContinuous-v0 training improves beyond threshold (marked as `@pytest.mark.slow`)

## Running Tests

### Using Makefile (Recommended)

```bash
# Show all available commands
make help

# Run smoke tests only (fast, ~2 seconds)
make test-smoke

# Run fast tests (excludes slow training tests)
make test-fast

# Run training tests
make test-training

# Run only slow tests (full MountainCar learning verification)
make test-slow

# Run all tests
make test

# Run with coverage report
make test-coverage
```

### Direct pytest Commands

If you prefer using pytest directly:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run smoke tests
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_smoke.py -v

# Run specific test
python -m pytest tests/test_training.py::TestTrainingBasics::test_train_returns_expected_outputs -v

# Run with markers
python -m pytest tests/ -m "not slow"  # Skip slow tests
python -m pytest tests/ -m "slow"       # Only slow tests
```

## Test Configuration

### Plugin Isolation
The test suite automatically isolates from system-installed pytest plugins (ROS, ament, etc.) using:
- `PYTHONNOUSERSITE=1`: Prevents loading system site-packages
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: Disables auto-loading of entry point plugins
- Explicit plugin disabling in `pytest.ini`

### Fixtures (`tests/conftest.py`)
- `temp_output_dir`: Temporary directory for test outputs (auto-cleaned)
- `test_seed`: Consistent random seed (42) for reproducible tests

### Test Markers
- `@pytest.mark.slow`: Tests that take >10 seconds (full training runs)
- `@pytest.mark.fast`: Tests that complete in <5 seconds
- `@pytest.mark.smoke`: Basic import and dependency tests

## Writing New Tests

### Example: Adding a Smoke Test

```python
# tests/test_smoke.py
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_new_module_imports(self):
        """Test that new module can be imported."""
        import robot_lab.new_module
        assert robot_lab.new_module is not None
```

### Example: Adding a Training Test

```python
# tests/test_training.py
class TestNewEnvironment:
    """Test training on new environment."""
    
    @pytest.mark.slow
    def test_new_env_learns(self, temp_output_dir, test_seed):
        \"\"\"Test that training improves on NewEnv-v0.\"\"\"
        model, model_path, vecnorm_path = train(
            env_name="NewEnv-v0",
            algorithm="SAC",
            seed=test_seed,
            output_dir=str(temp_output_dir),
        )
        
        # Evaluate and assert improvement
        mean_reward, _ = self._evaluate_model(model_path, vecnorm_path, "NewEnv-v0")
        assert mean_reward > THRESHOLD
```

## Continuous Integration

For CI/CD pipelines, skip slow tests:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: make test-fast  # Fast tests only in CI
```

For nightly builds, run full suite:

```yaml
- name: Run all tests
  run: make test
```

## Troubleshooting

### Issue: Import errors during tests
**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source .venv/bin/activate
uv sync
```

### Issue: ROS pytest plugin conflicts
**Solution**: The Makefile automatically handles this via `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

### Issue: Tests fail with "CUDA out of memory"
**Solution**: Training tests create temporary environments. Reduce `num_envs` in test configs or run tests sequentially:
```bash
pytest tests/ -n 1  # Single worker
```

### Issue: Pydantic deprecation warnings
**Solution**: These are expected during Pydantic V1→V2 migration. Tests still pass. To suppress:
```bash
pytest tests/ --disable-warnings
```

## Test Metrics

Expected test times (on GTX 1080 with CUDA):
- **Smoke tests**: ~2 seconds (16 tests)
- **Fast training tests**: ~30-60 seconds (3 tests)
- **Slow training tests**: ~5-10 minutes (1 test, full MountainCar training)

Success criteria:
- All smoke tests must pass
- Fast training tests complete without errors  
- Slow test: MountainCarContinuous-v0 mean reward > -50.0 after 20k timesteps

## Coverage

To generate coverage report:

```bash
make test-coverage

# Open HTML report
firefox htmlcov/index.html
```

Target coverage: >80% for core modules (training, config, utils)
