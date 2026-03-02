# Robot Lab Tests

This directory contains unit and integration tests for the robot_lab package.

## Test Structure

- **`test_smoke.py`**: Smoke tests that verify basic package functionality
  - Import tests for all modules
  - Dependency availability checks
  - Configuration file validation
  - Custom environment registration

- **`test_training.py`**: Training pipeline tests
  - Basic training functionality
  - MountainCarContinuous-v0 learning verification
  - Model saving and loading
  - Reward threshold validation

- **`conftest.py`**: Shared pytest fixtures and configuration
  - `temp_output_dir`: Temporary directory for test outputs
  - `test_seed`: Consistent random seed for reproducibility

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only fast tests (skip slow training tests)
```bash
pytest tests/ -m "not slow"
```

### Run only smoke tests
```bash
pytest tests/test_smoke.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test
```bash
pytest tests/test_training.py::TestMountainCarTraining::test_mountaincar_learns_to_improve -v
```

## Test Markers

Tests are marked with the following categories:

- **`@pytest.mark.slow`**: Training tests that take >10 seconds (e.g., full MountainCar training)
- **`@pytest.mark.fast`**: Quick tests that complete in <5 seconds
- **`@pytest.mark.smoke`**: Basic import and dependency tests (implied in test_smoke.py)

## Writing New Tests

When adding new tests:

1. **Follow naming convention**: `test_*.py` for files, `Test*` for classes, `test_*` for functions
2. **Use fixtures**: Leverage `temp_output_dir` and `test_seed` from conftest.py
3. **Add markers**: Mark slow tests with `@pytest.mark.slow`
4. **Document**: Add docstrings explaining what each test verifies
5. **Clean up**: Always clean up resources (environments, files) after tests

## Test Coverage

Key areas covered:
- ✓ Module imports and availability
- ✓ Dependency installation verification
- ✓ Configuration file loading
- ✓ Custom environment registration
- ✓ Training pipeline functionality
- ✓ Model saving and loading
- ✓ Reward threshold validation (MountainCar)

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest tests/ -m "not slow"  # Skip slow tests in CI
```

For full test suite including training:
```bash
pytest tests/ --durations=10  # Show 10 slowest tests
```
