"""
Test script to verify YAML-based experiment tracking and validation.

This script tests:
1. YAML-based hyperparameter logging
2. YAML-based environment config logging
3. YAML-based metadata and system info logging
4. Hyperparameter validation
5. File format verification

Usage:
    python tests/test_yaml_tracking.py
"""

import tempfile
import yaml
from pathlib import Path
from robot_lab.experiments import ExperimentTracker


def test_basic_yaml_logging():
    """Test basic YAML logging functionality."""
    print("🧪 Test 1: Basic YAML Logging")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="test_yaml",
            run_name="run1",
            base_dir=tmpdir
        )
        
        # Log hyperparameters
        hyperparams = {
            "algorithm": "SAC",
            "learning_rate": 0.0003,
            "batch_size": 256,
            "buffer_size": 300000,
            "gamma": 0.99,
            "tau": 0.02,
        }
        
        tracker.log_params(hyperparams)
        
        # Verify YAML file exists
        yaml_file = tracker.run_dir / "hyperparameters.yaml"
        assert yaml_file.exists(), "hyperparameters.yaml not created"
        
        # Load and verify
        with open(yaml_file) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded == hyperparams, "YAML content doesn't match original"
        print("  ✅ Basic YAML logging works")
        print(f"  📁 File: {yaml_file}")


def test_environment_config_yaml():
    """Test environment config YAML logging."""
    print("\n🧪 Test 2: Environment Config YAML")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="test_env_config",
            run_name="run1",
            base_dir=tmpdir
        )
        
        env_config = {
            "env_id": "A1Quadruped-v0",
            "control_parameters": {
                "kp": 100.0,
                "kd": 0.5,
                "actuator_type": "position",
            },
            "physics_parameters": {
                "timestep": 0.01,
                "gravity": [0, 0, -9.81],
                "integrator": "euler",
            }
        }
        
        tracker.log_env_config(env_config)
        
        # Verify YAML file exists
        yaml_file = tracker.run_dir / "environment_config.yaml"
        assert yaml_file.exists(), "environment_config.yaml not created"
        
        # Load and verify
        with open(yaml_file) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded == env_config, "ENV config doesn't match"
        print("  ✅ Environment config YAML logging works")
        print(f"  📁 File: {yaml_file}")


def test_hyperparameter_validation():
    """Test hyperparameter validation."""
    print("\n🧪 Test 3: Hyperparameter Validation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="test_validation",
            run_name="run1",
            base_dir=tmpdir
        )
        
        # Valid parameters should pass
        valid_params = {
            "learning_rate": 0.001,
            "batch_size": 256,
            "gamma": 0.99,
        }
        tracker.log_params(valid_params, validate=True)
        print("  ✅ Valid parameters accepted")
        
        # Invalid parameters should fail
        try:
            invalid_params = {
                "learning_rate": "invalid",  # Should be numeric
                "batch_size": 256,
            }
            tracker.log_params(invalid_params, validate=True)
            print("  ❌ Validation failed to catch invalid parameter")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  ✅ Validation correctly rejected invalid parameter: {e}")
        
        # Negative batch_size should fail
        try:
            invalid_params = {
                "batch_size": -10,  # Must be positive
            }
            tracker.log_params(invalid_params, validate=True)
            print("  ❌ Validation failed to catch negative batch_size")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  ✅ Validation correctly rejected negative value: {e}")


def test_yaml_readability():
    """Test that YAML files are human-readable."""
    print("\n🧪 Test 4: YAML Readability")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="test_readability",
            run_name="run1",
            base_dir=tmpdir
        )
        
        hyperparams = {
            "algorithm": "SAC",
            "learning_rate": 0.0003,
            "network_architecture": {
                "hidden_layers": [256, 256],
                "activation": "relu",
            },
            "training": {
                "total_timesteps": 1000000,
                "batch_size": 256,
            }
        }
        
        tracker.log_params(hyperparams)
        
        # Read raw file content
        yaml_file = tracker.run_dir / "hyperparameters.yaml"
        content = yaml_file.read_text()
        
        print("  📄 YAML content preview:")
        print("  " + "\n  ".join(content.split('\n')[:10]))
        
        # Verify it's readable (has proper indentation, no flow style)
        assert ":" in content, "YAML should have key:value pairs"
        assert "  " in content, "YAML should have indentation"
        assert "{" not in content or content.count("{") < 2, "Should use block style, not flow style"
        
        print("  ✅ YAML is human-readable")


def test_all_configs_yaml():
    """Test that all config files are YAML format."""
    print("\n🧪 Test 5: All Configs in YAML Format")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="test_all_yaml",
            run_name="run1",
            base_dir=tmpdir
        )
        
        # Log some data
        tracker.log_params({"algorithm": "SAC"})
        tracker.log_env_config({"env_id": "Test-v0"})
        tracker.set_status("completed")
        
        # Check that config files are YAML
        yaml_files = [
            "metadata.yaml",
            "hyperparameters.yaml",
            "environment_config.yaml",
            "system_info.yaml",
        ]
        
        for filename in yaml_files:
            file_path = tracker.run_dir / filename
            assert file_path.exists(), f"{filename} not created"
            
            # Verify it's valid YAML
            with open(file_path) as f:
                data = yaml.safe_load(f)
                assert data is not None, f"{filename} is empty"
            
            print(f"  ✅ {filename} is valid YAML")
        
        # Verify metrics.json is still JSON (not YAML)
        metrics_file = tracker.run_dir / "metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file) as f:
                data = json.load(f)
            print("  ✅ metrics.json is still JSON (for time-series data)")
        
        print("  ✅ All config files use YAML format")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing YAML-based Experiment Tracking")
    print("=" * 60)
    
    try:
        test_basic_yaml_logging()
        test_environment_config_yaml()
        test_hyperparameter_validation()
        test_yaml_readability()
        test_all_configs_yaml()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
