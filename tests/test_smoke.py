"""Smoke tests to verify basic package functionality."""

import pytest
import sys
from pathlib import Path


class TestImports:
    """Test that all core modules can be imported without errors."""
    
    def test_import_robot_lab(self):
        """Test that the main package can be imported."""
        import robot_lab
        assert robot_lab is not None
    
    def test_import_cli(self):
        """Test that the CLI module can be imported."""
        import robot_lab.cli
        assert robot_lab.cli is not None
        assert hasattr(robot_lab.cli, 'app')
    
    def test_import_training(self):
        """Test that the training module can be imported."""
        import robot_lab.training
        assert robot_lab.training is not None
        assert hasattr(robot_lab.training, 'train')
    
    def test_import_config(self):
        """Test that the config module can be imported."""
        import robot_lab.config
        assert robot_lab.config is not None
        assert hasattr(robot_lab.config, 'load_hyperparameters')
    
    def test_import_visualization(self):
        """Test that the visualization module can be imported."""
        import robot_lab.visualization
        assert robot_lab.visualization is not None
        assert hasattr(robot_lab.visualization, 'visualize_policy')
    
    def test_import_experiments(self):
        """Test that the experiments module can be imported."""
        import robot_lab.experiments
        assert robot_lab.experiments is not None
        # Check key components (schemas may have Pydantic v1/v2 warnings but should import)
        from robot_lab.experiments import ExperimentTracker, ResultsDatabase
        assert ExperimentTracker is not None
        assert ResultsDatabase is not None
        # Note: Pydantic warnings about @validator are expected during migration
    
    def test_import_utils(self):
        """Test that the utils modules can be imported."""
        from robot_lab.utils import paths, metadata, logger, run_utils
        assert paths is not None
        assert metadata is not None
        assert logger is not None
        assert run_utils is not None


class TestDependencies:
    """Test that all required dependencies are available."""
    
    def test_gymnasium_available(self):
        """Test that Gymnasium is installed and importable."""
        import gymnasium as gym
        assert gym is not None
        # Test that we can make a basic environment
        env = gym.make('CartPole-v1')
        assert env is not None
        env.close()
    
    def test_stable_baselines3_available(self):
        """Test that Stable-Baselines3 is installed."""
        from stable_baselines3 import SAC, PPO
        assert SAC is not None
        assert PPO is not None
    
    def test_torch_available(self):
        """Test that PyTorch is installed."""
        import torch
        assert torch is not None
        # Check CUDA availability (informational, not required)
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    def test_pydantic_available(self):
        """Test that Pydantic is installed."""
        import pydantic
        assert pydantic is not None
    
    def test_loguru_available(self):
        """Test that Loguru is installed."""
        from loguru import logger
        assert logger is not None


class TestConfigs:
    """Test that configuration files are properly bundled."""
    
    def test_default_configs_exist(self):
        """Test that default config files are accessible."""
        from robot_lab.config import load_hyperparameters
        
        # Try loading with fallback to default
        config = load_hyperparameters("UnknownEnv-v0", "SAC", custom_config_path=None)
        assert config is not None
        assert "algorithm" in config
        assert "num_envs" in config
        assert "total_timesteps" in config
    
    def test_mountaincar_config_exists(self):
        """Test that MountainCar config is available."""
        from robot_lab.config import load_hyperparameters
        
        config = load_hyperparameters("MountainCarContinuous-v0", "SAC", custom_config_path=None)
        assert config is not None
        assert config["algorithm"] == "SAC"
        assert config["total_timesteps"] > 0


class TestEnvironments:
    """Test that custom environments are properly registered."""
    
    def test_custom_envs_registered(self):
        """Test that custom environments are registered with Gymnasium."""
        import gymnasium as gym
        from gymnasium.envs.registration import registry
        
        # Import package to trigger registration
        import robot_lab.envs
        
        # Check that custom envs are available
        env_ids = [env_spec.id for env_spec in registry.values()]
        
        # GripperEnv should always be registered
        assert 'GripperEnv-v0' in env_ids, "GripperEnv-v0 not registered"
    
    def test_gripper_env_creation(self):
        """Test that GripperEnv can be instantiated."""
        import gymnasium as gym
        import robot_lab.envs  # Trigger registration
        
        # Note: GripperEnv requires gripper.xml file which may not be available
        # This test just verifies the environment is registered
        try:
            env = gym.make('GripperEnv-v0')
            # If it works, test basic operations
            obs, info = env.reset()
            assert obs is not None
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            env.close()
        except ValueError as e:
            # Expected if gripper.xml not found - just verify it's the right error
            assert "gripper.xml" in str(e), f"Unexpected error: {e}"
