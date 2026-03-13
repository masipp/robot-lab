"""Smoke tests to verify basic package functionality."""


import pytest


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
        from robot_lab.utils import logger, metadata, paths, run_utils
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
        from stable_baselines3 import PPO, SAC
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
        from gymnasium.envs.registration import registry
        
        # Import package to trigger registration
        
        # Check that custom envs are available
        env_ids = [env_spec.id for env_spec in registry.values()]
        
        # GripperEnv should always be registered
        assert 'GripperEnv-v0' in env_ids, "GripperEnv-v0 not registered"
    
    def test_gripper_env_creation(self):
        """Test that GripperEnv can be instantiated."""
        import gymnasium as gym

        
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


# ---------------------------------------------------------------------------
# Story 2.3: Video Render Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestVisualizationPipeline:
    """Smoke tests for the visualize() video render pipeline (Story 2.3)."""

    def test_visualize_writes_mp4(self, temp_output_dir):
        """visualize() writes a non-zero MP4 to the experiment run dir. (AC: 1, 2)"""
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from robot_lab.visualization import visualize

        # --- Setup: create a minimal trained model and vecnorm ---
        model_path = temp_output_dir / "test_model.zip"
        vecnorm_path = temp_output_dir / "test_vecnorm.pkl"

        train_env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        model = SAC("MlpPolicy", train_env, seed=0, device="cpu", verbose=0)
        model.learn(total_timesteps=100)
        model.save(str(model_path))
        train_env.save(str(vecnorm_path))
        train_env.close()

        # --- Act ---
        result_path = visualize(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            model_path=str(model_path),
            vecnorm_path=str(vecnorm_path),
            output_dir=str(temp_output_dir),
            num_episodes=1,
            record_video=True,
        )

        # --- Assert ---
        assert result_path is not None, "visualize() must return a Path when record_video=True"
        assert result_path.exists(), f"MP4 file not found at {result_path}"
        assert result_path.suffix == ".mp4", f"Expected .mp4, got {result_path.suffix}"
        assert result_path.stat().st_size > 0, "MP4 file must have non-zero size"

    def test_visualize_raises_on_missing_vecnorm(self, temp_output_dir):
        """visualize() raises ValueError with exact message when vecnorm missing. (AC: 3)"""
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv

        from robot_lab.visualization import visualize

        model_path = temp_output_dir / "test_model.zip"
        train_env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
        model = SAC("MlpPolicy", train_env, seed=0, device="cpu", verbose=0)
        model.save(str(model_path))
        train_env.close()

        missing_vecnorm = temp_output_dir / "nonexistent_vecnorm.pkl"

        with pytest.raises(ValueError, match=r"\[Visualize\] VecNorm stats file not found at"):
            visualize(
                env_name="MountainCarContinuous-v0",
                algorithm="SAC",
                model_path=str(model_path),
                vecnorm_path=str(missing_vecnorm),
                output_dir=str(temp_output_dir),
                num_episodes=1,
            )

    def test_visualize_importable(self):
        """visualize() is importable from robot_lab.visualization."""
        from robot_lab.visualization import visualize
        assert callable(visualize)
