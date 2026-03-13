"""Training tests to verify the RL training pipeline works correctly."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from robot_lab.training import train


class TestTrainingBasics:
    """Test basic training functionality."""
    
    def test_train_returns_expected_outputs(self, temp_output_dir, test_seed):
        """Test that train() returns model, model_path, and vecnorm_path."""
        # Use very short training for speed
        model, model_path, vecnorm_path = train(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            seed=test_seed,
            output_dir=str(temp_output_dir),
            eval_freq=5000,  # Don't eval often in tests
            eval_episodes=3,  # Few eval episodes for speed
            use_checkpoints=False
        )
        
        # Check that model is returned
        assert model is not None
        assert isinstance(model, SAC)
        
        # Check that paths are returned and exist
        assert model_path is not None
        assert vecnorm_path is not None
        assert model_path.exists(), f"Model file not created: {model_path}"
        assert vecnorm_path.exists(), f"VecNormalize file not created: {vecnorm_path}"
    
    def test_train_creates_output_structure(self, temp_output_dir, test_seed):
        """Test that training creates expected directory structure."""
        train(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            seed=test_seed,
            output_dir=str(temp_output_dir),
            eval_freq=5000,
            eval_episodes=3,
            use_checkpoints=False
        )
        
        # Check directory structure
        assert (temp_output_dir / "logs").exists(), "logs/ directory not created"
        assert (temp_output_dir / "models").exists(), "models/ directory not created"
        assert (temp_output_dir / "tensorboard").exists(), "tensorboard/ directory not created"
        
        # Check that at least one run directory exists in logs
        log_dirs = list((temp_output_dir / "logs").iterdir())
        assert len(log_dirs) > 0, "No run directory created in logs/"
        
        # Check run directory contents
        run_dir = log_dirs[0]
        assert (run_dir / "metadata.json").exists(), "metadata.json not created"
        assert (run_dir / "training.log").exists(), "training.log not created"


class TestMountainCarTraining:
    """Test training on MountainCarContinuous-v0 to verify learning works."""
    
    @pytest.mark.slow
    def test_mountaincar_learns_to_improve(self, temp_output_dir, test_seed):
        """Test that SAC improves mean reward on MountainCarContinuous beyond threshold.
        
        This is the key test that verifies the training pipeline actually works.
        MountainCarContinuous-v0 is solved at 90.0+ average reward.
        We train for a moderate amount and check if performance exceeds a threshold.
        """
        # Train with config from mountaincarcontinuous_sac.json (20k timesteps default)
        # This should be enough to show some learning
        model, model_path, vecnorm_path = train(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            seed=test_seed,
            output_dir=str(temp_output_dir),
            eval_freq=5000,
            eval_episodes=10,  # More episodes for better estimate
            use_checkpoints=False
        )
        
        # Evaluate the trained model
        mean_reward, std_reward = self._evaluate_model(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            env_name="MountainCarContinuous-v0",
            n_eval_episodes=20,
            seed=test_seed + 999
        )
        
        print(f"\nMountainCar evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Threshold for success: We expect at least -50 reward after 20k steps
        # Random policy gets around -200 to -100
        # Partially trained should be better than -50
        # Fully trained achieves 90+
        THRESHOLD = -50.0
        
        assert mean_reward > THRESHOLD, (
            f"Training did not improve enough. Mean reward: {mean_reward:.2f}, "
            f"Threshold: {THRESHOLD}. Expected reward > {THRESHOLD} after training."
        )
        
        print(f"✓ Training successful: mean reward {mean_reward:.2f} > {THRESHOLD}")
    
    def test_mountaincar_short_training(self, temp_output_dir, test_seed):
        """Test a very short training run completes without errors (fast test)."""
        # Override config to use very few timesteps for speed
        import json
        config_file = temp_output_dir / "test_config.json"
        
        # Create minimal config for fast testing
        config = {
            "environment": "MountainCarContinuous-v0",
            "algorithm": "SAC",
            "num_envs": 2,
            "total_timesteps": 5000,  # Very short for fast test
            "hyperparameters": {
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
                "buffer_size": 50000,
                "learning_starts": 500,
                "batch_size": 64,
                "tau": 0.02,
                "gamma": 0.99,
                "train_freq": 8,
                "gradient_steps": 8
            },
            "vec_normalize": {
                "norm_obs": True,
                "norm_reward": True,
                "clip_obs": 10.0,
                "clip_reward": 10.0,
                "gamma": 0.99
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Train with custom config
        model, model_path, vecnorm_path = train(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            config_path=str(config_file),
            seed=test_seed,
            output_dir=str(temp_output_dir),
            eval_freq=10000,  # No eval during short training
            eval_episodes=3,
            use_checkpoints=False
        )
        
        assert model is not None
        assert model_path.exists()
        assert vecnorm_path.exists()
        
        print("✓ Short training completed successfully")
    
    @staticmethod
    def _evaluate_model(
        model_path: Path,
        vecnorm_path: Path,
        env_name: str,
        n_eval_episodes: int = 10,
        seed: int = 0
    ) -> tuple[float, float]:
        """Evaluate a trained model and return mean and std reward.
        
        Args:
            model_path: Path to saved model
            vecnorm_path: Path to saved VecNormalize stats
            env_name: Name of environment
            n_eval_episodes: Number of episodes to evaluate
            seed: Random seed
        
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Load model
        model = SAC.load(str(model_path))
        
        # Create evaluation environment
        env = DummyVecEnv([lambda: gym.make(env_name)])
        
        # Load normalization stats
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False  # Don't normalize rewards during evaluation
        
        # Run evaluation episodes
        episode_rewards = []
        
        for episode in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
        
        env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward


class TestModelPersistence:
    """Test that models can be saved and loaded correctly."""
    
    def test_model_can_be_loaded(self, temp_output_dir, test_seed):
        """Test that saved model can be loaded and used for prediction."""
        # Train a model
        _, model_path, vecnorm_path = train(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            seed=test_seed,
            output_dir=str(temp_output_dir),
            eval_freq=10000,
            eval_episodes=3,
            use_checkpoints=False
        )
        
        # Load the model
        loaded_model = SAC.load(str(model_path))
        assert loaded_model is not None
        
        # Load VecNormalize stats
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
        env = VecNormalize.load(str(vecnorm_path), env)
        
        # Test prediction
        obs = env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        
        assert action is not None
        assert action.shape == (1, 1)  # MountainCar has 1D action space
        
        env.close()
        print("✓ Model loaded and used for prediction successfully")


class TestEpic1Pipeline:
    """End-to-end integration: all Epic 1 components wired together."""

    def test_full_pipeline_artifacts(self, temp_output_dir: Path, test_seed: int) -> None:
        """Short training run must produce all Epic 1 artifacts.

        Verifies: ExperimentTracker metadata.json (5 sections), experiment_summary.md,
        results_index.jsonl, and best_model.zip + best_model_vecnorm.pkl pair.
        """
        import json as _json

        train(
            env_name="MountainCarContinuous-v0",
            algorithm="SAC",
            seed=test_seed,
            output_dir=str(temp_output_dir),
            eval_freq=5000,
            eval_episodes=3,
            use_checkpoints=False,
            total_timesteps=10000,
        )

        # --- ExperimentTracker artifacts ---
        exp_runs = list((temp_output_dir / "experiments").glob("*/runs/*"))
        assert len(exp_runs) >= 1, "ExperimentTracker run directory not created"
        tracker_run_dir = exp_runs[0]

        metadata_path = tracker_run_dir / "metadata.json"
        assert metadata_path.exists(), "metadata.json not written by ExperimentTracker"

        metadata = _json.loads(metadata_path.read_text(encoding="utf-8"))
        for section in ("run", "config", "system", "metrics", "custom"):
            assert section in metadata, f"metadata.json missing section: {section}"
        assert metadata["run"]["status"] == "COMPLETED"

        # --- experiment_summary.md ---
        summary_path = tracker_run_dir / "experiment_summary.md"
        assert summary_path.exists(), "experiment_summary.md not written"

        # --- results_index.jsonl ---
        index_path = temp_output_dir / "experiments" / "results_index.jsonl"
        assert index_path.exists(), "results_index.jsonl not created"
        lines = index_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1, "results_index.jsonl is empty"
        record = _json.loads(lines[0])
        assert record["status"] == "COMPLETED"

        # --- Atomic best-model pair (from RobotLabEvalCallback) ---
        best_dir = temp_output_dir / "models" / "best"
        assert (best_dir / "best_model.zip").exists(), "best_model.zip not saved"
        assert (best_dir / "best_model_vecnorm.pkl").exists(), "best_model_vecnorm.pkl not saved"

        print("✓ All Epic 1 pipeline artifacts verified")

