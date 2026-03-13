"""Core training functionality for robot_lab."""

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from robot_lab.config import load_hyperparameters
from robot_lab.experiments.tracker import ExperimentTracker
from robot_lab.utils.callbacks import RobotLabCheckpointCallback, RobotLabEvalCallback
from robot_lab.utils.metadata import append_final_metrics, save_training_metadata
from robot_lab.utils.paths import get_logs_dir, get_models_dir, get_tensorboard_dir
from robot_lab.utils.run_utils import cleanup_old_runs, generate_run_id


def train(
    env_name: str,
    algorithm: str,
    config_path: Optional[str] = None,
    env_config_path: Optional[str] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
    eval_freq: int = 10000,
    eval_episodes: int = 10,
    save_freq: Optional[int] = None,
    use_checkpoints: bool = False,
    # Optional direct parameter overrides (for experiment runner)
    total_timesteps: Optional[int] = None,
    num_envs: Optional[int] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    env_wrapper_fn: Optional[Any] = None,
) -> Tuple[Any, Path, Path]:
    """Train a reinforcement learning agent.
    
    Args:
        env_name: Name of the gymnasium environment
        algorithm: Algorithm to use ("SAC" or "PPO")
        config_path: Optional path to custom configuration file
        env_config_path: Optional path to environment configuration YAML
            (control params, physics, etc.)
        seed: Random seed for reproducibility
        output_dir: Optional custom output directory
        eval_freq: Frequency (in timesteps) of evaluation
        eval_episodes: Number of episodes for evaluation
        save_freq: Frequency (in timesteps) to save checkpoints (None = disabled)
        use_checkpoints: Whether to save intermediate checkpoints
        total_timesteps: Override total_timesteps from config (for experiment runner)
        num_envs: Override num_envs from config (for experiment runner)
        env_kwargs: Override env_kwargs from env_config_path (for experiment runner)
        env_wrapper_fn: Optional function to wrap environments (called after gym.make)
    
    Returns:
        Tuple of (trained_model, model_path, vecnorm_path)
    """
    # Set up output directories
    models_dir = get_models_dir(output_dir)
    logs_dir = get_logs_dir(output_dir)
    tensorboard_dir = get_tensorboard_dir(output_dir)
    
    # Clean up old logs and tensorboard runs (older than 14 days)
    logger.info("Checking for old logs to clean up...")
    deleted_count, freed_mb = cleanup_old_runs(
        directory=logs_dir,
        max_age_days=14,
        pattern="*",
        dry_run=False
    )
    cleanup_old_runs(
        directory=tensorboard_dir,
        max_age_days=14,
        pattern="*",
        dry_run=False
    )
    
    # Load hyperparameters
    config = load_hyperparameters(env_name, algorithm, config_path)
    
    # Load environment configuration if provided
    env_kwargs_final = env_kwargs if env_kwargs is not None else {}
    if env_config_path:
        import yaml
        logger.info(f"Loading environment configuration from {env_config_path}")
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        # Pass configuration as kwargs to gym.make
        # Note: This requires the environment to accept these parameters
        if env_config:
            env_kwargs_final.update(env_config)
            logger.info(f"Loaded env config with keys: {list(env_config.keys())}")
    
    # Apply parameter overrides (for experiment runner)
    num_envs_final = num_envs if num_envs is not None else config["num_envs"]
    total_timesteps_final = (
        total_timesteps if total_timesteps is not None else config["total_timesteps"]
    )
    
    vec_norm_config = config["vec_normalize"]
    hyperparams = config["hyperparameters"]
    
    # Extract environment base name for file naming
    env_base_name = env_name.split('-')[0].lower()
    algo_name = algorithm.lower()
    
    # Generate unique run ID with timestamp and hash
    run_id = generate_run_id(suffix=f"{algo_name}_{env_base_name}")
    
    # Create test environment to check spaces
    test_env = gym.make(env_name, **env_kwargs_final)
    
    # Apply wrapper function if provided (for experiment runner)
    if env_wrapper_fn is not None:
        test_env = env_wrapper_fn(test_env)
    
    logger.info("="*60)
    logger.info(f"Environment: {env_name}")
    logger.info(f"Observation space: {test_env.observation_space}")
    logger.info(f"Action space: {test_env.action_space}")
    logger.info("="*60)
    test_env.close()
    
    # Create vectorized environments
    logger.info(f"Creating {num_envs_final} parallel environments...")
    
    # Make environment factory with wrapper support
    def make_env_with_wrapper(env_name, rank, seed, **kwargs):
        def _init():
            env = gym.make(env_name, **kwargs)
            if env_wrapper_fn is not None:
                env = env_wrapper_fn(env)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    envs = SubprocVecEnv(
        [
            make_env_with_wrapper(env_name, i, seed, **env_kwargs_final)
            for i in range(num_envs_final)
        ]
    )
    
    # Apply normalization
    logger.info("Applying observation and reward normalization...")
    envs = VecNormalize(
        envs,
        norm_obs=vec_norm_config["norm_obs"],
        norm_reward=vec_norm_config["norm_reward"],
        clip_obs=vec_norm_config["clip_obs"],
        clip_reward=vec_norm_config.get("clip_reward", 10.0),
        gamma=vec_norm_config.get("gamma", 0.99)
    )
    
    # Set up model paths with run ID
    run_dir = logs_dir / run_id  # Main run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models in run directory
    model_path = run_dir / f"{algo_name}_{env_base_name}.zip"
    vecnorm_path = run_dir / f"{algo_name}_{env_base_name}_vecnorm.pkl"
    
    # Tensorboard in separate directory
    tensorboard_path = tensorboard_dir / run_id
    
    # Configure run-specific log file
    log_file = run_dir / "training.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
    )
    logger.info(f"Run-specific log file: {log_file}")
    
    # Save metadata
    metadata_path = save_training_metadata(
        run_dir=run_dir,
        run_id=run_id,
        env_name=env_name,
        algorithm=algorithm,
        config=config,
        seed=seed,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        save_freq=save_freq,
        use_checkpoints=use_checkpoints,
        output_dir=output_dir,
        actual_total_timesteps=total_timesteps_final,
        actual_num_envs=num_envs_final,
    )
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Detect and configure device (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("="*60)
    logger.info("DEVICE CONFIGURATION")
    if device == "cuda":
        logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  PyTorch CUDA: {torch.version.cuda}")
        logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
        logger.success("Training will use GPU acceleration")
    else:
        logger.warning("⚠ No GPU detected - training will use CPU")
        logger.warning("  For GPU support:")
        logger.warning("  1. Install NVIDIA drivers: sudo ubuntu-drivers autoinstall")
        logger.warning("  2. Reboot system")
        logger.warning("  3. Verify with: nvidia-smi")
    logger.info("="*60)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env_with_wrapper(env_name, 0, seed + 1000, **env_kwargs_final)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=vec_norm_config["norm_obs"],
        norm_reward=False,  # Don't normalize reward during eval
        clip_obs=vec_norm_config["clip_obs"],
        training=False
    )
    
    # Set up callbacks
    callbacks = []

    # Evaluation callback — saves best_model.zip + best_model_vecnorm.pkl as an atomic pair
    eval_callback = RobotLabEvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(run_dir),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Atomic checkpoint callback — saves .zip + _vecnorm.pkl together every 10 minutes
    time_checkpoint_dir = models_dir / "checkpoints" / run_id
    checkpoint_callback = RobotLabCheckpointCallback(
        save_freq_seconds=600,
        save_path=str(time_checkpoint_dir),
        name_prefix=f"{algo_name}_{env_base_name}",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Timestep-based atomic checkpoint (if enabled)
    if use_checkpoints and save_freq:
        timestep_ckpt = RobotLabCheckpointCallback(
            save_freq_seconds=0,
            save_path=str(models_dir / "checkpoints" / "timestep_based"),
            name_prefix=f"{algo_name}_{env_base_name}",
        )
        # Override interval check to fire on timestep cadence instead of wall-clock
        def _timestep_on_step(cb=timestep_ckpt, freq=save_freq):
            if cb.num_timesteps % freq == 0:
                cb._save_atomic_pair()
            return True

        timestep_ckpt._on_step = _timestep_on_step
        callbacks.append(timestep_ckpt)

    callback_list = CallbackList(callbacks)
    
    # Create and train model
    if algorithm.upper() == "SAC":
        logger.info(f"Training with SAC on {env_name}...")
        model = SAC(
            env=envs,
            verbose=1,
            tensorboard_log=str(tensorboard_path),
            seed=seed,
            device=device,
            **hyperparams
        )
    elif algorithm.upper() == "PPO":
        logger.info(f"Training with PPO on {env_name}...")
        model = PPO(
            env=envs,
            verbose=1,
            tensorboard_log=str(tensorboard_path),
            seed=seed,
            device=device,
            **hyperparams
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'SAC' or 'PPO'.")
    
    logger.info(f"Starting training for {total_timesteps_final:,} timesteps...")
    logger.info(f"Seed: {seed}")
    logger.info(
        f"Callbacks: Eval every {eval_freq} steps | Time-based checkpoints every 10 minutes"
    )
    if use_checkpoints and save_freq:
        logger.info(f"  Additional timestep-based checkpoints every {save_freq:,} steps")

    # Set global seeds for reproducibility before any training steps
    random.seed(seed)
    np.random.seed(seed)

    # Start experiment tracker (metadata.json in experiments/<name>/runs/<run_id>/)
    tracker = ExperimentTracker(
        experiment_name=f"{algo_name}_{env_base_name}",
        run_name=run_id,
        seed=seed,
        phase=0,
        config_snapshot=config,
        output_dir=output_dir,
    )
    tracker.start_run()

    # Train
    try:
        model.learn(total_timesteps=total_timesteps_final, callback=callback_list)
        tracker.end_run("COMPLETED")
    except KeyboardInterrupt:
        tracker.end_run("INTERRUPTED")
        raise
    except Exception:
        tracker.end_run("FAILED")
        raise

    # Save final model and normalization stats
    logger.info(f"Saving model to {model_path}...")
    model.save(str(model_path))
    
    logger.info(f"Saving normalization stats to {vecnorm_path}...")
    envs.save(str(vecnorm_path))
    
    # Append final evaluation metrics to metadata
    logger.info("Extracting final evaluation metrics...")
    append_final_metrics(run_dir)
    
    logger.info("="*60)
    logger.success("Training completed!")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"  - Model: {model_path.name}")
    logger.info(f"  - VecNormalize: {vecnorm_path.name}")
    logger.info("  - Training log: training.log")
    logger.info("  - Evaluation: evaluations.npz")
    logger.info(f"TensorBoard: {tensorboard_path}")
    logger.info(f"Best model: {models_dir / 'best'}")
    logger.info(f"Checkpoints (10 min intervals): {time_checkpoint_dir}")
    if use_checkpoints and save_freq:
        logger.info(f"Timestep-based checkpoints: {models_dir / 'checkpoints' / 'timestep_based'}")
    logger.info("="*60)
    logger.info("💡 To save this model permanently, copy files from run directory to data/models/")
    logger.info("")
    visualization_command = (
        f"robot-lab visualize --env {env_name} --algo {algorithm} "
        f"--model-path {model_path} --vecnorm-path {vecnorm_path}"
    )
    logger.info("🎬 To visualize the trained policy, run:")
    logger.info(f"   {visualization_command}")
    
    # Clean up
    envs.close()
    eval_env.close()
    
    return model, model_path, vecnorm_path
