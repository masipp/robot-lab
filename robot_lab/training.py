"""Core training functionality for robot_lab."""

import gymnasium as gym
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from loguru import logger

from robot_lab.config import load_hyperparameters
from robot_lab.utils.paths import get_models_dir, get_logs_dir, get_tensorboard_dir
from robot_lab.utils.run_utils import generate_run_id, cleanup_old_runs
from robot_lab.utils.metadata import save_training_metadata, append_final_metrics
from robot_lab.utils.callbacks import TimeBasedCheckpointCallback, VecNormalizeSaveCallback
from robot_lab.envs import make_env


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
        env_config_path: Optional path to environment configuration YAML (control params, physics, etc.)
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
    total_timesteps_final = total_timesteps if total_timesteps is not None else config["total_timesteps"]
    
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
    
    envs = SubprocVecEnv([make_env_with_wrapper(env_name, i, seed, **env_kwargs_final) for i in range(num_envs_final)])
    
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
        logger.success(f"Training will use GPU acceleration")
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
    
    # Evaluation callback - save to run directory
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(run_dir),  # Save eval results to run directory
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Time-based checkpoint callback - saves every 10 minutes
    time_checkpoint_dir = models_dir / "checkpoints" / run_id
    time_checkpoint_callback = TimeBasedCheckpointCallback(
        save_freq_seconds=600,  # 10 minutes
        save_path=str(time_checkpoint_dir),
        name_prefix=f"{algo_name}_{env_base_name}",
        verbose=1
    )
    callbacks.append(time_checkpoint_callback)
    
    # Save VecNormalize alongside model checkpoints
    vecnorm_checkpoint_callback = VecNormalizeSaveCallback(
        save_path=str(time_checkpoint_dir),
        name_prefix=f"{algo_name}_{env_base_name}_vecnorm",
        save_freq_seconds=600,  # 10 minutes, synchronized with model checkpoints
        verbose=1
    )
    callbacks.append(vecnorm_checkpoint_callback)
    
    # Timestep-based checkpoint callback (if enabled)
    if use_checkpoints and save_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(models_dir / "checkpoints" / "timestep_based"),
            name_prefix=f"{algo_name}_{env_base_name}"
        )
        callbacks.append(checkpoint_callback)
    
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
    logger.info(f"Callbacks: Eval every {eval_freq} steps | Time-based checkpoints every 10 minutes")
    if use_checkpoints and save_freq:
        logger.info(f"  Additional timestep-based checkpoints every {save_freq:,} steps")
    
    # Train
    model.learn(total_timesteps=total_timesteps_final, callback=callback_list)
    
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
    logger.info(f"  - Training log: training.log")
    logger.info(f"  - Evaluation: evaluations.npz")
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
