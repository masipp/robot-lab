"""Policy visualization functionality for robot_lab."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_lab.envs import get_env_registry
from robot_lab.utils.metadata import save_visualization_metadata, update_visualization_results
from robot_lab.utils.paths import get_logs_dir, get_models_dir
from robot_lab.utils.run_utils import cleanup_old_runs, generate_run_id

console = Console()


def visualize(
    env_name: str,
    algorithm: str,
    model_path: str,
    vecnorm_path: str,
    output_dir: Optional[str] = None,
    num_episodes: int = 1,
    record_video: bool = True,
) -> Optional[Path]:
    """Run a trained policy and optionally record an MP4 video.

    Args:
        env_name: Gymnasium environment name (e.g. "MountainCarContinuous-v0").
        algorithm: Algorithm name ("SAC" or "PPO").
        model_path: Path to the saved model zip.
        vecnorm_path: Path to the VecNormalize stats pkl.
        output_dir: Root directory for output (defaults to cwd).
        num_episodes: Number of episodes to record.
        record_video: When False, runs without recording and returns None.

    Returns:
        Path to the MP4 file, or None when record_video is False.

    Raises:
        ValueError: If vecnorm_path does not exist.
    """
    import imageio.v2 as iio

    vecnorm_path_obj = Path(vecnorm_path)
    model_path_obj = Path(model_path)

    if not vecnorm_path_obj.exists():
        raise ValueError(
            f"[Visualize] VecNorm stats file not found at {vecnorm_path_obj}. "
            "Ensure model and VecNorm were saved as a pair."
        )

    env_base_name = env_name.split("-")[0].lower()
    algo_name = algorithm.lower()

    base_out = Path(output_dir) if output_dir else Path.cwd()
    video_dir = base_out / "videos" / f"{algo_name}_{env_base_name}"
    video_dir.mkdir(parents=True, exist_ok=True)

    base_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = DummyVecEnv([lambda env=base_env: env])
    eval_env = VecNormalize.load(str(vecnorm_path_obj), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    algo_cls = SAC if algo_name == "sac" else PPO
    model = algo_cls.load(str(model_path_obj), env=eval_env)

    frames: List[Any] = []
    obs = eval_env.reset()
    if record_video:
        frame = base_env.render()
        if frame is not None:
            frames.append(frame)

    episodes_done = 0
    while episodes_done < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        if record_video:
            frame = base_env.render()
            if frame is not None:
                frames.append(frame)
        if dones[0]:
            episodes_done += 1

    eval_env.close()

    if not record_video or not frames:
        return None

    video_path = video_dir / f"{algo_name}_{env_base_name}.mp4"
    iio.mimsave(str(video_path), frames, fps=30)
    return video_path


def load_training_evaluations(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation results from training run.
    
    Args:
        run_dir: Directory containing training run data
    
    Returns:
        Dictionary with evaluation metrics or None if not found
    """
    eval_file = run_dir / "evaluations.npz"
    if not eval_file.exists():
        return None
    
    try:
        data = np.load(str(eval_file))
        # Get final evaluation results
        timesteps = data['timesteps']
        results = data['results']
        ep_lengths = data['ep_lengths']
        
        # Calculate statistics from last evaluation
        last_rewards = results[-1]  # Last evaluation episode rewards
        last_lengths = ep_lengths[-1]  # Last evaluation episode lengths
        
        return {
            'mean_reward': np.mean(last_rewards),
            'std_reward': np.std(last_rewards),
            'min_reward': np.min(last_rewards),
            'max_reward': np.max(last_rewards),
            'mean_length': np.mean(last_lengths),
            'std_length': np.std(last_lengths),
            'total_timesteps': timesteps[-1],
            'num_evaluations': len(timesteps),
        }
    except Exception as e:
        logger.warning(f"Failed to load evaluations.npz: {e}")
        return None


def visualize_policy(
    env_name: str,
    algorithm: str,
    model_path: Optional[str] = None,
    vecnorm_path: Optional[str] = None,
    env_config_path: Optional[str] = None,
    num_episodes: int = 3,
    render: bool = True,
    save_plot: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[List[float], List[int], Optional[Dict[str, Any]]]:
    """Visualize a trained policy.
    
    Args:
        env_name: Name of the gymnasium environment
        algorithm: Algorithm used ("SAC" or "PPO")
        model_path: Path to the trained model (None = auto-detect)
        vecnorm_path: Path to VecNormalize stats (None = auto-detect)
        env_config_path: Path to environment configuration YAML (control params, physics, etc.)
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        save_plot: Whether to save performance plots
        output_dir: Optional custom output directory
    
    Returns:
        Tuple of (episode_rewards, episode_lengths, training_eval_metrics)
    """
    # Extract environment base name
    env_base_name = env_name.split('-')[0].lower()
    algo_name = algorithm.lower()
    
    # Generate run ID for output files
    run_id = generate_run_id(suffix=f"{algo_name}_{env_base_name}_viz")
    
    # Create run directory for this visualization
    logs_dir = get_logs_dir(output_dir)
    run_dir = logs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure run-specific log file
    log_file = run_dir / "visualization.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )
    logger.info(f"Run-specific log file: {log_file}")
    
    # Clean up old visualization runs (older than 14 days)
    logger.info("Checking for old visualization runs to clean up...")
    cleanup_old_runs(
        directory=logs_dir,
        max_age_days=14,
        pattern="*_viz",
        dry_run=False
    )
    
    # Auto-detect paths if not provided
    training_run_dir = None  # Track the training run directory for loading evaluations
    
    if model_path is None:
        # First check models directory for manually saved models
        models_dir = get_models_dir(output_dir)
        default_model = models_dir / f"{algo_name}_{env_base_name}.zip"
        
        if default_model.exists():
            model_path = default_model
        else:
            # Look for most recent run in logs directory
            matching_runs = sorted(
                [d for d in logs_dir.glob(f"*_{algo_name}_{env_base_name}")
                 if d.is_dir() and not d.name.endswith('_viz')],
                reverse=True
            )
            if matching_runs:
                training_run_dir = matching_runs[0]  # Store for loading evaluations
                model_path = training_run_dir / f"{algo_name}_{env_base_name}.zip"
                logger.info(f"Auto-detected model from latest run: {training_run_dir.name}")
            else:
                raise FileNotFoundError(
                    "No model found. Train a model first or specify --model-path"
                )
    else:
        # If model path was provided, try to infer the training run directory
        model_path_obj = Path(model_path)
        if model_path_obj.parent.name.startswith('2026'):
            training_run_dir = model_path_obj.parent
    
    if vecnorm_path is None:
        # Infer vecnorm path from model path location
        model_path_obj = Path(model_path)
        vecnorm_path = model_path_obj.parent / f"{algo_name}_{env_base_name}_vecnorm.pkl"
    
    model_path = Path(model_path)
    vecnorm_path = Path(vecnorm_path)
    
    logger.info("="*60)
    logger.info("Visualizing the learned policy...")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Model: {model_path}")
    logger.info(f"VecNormalize: {vecnorm_path}")
    logger.info("="*60)
    
    # Save metadata
    metadata_path = save_visualization_metadata(
        run_dir=run_dir,
        run_id=run_id,
        env_name=env_name,
        algorithm=algorithm,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        num_episodes=num_episodes,
        render=render,
        save_plot=save_plot,
        output_dir=output_dir,
    )
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Load environment configuration if provided
    env_kwargs = {}
    if env_config_path:
        import yaml
        logger.info(f"Loading environment configuration from {env_config_path}")
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        if env_config:
            env_kwargs = env_config
            logger.info(f"Loaded env config with keys: {list(env_config.keys())}")
    
    # Ensure custom environments are registered
    registry = get_env_registry()
    registry.register_custom_envs()
    
    # Create evaluation environment
    render_mode = "human" if render else None
    base_env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
    eval_env = DummyVecEnv([lambda: base_env])
    
    # Load VecNormalize statistics
    if vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        # Important: set training to False and don't update stats during evaluation
        eval_env.training = False
        eval_env.norm_reward = False
        logger.success(f"Loaded VecNormalize statistics from {vecnorm_path}")
    else:
        logger.warning(f"Could not find {vecnorm_path}")
        logger.warning("Evaluation will use unnormalized observations (may perform poorly!)")
    
    # Get reference to the unwrapped environment for direct rendering
    # This avoids issues with VecEnv wrapper and MuJoCo passive viewer
    unwrapped_env = base_env
    
    # Load the trained model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if algorithm.upper() == "SAC":
        model = SAC.load(str(model_path))
    elif algorithm.upper() == "PPO":
        model = PPO.load(str(model_path))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    logger.success(f"Loaded model from {model_path}")
    
    # Run episodes and collect data
    episode_rewards = []
    episode_lengths = []
    
    env_display_name = env_name.split('-')[0]
    logger.info(f"Watch the {env_display_name} perform!")
    if render:
        logger.info("Close the window to continue to the next episode...")
    
    for episode in range(num_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        logger.info(f"Starting Episode {episode + 1}/{num_episodes}...")
        
        while not done:
            # Render using the unwrapped environment directly
            # This avoids issues with VecEnv wrapper and MuJoCo passive viewer
            if render:
                unwrapped_env.render()
            
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            episode_reward += reward[0]  # Extract scalar from array
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        logger.info(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}")
    
    eval_env.close()
    
    # Load training evaluation metrics if available
    training_eval = None
    if training_run_dir:
        training_eval = load_training_evaluations(training_run_dir)
        if training_eval:
            logger.info(f"Loaded training evaluation data from {training_run_dir.name}")
    
    # Print summary statistics
    logger.info("="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Average episode length: {np.mean(episode_lengths):.2f} ± "
          f"{np.std(episode_lengths):.2f}")
    logger.info(f"Min reward: {np.min(episode_rewards):.2f}")
    logger.info(f"Max reward: {np.max(episode_rewards):.2f}")
    logger.info("="*60)
    
    # Display comparison table with training evaluation
    _display_comparison_table(
        vis_rewards=episode_rewards,
        vis_lengths=episode_lengths,
        training_eval=training_eval,
        env_name=env_name,
        algorithm=algorithm
    )
    
    # Plot the results
    if save_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot episode rewards
        ax1.bar(range(1, num_episodes + 1), episode_rewards, color='blue', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards (Learned Policy)')
        ax1.axhline(
            y=np.mean(episode_rewards),
            color='r',
            linestyle='--',
            label=f'Mean: {np.mean(episode_rewards):.1f}'
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax2.bar(range(1, num_episodes + 1), episode_lengths, color='green', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths (Learned Policy)')
        ax2.axhline(
            y=np.mean(episode_lengths),
            color='r',
            linestyle='--',
            label=f'Mean: {np.mean(episode_lengths):.1f}'
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure to run directory
        output_file = run_dir / "results.png"
        
        plt.savefig(output_file, dpi=150)
        logger.success(f"Visualization saved to {output_file}")
        
        plt.show()
    
    # Update metadata with results
    update_visualization_results(
        run_dir=run_dir,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )
    logger.info("Updated metadata with results")
    
    return episode_rewards, episode_lengths, training_eval


def _display_comparison_table(
    vis_rewards: List[float],
    vis_lengths: List[int],
    training_eval: Optional[Dict[str, Any]],
    env_name: str,
    algorithm: str,
) -> None:
    """Display comparison table between visualization and training evaluation.
    
    Args:
        vis_rewards: Visualization episode rewards
        vis_lengths: Visualization episode lengths
        training_eval: Training evaluation metrics (or None)
        env_name: Environment name
        algorithm: Algorithm name
    """
    table = Table(title=f"\n{env_name} - {algorithm.upper()} Policy Performance", show_header=True)
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Visualization\n(Now)", style="magenta", justify="right")
    
    if training_eval:
        table.add_column("Training Eval\n(Final)", style="green", justify="right")
        table.add_column("Difference", style="yellow", justify="right")
    
    # Reward metrics
    vis_mean_reward = np.mean(vis_rewards)
    vis_std_reward = np.std(vis_rewards)
    
    if training_eval:
        train_mean = training_eval['mean_reward']
        diff = vis_mean_reward - train_mean
        diff_pct = (diff / abs(train_mean) * 100) if train_mean != 0 else 0
        
        table.add_row(
            "Mean Reward",
            f"{vis_mean_reward:.2f}",
            f"{train_mean:.2f}",
            f"{diff:+.2f} ({diff_pct:+.1f}%)"
        )
        table.add_row(
            "Std Reward",
            f"±{vis_std_reward:.2f}",
            f"±{training_eval['std_reward']:.2f}",
            "—"
        )
        table.add_row(
            "Min Reward",
            f"{np.min(vis_rewards):.2f}",
            f"{training_eval['min_reward']:.2f}",
            "—"
        )
        table.add_row(
            "Max Reward",
            f"{np.max(vis_rewards):.2f}",
            f"{training_eval['max_reward']:.2f}",
            "—"
        )
        
        # Length metrics
        vis_mean_length = np.mean(vis_lengths)
        table.add_row(
            "Mean Episode Length",
            f"{vis_mean_length:.1f}",
            f"{training_eval['mean_length']:.1f}",
            f"{vis_mean_length - training_eval['mean_length']:+.1f}"
        )
        
        # Training info
        table.add_row("─" * 20, "─" * 15, "─" * 15, "─" * 15)
        table.add_row(
            "Training Timesteps",
            "—",
            f"{training_eval['total_timesteps']:,}",
            "—"
        )
        table.add_row(
            "Num Evaluations",
            f"{len(vis_rewards)}",
            f"{training_eval['num_evaluations']}",
            "—"
        )
    else:
        table.add_row("Mean Reward", f"{vis_mean_reward:.2f}")
        table.add_row("Std Reward", f"±{vis_std_reward:.2f}")
        table.add_row("Min Reward", f"{np.min(vis_rewards):.2f}")
        table.add_row("Max Reward", f"{np.max(vis_rewards):.2f}")
        table.add_row("Mean Episode Length", f"{np.mean(vis_lengths):.1f}")
        table.add_row("Num Episodes", f"{len(vis_rewards)}")
    
    console.print()
    console.print(table)
    console.print()
    
    if training_eval:
        # Show interpretation
        diff = vis_mean_reward - training_eval['mean_reward']
        if abs(diff) < vis_std_reward:
            console.print("[green]✓ Visualization performance matches training evaluation (within std)[/green]")
        elif diff > 0:
            console.print(f"[cyan]↑ Visualization shows {diff:.2f} better reward than training eval[/cyan]")
        else:
            console.print(f"[yellow]↓ Visualization shows {-diff:.2f} worse reward than training eval[/yellow]")
            console.print("[yellow]  (This is normal due to stochasticity and sample size)[/yellow]")
