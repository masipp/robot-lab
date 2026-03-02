"""Utilities for saving and loading run metadata."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import platform
import sys
import numpy as np


def get_system_info() -> Dict[str, Any]:
    """Get system information for reproducibility.
    
    Returns:
        Dictionary with system information
    """
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # Add GPU/CUDA information if available
    try:
        import torch
        system_info['pytorch_version'] = torch.__version__
        system_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
        else:
            system_info['cuda_version'] = None
            system_info['gpu_count'] = 0
            system_info['gpu_name'] = None
    except ImportError:
        system_info['pytorch_version'] = None
        system_info['cuda_available'] = False
    
    return system_info


def save_training_metadata(
    run_dir: Path,
    run_id: str,
    env_name: str,
    algorithm: str,
    config: Dict[str, Any],
    seed: int,
    eval_freq: int,
    eval_episodes: int,
    save_freq: Optional[int],
    use_checkpoints: bool,
    output_dir: Optional[str] = None,
    # Actual values used (may differ from config)
    actual_total_timesteps: Optional[int] = None,
    actual_num_envs: Optional[int] = None,
) -> Path:
    """Save training run metadata as JSON.
    
    Args:
        run_dir: Directory where metadata will be saved
        run_id: Unique run identifier
        env_name: Environment name
        algorithm: Algorithm name
        config: Full configuration dictionary
        seed: Random seed
        eval_freq: Evaluation frequency
        eval_episodes: Number of evaluation episodes
        save_freq: Checkpoint save frequency
        use_checkpoints: Whether checkpoints are enabled
        output_dir: Custom output directory
        actual_total_timesteps: Actually used total timesteps (overrides config)
        actual_num_envs: Actually used num_envs (overrides config)
    
    Returns:
        Path to the saved metadata file
    """
    # Use actual values if provided, otherwise fall back to config
    total_timesteps = actual_total_timesteps if actual_total_timesteps is not None else config['total_timesteps']
    num_envs = actual_num_envs if actual_num_envs is not None else config['num_envs']
    
    metadata = {
        'type': 'training',
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'name': env_name,
            'base_name': env_name.split('-')[0].lower(),
        },
        'algorithm': algorithm,
        'training': {
            'seed': seed,
            'total_timesteps': total_timesteps,
            'num_envs': num_envs,
            'eval_freq': eval_freq,
            'eval_episodes': eval_episodes,
            'save_freq': save_freq,
            'use_checkpoints': use_checkpoints,
        },
        'hyperparameters': config['hyperparameters'],
        'vec_normalize': config['vec_normalize'],
        'output_dir': str(output_dir) if output_dir else None,
        'system': get_system_info(),
        'final_metrics': {},  # Will be populated after training
    }
    
    metadata_path = run_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def append_final_metrics(run_dir: Path, evaluations_file: Optional[Path] = None) -> None:
    """Append final evaluation metrics to metadata.json.
    
    Extracts metrics from evaluations.npz and appends them to the metadata file.
    
    Args:
        run_dir: Directory containing metadata.json
        evaluations_file: Optional path to evaluations.npz (defaults to run_dir/evaluations.npz)
    """
    metadata_path = run_dir / 'metadata.json'
    if not metadata_path.exists():
        return
    
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Find evaluations file
    if evaluations_file is None:
        evaluations_file = run_dir / 'evaluations.npz'
    
    if not evaluations_file.exists():
        return
    
    # Extract final metrics
    eval_data = np.load(evaluations_file)
    final_metrics = {}
    
    if "results" in eval_data and len(eval_data["results"]) > 0:
        final_rewards = eval_data["results"][-1]  # Last eval episode rewards
        final_metrics["ep_rew_mean"] = float(np.mean(final_rewards))
        final_metrics["ep_rew_std"] = float(np.std(final_rewards))
        final_metrics["ep_rew_max"] = float(np.max(final_rewards))
        final_metrics["ep_rew_min"] = float(np.min(final_rewards))
    
    if "ep_lengths" in eval_data and len(eval_data["ep_lengths"]) > 0:
        final_lengths = eval_data["ep_lengths"][-1]  # Last eval episode lengths
        final_metrics["ep_len_mean"] = float(np.mean(final_lengths))
        final_metrics["ep_len_std"] = float(np.std(final_lengths))
    
    if "timesteps" in eval_data and len(eval_data["timesteps"]) > 0:
        final_metrics["eval_at_timestep"] = int(eval_data["timesteps"][-1])
    
    # Update metadata
    metadata["final_metrics"] = final_metrics
    metadata["evaluations_completed_at"] = datetime.now().isoformat()
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def append_computed_metrics(run_dir: Path, metrics_payload: Dict[str, Any]) -> None:
    """Append computed experiment metrics to metadata.json.

    Args:
        run_dir: Directory containing metadata.json
        metrics_payload: Metrics payload generated by experiment evaluation
    """
    metadata_path = run_dir / 'metadata.json'
    if not metadata_path.exists():
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    metadata['computed_metrics'] = metrics_payload
    metadata['computed_metrics_updated_at'] = datetime.now().isoformat()

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_visualization_metadata(
    run_dir: Path,
    run_id: str,
    env_name: str,
    algorithm: str,
    model_path: Path,
    vecnorm_path: Optional[Path],
    num_episodes: int,
    render: bool,
    save_plot: bool,
    output_dir: Optional[str] = None,
) -> Path:
    """Save visualization run metadata as JSON.
    
    Args:
        run_dir: Directory where metadata will be saved
        run_id: Unique run identifier
        env_name: Environment name
        algorithm: Algorithm name
        model_path: Path to model file
        vecnorm_path: Path to VecNormalize file
        num_episodes: Number of episodes run
        render: Whether environment was rendered
        save_plot: Whether plots were saved
        output_dir: Custom output directory
    
    Returns:
        Path to the saved metadata file
    """
    metadata = {
        'type': 'visualization',
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'name': env_name,
            'base_name': env_name.split('-')[0].lower(),
        },
        'algorithm': algorithm,
        'visualization': {
            'model_path': str(model_path),
            'vecnorm_path': str(vecnorm_path) if vecnorm_path else None,
            'num_episodes': num_episodes,
            'render': render,
            'save_plot': save_plot,
        },
        'output_dir': str(output_dir) if output_dir else None,
        'system': get_system_info(),
    }
    
    metadata_path = run_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def load_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metadata from a run directory.
    
    Args:
        run_dir: Directory containing metadata.json
    
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = run_dir / 'metadata.json'
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def update_visualization_results(
    run_dir: Path,
    episode_rewards: list,
    episode_lengths: list,
) -> None:
    """Update metadata with visualization results.
    
    Args:
        run_dir: Directory containing metadata.json
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
    """
    metadata_path = run_dir / 'metadata.json'
    if not metadata_path.exists():
        return
    print(f"Updating metadata at {metadata_path} with results...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    std_reward = np.std(episode_rewards) if episode_rewards else 0.0
    # Add results
    metadata['results'] = {
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': episode_lengths,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'min_reward': float(np.min(episode_rewards)) if episode_rewards else 0.0,
        'max_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
