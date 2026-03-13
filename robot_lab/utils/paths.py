"""Path utilities for managing user data directories."""

from pathlib import Path
from typing import Optional


def get_user_data_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the user data directory for robot_lab.
    
    Args:
        custom_dir: Optional custom directory path. If provided, uses this directory.
                   Otherwise, uses 'data' directory at workspace root.
    
    Returns:
        Path to the user data directory.
    """
    if custom_dir:
        base_dir = Path(custom_dir)
    else:
        # Use 'data' directory at workspace root by default
        base_dir = Path.cwd() / "data"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_models_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the models directory.
    
    Args:
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the models directory.
    """
    models_dir = get_user_data_dir(custom_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_logs_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the logs directory.
    
    Args:
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the logs directory.
    """
    logs_dir = get_user_data_dir(custom_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_experiments_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the experiments directory.
    
    Args:
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the experiments directory.
    """
    experiments_dir = get_user_data_dir(custom_dir) / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir


def get_results_index_path(custom_dir: Optional[str] = None) -> Path:
    """Return the path to the JSONL results index file.

    The file is NOT created — callers are responsible for opening it in
    append mode.  This intentional design prevents any disk write on import.

    Args:
        custom_dir: Optional custom output root (same as other path helpers).

    Returns:
        Path to ``data/experiments/results_index.jsonl``.
    """
    return get_user_data_dir(custom_dir) / "experiments" / "results_index.jsonl"

def get_graphs_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the graphs directory.
    
    Args:
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the graphs directory.
    """
    graphs_dir = get_user_data_dir(custom_dir) / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir


def get_tensorboard_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the tensorboard directory.
    
    Args:
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the tensorboard directory.
    """
    tensorboard_dir = get_user_data_dir(custom_dir) / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return tensorboard_dir


def get_debug_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the debug directory for debug configuration files.
    
    Args:
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the debug directory.
    """
    debug_dir = get_user_data_dir(custom_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir
