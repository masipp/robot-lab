"""Debug configuration loader for development."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from robot_lab.utils.paths import get_debug_dir


def load_debug_config(config_name: str, custom_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a debug configuration from JSON file.
    
    Args:
        config_name: Name of the debug config file (with or without .json extension).
        custom_dir: Optional custom directory path.
    
    Returns:
        Dictionary containing the configuration parameters.
        
    Raises:
        FileNotFoundError: If the debug configuration file doesn't exist.
        json.JSONDecodeError: If the configuration file is not valid JSON.
    """
    # Ensure .json extension
    if not config_name.endswith('.json'):
        config_name = f"{config_name}.json"
    
    debug_dir = get_debug_dir(custom_dir)
    config_path = debug_dir / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Debug configuration not found: {config_path}\n"
            f"Available configs in {debug_dir}: "
            f"{', '.join(f.name for f in debug_dir.glob('*.json')) or 'none'}"
        )
    
    logger.info(f"Loading debug configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.debug(f"Loaded debug config: {config}")
    return config


def create_debug_config_template(config_name: str, config_type: str = "train", 
                                 custom_dir: Optional[str] = None) -> Path:
    """Create a debug configuration template file.
    
    Args:
        config_name: Name for the debug config file (with or without .json extension).
        config_type: Type of config - 'train' or 'visualize'.
        custom_dir: Optional custom directory path.
    
    Returns:
        Path to the created configuration file.
    """
    # Ensure .json extension
    if not config_name.endswith('.json'):
        config_name = f"{config_name}.json"
    
    debug_dir = get_debug_dir(custom_dir)
    config_path = debug_dir / config_name
    
    if config_type == "train":
        template = {
            "env": "Walker2d-v5",
            "algo": "SAC",
            "seed": 42,
            "config": None,
            "output_dir": None,
            "eval_freq": 10000,
            "eval_episodes": 10,
            "save_freq": None,
            "checkpoints": False
        }
    elif config_type == "visualize":
        template = {
            "env": "Walker2d-v5",
            "algo": "SAC",
            "model_path": None,
            "vecnorm_path": None,
            "episodes": 3,
            "no_render": False,
            "no_plot": False,
            "output_dir": None
        }
    else:
        raise ValueError(f"Unknown config_type: {config_type}. Must be 'train' or 'visualize'")
    
    with open(config_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created debug config template: {config_path}")
    return config_path
