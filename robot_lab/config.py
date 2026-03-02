"""Configuration loading and management for robot_lab."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from importlib.resources import files
from loguru import logger


def load_hyperparameters(
    env_name: str,
    algorithm: str,
    custom_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load hyperparameters from JSON file with fallback hierarchy.
    
    Priority order:
    1. Custom config path if provided
    2. Environment-specific config (e.g., walker2d_sac.json)
    3. Algorithm-specific default (e.g., default_sac.json)
    4. Generic default (default.json)
    
    Args:
        env_name: Environment name (e.g., "Walker2d-v5")
        algorithm: Algorithm name ("SAC" or "PPO")
        custom_config_path: Optional path to custom configuration file
    
    Returns:
        Dictionary containing configuration
    
    Raises:
        ValueError: If no configuration file can be found
    """
    env_base_name = env_name.split('-')[0].lower()
    algo_name = algorithm.lower()
    
    # Try custom path first
    if custom_config_path:
        custom_path = Path(custom_config_path)
        if custom_path.exists():
            with open(custom_path, 'r') as f:
                config = json.load(f)
            logger.success(f"Loaded hyperparameters from {custom_config_path}")
            return config
        else:
            logger.warning(f"Custom config not found at {custom_config_path}")
    
    # Try package configs with fallback hierarchy
    configs_path = files('robot_lab').joinpath('configs')
    
    # Priority 1: Environment-specific config
    config_filename = f"{env_base_name}_{algo_name}.json"
    try:
        config_data = configs_path.joinpath(config_filename).read_text()
        config = json.loads(config_data)
        logger.success(f"Loaded hyperparameters from {config_filename}")
        return config
    except FileNotFoundError:
        pass
    
    # Priority 2: Algorithm-specific default
    algo_default_filename = f"default_{algo_name}.json"
    try:
        config_data = configs_path.joinpath(algo_default_filename).read_text()
        config = json.loads(config_data)
        logger.warning(f"No specific config found for {env_base_name}_{algo_name}")
        logger.success(f"Loading {algorithm} default hyperparameters from {algo_default_filename}")
        logger.info(f"Consider creating {config_filename} for better performance.")
        return config
    except FileNotFoundError:
        pass
    
    # Priority 3: Generic default
    try:
        config_data = configs_path.joinpath('default.json').read_text()
        config = json.loads(config_data)
        logger.warning(f"No specific config found for {env_base_name}_{algo_name}")
        logger.warning(f"No algorithm default found for {algorithm}")
        logger.success(f"Loading generic default hyperparameters from default.json")
        logger.warning(f"This may not work well. Please create a dedicated config.")
        return config
    except FileNotFoundError:
        pass
    
    raise ValueError(
        f"No hyperparameter file found for {env_name} with {algorithm}. "
        f"Expected one of: {config_filename}, {algo_default_filename}, or default.json"
    )


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path where to save the configuration
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.success(f"Configuration saved to {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that a configuration dictionary has required fields.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ['algorithm', 'num_envs', 'total_timesteps', 
                       'hyperparameters', 'vec_normalize']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    return True
