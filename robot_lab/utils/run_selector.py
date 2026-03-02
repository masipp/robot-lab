"""Utilities for selecting and managing training runs."""

import json
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from loguru import logger
from robot_lab.envs import get_env_registry


def list_training_runs(logs_dir: Path, limit: int = 20) -> List[Tuple[Path, dict]]:
    """List recent training runs from the logs directory.
    
    Args:
        logs_dir: Path to the logs directory
        limit: Maximum number of runs to return
    
    Returns:
        List of tuples (run_path, run_info) sorted by most recent first
    """
    if not logs_dir.exists():
        return []
    
    runs = []
    for run_dir in logs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Skip visualization runs
        if run_dir.name.endswith('_viz'):
            continue
        
        # Try to parse run_id format: YYYYMMDD_HHMMSS_hash_suffix
        parts = run_dir.name.split('_')
        if len(parts) < 4:
            continue
        
        try:
            # Parse timestamp from run_id
            timestamp_str = f"{parts[0]}_{parts[1]}"
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            # Extract algorithm and environment from suffix
            suffix_parts = parts[3:]  # Everything after hash
            if len(suffix_parts) >= 2:
                algo = suffix_parts[0].upper()
                env = '_'.join(suffix_parts[1:])
            else:
                algo = "UNKNOWN"
                env = "unknown"
            
            # Check for model file
            model_files = list(run_dir.glob("*.zip"))
            vecnorm_files = list(run_dir.glob("*_vecnorm.pkl"))
            
            if model_files:
                run_info = {
                    'run_id': run_dir.name,
                    'run_dir': run_dir,  # Include run directory path
                    'timestamp': timestamp,
                    'algo': algo,
                    'env': env,
                    'model_file': model_files[0],
                    'vecnorm_file': vecnorm_files[0] if vecnorm_files else None,
                    'age_hours': (datetime.now() - timestamp).total_seconds() / 3600,
                }
                runs.append((run_dir, run_info))
        except (ValueError, IndexError):
            continue
    
    # Sort by timestamp (most recent first)
    runs.sort(key=lambda x: x[1]['timestamp'], reverse=True)
    
    return runs[:limit]


def format_run_option(run_info: dict, index: int, is_latest: bool = False) -> str:
    """Format a run as a selection option.
    
    Args:
        run_info: Dictionary with run information
        index: Index number for display
        is_latest: Whether this is the most recent run
    
    Returns:
        Formatted string for display
    """
    age = run_info['age_hours']
    if age < 1:
        age_str = f"{int(age * 60)}m ago"
    elif age < 24:
        age_str = f"{int(age)}h ago"
    else:
        age_str = f"{int(age / 24)}d ago"
    
    timestamp_str = run_info['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    
    latest_marker = " [LATEST]" if is_latest else ""
    
    return (
        f"{index}. {run_info['algo']:>3} | {run_info['env']:<25} | "
        f"{timestamp_str} ({age_str}){latest_marker}"
    )


def get_full_env_name(run_info: dict) -> Optional[str]:
    """Try to reconstruct the full environment name from run info.
    First attempts to read the actual environment name from metadata.json,
    then falls back to using the environment registry to match abbreviated names.
    
    Args:
        run_info: Dictionary with run information (must include 'run_dir')
    
    Returns:
        Full environment name or None
    """
    # First, try to read from metadata.json if available
    run_dir = run_info.get('run_dir')
    if run_dir:
        metadata_file = run_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                # Get actual environment name from metadata
                env_name = metadata.get('environment', {}).get('name')
                if env_name:
                    logger.debug(f"Found environment name in metadata: {env_name}")
                    return env_name
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Could not read metadata.json: {e}")
    
    # Fallback: Use registry to match abbreviated name
    env_abbrev = run_info['env'].lower()
    
    # Query registry for all available environments
    registry = get_env_registry()
    all_envs = registry.list_envs(include_custom=True)
    
    # Try to match abbreviated name against full environment IDs
    for metadata in all_envs:
        env_id = metadata.env_id
        
        # Extract base name (everything before version suffix)
        # e.g., "Walker2d-v5" -> "walker2d"
        base_name = env_id.split('-')[0].lower()
        
        # Match against abbreviated name
        if base_name == env_abbrev:
            logger.debug(f"Matched {env_abbrev} to {env_id} via registry")
            return env_id
    
    # No match found
    logger.warning(f"Could not find full environment name for '{env_abbrev}'")
    # No match found
    return None
