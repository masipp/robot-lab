"""Experiment tracking for robot_lab using JSON and YAML files."""

import json
import yaml
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel, ValidationError

from robot_lab.experiments.schemas import RunMetadata


class ExperimentTracker:
    """Lightweight experiment tracker using JSON files."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        base_dir: str = "experiments",
        tag: Optional[str] = None,
    ):
        """Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment campaign (e.g., "0_foundations")
            run_name: Name of this specific run (e.g., "exp0_baseline")
            base_dir: Base directory for experiments
            tag: Optional tag for grouping experiments (e.g., "SmoothMotion-ControlGains")
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tag = tag
        self.run_id = f"{experiment_name}_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directory structure
        # If tag is provided: base_dir / experiment_name / tag / run_name
        # Otherwise: base_dir / experiment_name / run_name
        self.experiment_dir = Path(base_dir) / experiment_name
        if tag:
            self.run_dir = self.experiment_dir / tag / run_name
        else:
            self.run_dir = self.experiment_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.env_config_file = self.run_dir / "environment_config.yaml"
        
        # Store experiment metadata (will be written to env_config)
        self.config_data = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "tag": tag,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
        }
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log training parameters.
        
        Args:
            params: Dictionary of training parameters (seed, total_timesteps)
        """
        self.config_data["training"] = params
        logger.debug(f"Logged training params")
    
    def log_env_config(self, env_config: Dict[str, Any]) -> None:
        """Log comprehensive environment configuration as YAML.
        
        This creates a complete experiment specification including:
        - Environment name and algorithm
        - Control parameters (kp, kd, etc.)
        - Action wrapper settings (frameskip, filters)
        - Training configuration (timesteps, seed)
        - Experiment metadata (name, tag, description)
        
        This file should be sufficient to reproduce the experiment.
        
        Args:
            env_config: Dictionary of environment configuration parameters
        """
        # Build comprehensive config
        comprehensive_config = {
            **self.config_data,  # experiment_name, run_name, tag, created_at, status
            "environment": env_config.get("environment"),
            "algorithm": env_config.get("algorithm"),
            "control_params": env_config.get("control_params", {}),
            "num_envs": env_config.get("num_envs"),
            "metrics": env_config.get("metrics", {}),
        }
        
        # Add training config if present
        if "training" in self.config_data:
            comprehensive_config["training"] = self.config_data["training"]
        
        # Add action wrappers if present
        if "action_repeat" in env_config:
            comprehensive_config["action_repeat"] = env_config["action_repeat"]
        if "action_filter" in env_config:
            comprehensive_config["action_filter"] = env_config["action_filter"]
        
        # Add description/notes if present
        if "description" in self.config_data:
            comprehensive_config["description"] = self.config_data["description"]
        if "notes" in self.config_data:
            comprehensive_config["notes"] = self.config_data["notes"]
        
        # Remove None values
        comprehensive_config = {k: v for k, v in comprehensive_config.items() if v is not None}
        
        with open(self.env_config_file, 'w') as f:
            yaml.dump(comprehensive_config, f, default_flow_style=False, sort_keys=False)
        logger.success(f"Saved experiment config to {self.env_config_file}")

    def set_computed_metrics(self, metrics_payload: Dict[str, Any]) -> None:
        """Persist computed metrics into environment_config.yaml.

        Args:
            metrics_payload: Metrics evaluation payload
        """
        self.config_data["computed_metrics"] = metrics_payload
        self.config_data["metrics_updated_at"] = datetime.now().isoformat()

        if self.env_config_file.exists():
            with open(self.env_config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = dict(self.config_data)

        config["computed_metrics"] = metrics_payload
        config["metrics_updated_at"] = self.config_data["metrics_updated_at"]

        with open(self.env_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional timestep
        """
        # Load existing metrics
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Add new metrics with timestamp
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        if step is not None:
            entry["step"] = step
        
        all_metrics.append(entry)
        
        # Save updated metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
    
    def log_artifact(self, file_path: str, artifact_type: str = "file") -> None:
        """Log a file artifact.
        
        Args:
            file_path: Path to the file
            artifact_type: Type of artifact (model, plot, etc.)
        """
        # Record artifact in metadata
        if "artifacts" not in self.metadata:
            self.metadata["artifacts"] = []
        
        self.metadata["artifacts"].append({
            "type": artifact_type,
            "path": file_path,
            "logged_at": datetime.now().isoformat()
        })
        self._save_metadata()
    
    def set_tag(self, key: str, value: str) -> None:
        """Add metadata tags.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if "tags" not in self.metadata:
            self.metadata["tags"] = {}
        
        self.metadata["tags"][key] = value
        self._save_metadata()
    
    def set_status(self, status: str) -> None:
        """Update experiment status.
        
        Args:
            status: New status (running, completed, failed, etc.)
        """
        self.config_data["status"] = status
        self.config_data["updated_at"] = datetime.now().isoformat()
        # Re-save environment config with updated status
        if self.env_config_file.exists():
            with open(self.env_config_file, 'r') as f:
                config = yaml.safe_load(f)
            config["status"] = status
            config["updated_at"] = self.config_data["updated_at"]
            with open(self.env_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def set_description(self, description: str, notes: str = "") -> None:
        """Set experiment description and notes.
        
        Args:
            description: Experiment description
            notes: Optional notes
        """
        self.config_data["description"] = description
        self.config_data["notes"] = notes
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility.
        
        Returns:
            Dictionary containing system information
        """
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node(),
        }
        
        # Try to get GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", 
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')
                system_info["gpus"] = gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            system_info["gpus"] = None
        
        # Try to get git commit
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                system_info["git_commit"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            system_info["git_commit"] = None
        
        return system_info
    
    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
    
    def _validate_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Validate hyperparameter structure.
        
        Args:
            params: Hyperparameters to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Basic validation: ensure required fields are present
        if not isinstance(params, dict):
            raise ValueError("Hyperparameters must be a dictionary")
        
        # Check for common mistakes
        if not params:
            logger.warning("Empty hyperparameters dictionary")
        
        # Validate numeric parameters are actually numeric
        numeric_keys = ['learning_rate', 'gamma', 'batch_size', 'buffer_size', 
                       'tau', 'total_timesteps', 'num_envs']
        for key in numeric_keys:
            if key in params:
                value = params[key]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{key}' must be numeric, got {type(value)}")
                if key in ['learning_rate', 'gamma', 'tau'] and not (0 <= value <= 1):
                    logger.warning(f"Parameter '{key}'={value} is outside typical range [0, 1]")
        
        # Validate positive integers
        positive_int_keys = ['batch_size', 'buffer_size', 'total_timesteps', 'num_envs']
        for key in positive_int_keys:
            if key in params:
                value = params[key]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Parameter '{key}' must be a positive integer, got {value}")
        
        logger.debug("Hyperparameter validation passed")
