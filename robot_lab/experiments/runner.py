"""
Experiment runner for executing YAML-defined experiment campaigns.

This module orchestrates training runs from YAML configuration files, handling:
- Environment creation with control parameter injection
- Action wrapper application (frameskip, filters)
- Experiment tracking and metadata collection
- Error handling and progress reporting
- Graceful parameter validation (filters unsupported params with warnings)

Parameter Validation:
    By default, environment kwargs are validated in non-strict mode:
    - Unsupported parameters are filtered out with warnings
    - Only valid parameters are passed to the environment
    - Environments with **kwargs accept all parameters
    
    Set strict=True in _validate_env_kwargs() to raise errors instead.
"""

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import yaml
from loguru import logger

from robot_lab.experiments.tracker import ExperimentTracker
from robot_lab.training import train
from robot_lab.utils.metadata import append_computed_metrics
from robot_lab.utils.smoothness_metrics import (
    evaluate_smoothness_metrics,
    flatten_metric_groups,
)
from robot_lab.wrappers import create_action_wrapper


class ExperimentRunner:
    """
    Orchestrates training experiments from YAML specifications.
    
    Usage:
        >>> runner = ExperimentRunner("experiments/0_foundations/configs/smooth_locomotion_experiments.yaml")
        >>> runner.run_all()  # Run all enabled experiments
        >>> runner.run_experiment("exp0_baseline")  # Run specific experiment
    """

    def __init__(self, config_path: str, output_dir: str = "data/experiments"):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to YAML experiment configuration
            output_dir: Base directory for experiment outputs
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.experiments = self.config.get("experiments", {})
        self.common_config = self.config.get("common", {})
        
        logger.info(f"Loaded {len(self.experiments)} experiments from {self.config_path}")
    
    def run_all(self, dry_run: bool = False) -> Dict[str, bool]:
        """
        Run all enabled experiments in sequence.
        
        Args:
            dry_run: If True, only print what would be run without executing
        
        Returns:
            Dict mapping experiment_id to success status
        """
        results = {}
        enabled_experiments = [
            exp_id for exp_id, exp_config in self.experiments.items()
            if exp_config.get("enabled", True)  # Default to enabled if not specified
        ]
        
        logger.info(f"Running {len(enabled_experiments)} enabled experiments")
        
        for exp_id in enabled_experiments:
            logger.info(f"{'='*80}")
            logger.info(f"Experiment: {exp_id}")
            logger.info(f"{'='*80}\n")
            
            try:
                if dry_run:
                    self._print_experiment_plan(exp_id)
                    results[exp_id] = True
                else:
                    self.run_experiment(exp_id)
                    results[exp_id] = True
                    logger.success(f"✓ Completed: {exp_id}")
            except Exception as e:
                logger.error(f"✗ Failed: {exp_id} - {e}")
                results[exp_id] = False
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("Experiment Campaign Summary")
        logger.info(f"{'='*80}")
        successes = sum(1 for success in results.values() if success)
        logger.info(f"Completed: {successes}/{len(results)}")
        
        return results
    
    def run_experiment(self, experiment_id: str) -> None:
        """
        Run a single experiment by ID.
        
        Args:
            experiment_id: ID of experiment from YAML config (e.g., 'exp0_baseline')
        
        Raises:
            KeyError: If experiment_id not found in config
            ValueError: If experiment configuration is invalid
        """
        if experiment_id not in self.experiments:
            raise KeyError(
                f"Experiment '{experiment_id}' not found. "
                f"Available: {list(self.experiments.keys())}"
            )
        
        exp_config = self.experiments[experiment_id]
        
        # Check if disabled
        if not exp_config.get("enabled", True):
            logger.warning(f"Experiment {experiment_id} is disabled, skipping")
            return
        
        # Merge with common config
        merged_config = self._merge_configs(self.common_config, exp_config)
        
        # Extract components
        environment = merged_config.get("environment", "A1Quadruped-v0")
        algorithm = merged_config.get("algorithm", "SAC")
        training_config = merged_config.get("training", {})
        metrics_config = merged_config.get("metrics", {})
        control_params = merged_config.get("control_params", {})
        tag = exp_config.get("tag")  # Optional tag for grouping
        
        # Validate and filter environment parameters early
        # Non-strict mode: filter invalid params and warn (graceful handling)
        logger.info(f"Validating control parameters for {environment}...")
        control_params = self._validate_env_kwargs(environment, control_params, strict=False)
        
        # Setup experiment tracking
        experiment_name = self.config_path.parent.parent.name  # e.g., "0_foundations"
        tracker = ExperimentTracker(
            experiment_name=experiment_name,
            run_name=experiment_id,
            base_dir=str(self.output_dir),
            tag=tag,  # Pass tag for subdirectory organization
        )
        
        # Save experiment configuration
        tracker.set_description(
            description=exp_config.get("description", "No description"),
            notes=exp_config.get("notes", "")
        )
        tracker.log_params(training_config)
        tracker.log_env_config(merged_config)
        
        # Create environment with wrappers
        logger.info(f"Creating environment: {environment}")
        
        # Create wrapper function that applies action wrappers
        def env_wrapper_fn(env: gym.Env) -> gym.Env:
            return create_action_wrapper(env, merged_config)
        
        # Training parameters
        total_timesteps = training_config.get("total_timesteps", 100000)
        seed = training_config.get("seed", 42)
        num_envs = merged_config.get("num_envs", 8)
        
        # Output directory for this specific run
        run_output_dir = tracker.run_dir
        
        logger.info(f"Training {algorithm} on {environment}")
        logger.info(f"  Timesteps: {total_timesteps}")
        logger.info(f"  Seed: {seed}")
        logger.info(f"  Num envs: {num_envs}")
        logger.info(f"  Control params: {control_params}")
        logger.info(f"  Output: {run_output_dir}")
        
        # Update tracker status
        tracker.set_status("running")
        
        try:
            # Run training with direct parameter passing
            model, model_path, vecnorm_path = train(
                env_name=environment,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
                num_envs=num_envs,
                output_dir=str(run_output_dir),
                env_kwargs=control_params,  # Pass control params to environment
                env_wrapper_fn=env_wrapper_fn,  # Pass wrapper function
            )

            requested_metrics = flatten_metric_groups(metrics_config)
            if requested_metrics:
                logger.info(
                    f"Computing {len(requested_metrics)} configured experiment metrics for {experiment_id}"
                )
                metrics_payload = evaluate_smoothness_metrics(
                    env_name=environment,
                    algorithm=algorithm,
                    model_path=str(model_path),
                    vecnorm_path=str(vecnorm_path),
                    requested_metrics=requested_metrics,
                    num_episodes=training_config.get("eval_episodes", 10),
                    seed=seed,
                    env_kwargs=control_params,
                    env_wrapper_fn=env_wrapper_fn,
                )

                tracker.set_computed_metrics(metrics_payload)
                append_computed_metrics(model_path.parent, metrics_payload)
                logger.success("✓ Computed metrics saved to environment config and training metadata")
            else:
                logger.info("No custom metrics configured in YAML; skipping metrics evaluation")
            
            tracker.set_status("completed")
            logger.success(f"Experiment {experiment_id} completed successfully")
        
        except Exception as e:
            tracker.set_status("failed")
            logger.error(f"Experiment {experiment_id} failed: {e}")
            raise
    
    def _merge_configs(
        self, common: Dict[str, Any], experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge common and experiment-specific configs.
        
        Experiment config takes precedence over common config.
        
        Args:
            common: Common configuration shared across experiments
            experiment: Experiment-specific configuration
        
        Returns:
            Merged configuration dictionary
        """
        merged = common.copy()
        
        for key, value in experiment.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Deep merge for nested dicts
                merged[key] = {**merged[key], **value}
            else:
                # Direct override
                merged[key] = value
        
        return merged
    
    def _validate_env_kwargs(
        self, env_name: str, env_kwargs: Dict[str, Any], strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate and filter environment keyword arguments.
        
        Args:
            env_name: Gymnasium environment ID
            env_kwargs: Keyword arguments to pass to environment
            strict: If True, raise error on invalid params. If False, filter and warn.
        
        Returns:
            Validated (and potentially filtered) kwargs dictionary
        
        Raises:
            ValueError: If strict=True and environment doesn't accept provided kwargs
        """
        if not env_kwargs:
            return {}  # Nothing to validate
        
        try:
            # Get the environment spec
            env_spec = gym.spec(env_name)
            
            # Get the environment class
            # Handle both module:Class and direct Class references
            entry_point = env_spec.entry_point
            if isinstance(entry_point, str):
                module_name, class_name = entry_point.split(':')
                module = __import__(module_name, fromlist=[class_name])
                env_class = getattr(module, class_name)
            else:
                env_class = entry_point
            
            # Get __init__ signature
            sig = inspect.signature(env_class.__init__)
            
            # Get accepted parameters (excluding 'self')
            accepted_params = set(sig.parameters.keys()) - {'self'}
            
            # Check for **kwargs support
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD 
                for p in sig.parameters.values()
            )
            
            # If environment accepts **kwargs, allow everything
            if has_var_keyword:
                logger.debug(f"✓ Environment {env_name} accepts **kwargs, all params allowed")
                return env_kwargs
            
            # Validate provided kwargs
            invalid_params = []
            valid_kwargs = {}
            
            for param, value in env_kwargs.items():
                if param in accepted_params:
                    valid_kwargs[param] = value
                else:
                    invalid_params.append(param)
            
            # Handle invalid parameters
            if invalid_params:
                error_msg = (
                    f"Environment '{env_name}' does not accept parameters: {invalid_params}\n"
                    f"Accepted parameters: {sorted(accepted_params)}\n"
                    f"Provided: {list(env_kwargs.keys())}\n"
                )
                
                if strict:
                    # Strict mode: raise error
                    error_msg += (
                        f"\nTo fix this, either:\n"
                        f"  1. Remove unsupported parameters from control_params in your YAML config\n"
                        f"  2. Update {env_class.__name__}.__init__() to accept these parameters"
                    )
                    raise ValueError(error_msg)
                else:
                    # Graceful mode: filter and warn
                    logger.warning(
                        f"⚠ Filtered out unsupported parameters for {env_name}: {invalid_params}\n"
                        f"   Using only: {list(valid_kwargs.keys())}"
                    )
                    return valid_kwargs
            
            logger.debug(f"✓ All environment kwargs validated for {env_name}")
            return env_kwargs
        
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise our validation error
            
            # If we can't validate, warn and return original kwargs
            logger.warning(f"Could not validate env kwargs for {env_name}: {e}")
            logger.warning(f"Proceeding with provided kwargs: {list(env_kwargs.keys())}")
            return env_kwargs  # Let gym.make handle any errors
    
    def _print_experiment_plan(self, experiment_id: str) -> None:
        """Print what would be executed for an experiment (dry-run mode)."""
        exp_config = self.experiments[experiment_id]
        merged_config = self._merge_configs(self.common_config, exp_config)
        
        environment = merged_config.get("environment", "A1Quadruped-v0")
        control_params = merged_config.get("control_params", {})
        
        print(f"\nExperiment: {experiment_id}")
        print(f"Description: {exp_config.get('description', 'N/A')}")
        
        # Show tag if present
        tag = exp_config.get('tag')
        if tag:
            print(f"Tag: {tag}")
        
        print(f"\nConfiguration:")
        print(f"  Environment: {environment}")
        print(f"  Algorithm: {merged_config.get('algorithm')}")
        print(f"  Control Params: {control_params}")
        
        # Validate control params (non-strict for dry-run)
        try:
            validated_params = self._validate_env_kwargs(environment, control_params, strict=False)
            if validated_params != control_params:
                filtered_out = set(control_params.keys()) - set(validated_params.keys())
                print(f"  ⚠ Filtered params: {list(filtered_out)}")
                print(f"  ✓ Valid params: {list(validated_params.keys())}")
            else:
                print(f"  ✓ Control params validated")
        except ValueError as e:
            print(f"  ✗ Control params validation FAILED:")
            print(f"     {str(e).replace(chr(10), chr(10) + '     ')}")
            return  # Don't show rest of plan if validation fails
        
        if 'action_repeat' in merged_config:
            print(f"  Action Repeat: {merged_config['action_repeat']}")
        
        if 'action_filter' in merged_config:
            print(f"  Action Filter: {merged_config['action_filter']}")
        
        training = merged_config.get('training', {})
        print(f"  Timesteps: {training.get('total_timesteps')}")
        print(f"  Seed: {training.get('seed')}")
        
        # Show output directory structure
        experiment_name = self.config_path.parent.parent.name
        if tag:
            output_path = f"{self.output_dir}/{experiment_name}/{tag}/{experiment_id}"
        else:
            output_path = f"{self.output_dir}/{experiment_name}/{experiment_id}"
        print(f"\nOutput Directory: {output_path}")
        
        if exp_config.get('notes'):
            print(f"\nNotes: {exp_config['notes']}")
    
    def list_experiments(self, enabled_only: bool = False) -> List[str]:
        """
        List available experiments.
        
        Args:
            enabled_only: If True, only return enabled experiments
        
        Returns:
            List of experiment IDs
        """
        if enabled_only:
            return [
                exp_id for exp_id, exp_config in self.experiments.items()
                if exp_config.get("enabled", True)
            ]
        return list(self.experiments.keys())


def run_experiment_from_yaml(
    config_path: str,
    experiment_id: Optional[str] = None,
    output_dir: str = "data/experiments",
    dry_run: bool = False,
) -> None:
    """
    Convenience function to run experiments from YAML.
    
    Args:
        config_path: Path to YAML experiment configuration
        experiment_id: Specific experiment to run (None = run all)
        output_dir: Base output directory
        dry_run: If True, only print execution plan
    """
    runner = ExperimentRunner(config_path, output_dir)
    
    if experiment_id:
        if dry_run:
            runner._print_experiment_plan(experiment_id)
        else:
            runner.run_experiment(experiment_id)
    else:
        runner.run_all(dry_run=dry_run)
