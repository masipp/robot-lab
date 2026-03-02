# Experiment Specification Schema

This document describes the JSON schema for defining experiment campaigns in robot_lab.

## Overview

Experiments are defined using a JSON specification that describes:
- Which environments and algorithms to test
- Hyperparameter configurations and sweeps
- Training parameters (seeds, timesteps, etc.)
- Resource limits and output configuration

## Schema Structure

### ExperimentSpec

Top-level experiment specification.

```json
{
  "experiment_metadata": { ... },
  "environments": [ ... ],
  "algorithms": [ ... ],
  "base_hyperparameters": { ... },
  "hyperparameter_sweeps": [ ... ],
  "training_config": { ... },
  "evaluation_criteria": { ... },
  "resource_limits": { ... },
  "output_config": { ... }
}
```

### ExperimentMetadata

Metadata about the experiment campaign.

- `name` (string, required): Unique experiment name
- `description` (string, required): Human-readable description
- `created_by` (string): Who created this experiment (default: "user")
- `created_at` (string): ISO timestamp (auto-generated)
- `tags` (array of strings): Tags for categorization

### Environments

List of environments to test.

```json
"environments": [
  {
    "name": "Walker2d-v5",
    "config_overrides": {}
  }
]
```

### Algorithms

List of algorithms to use. Must be one or more of: `["SAC", "PPO"]`

### BaseHyperparameters

Hyperparameter configurations for each algorithm.

```json
"base_hyperparameters": {
  "SAC": {
    "source": "configs/walker2d_sac.json",
    "overrides": {
      "learning_rate": 0.001
    }
  }
}
```

- `source`: Path to base configuration JSON file
- `overrides`: Optional parameter overrides

### HyperparameterSweeps

Defines hyperparameter search spaces.

**Discrete sweep:**
```json
{
  "parameter": "learning_rate",
  "values": [0.0001, 0.0003, 0.001, 0.003],
  "type": "discrete"
}
```

**Continuous sweep:**
```json
{
  "parameter": "gamma",
  "range": [0.95, 0.999],
  "type": "continuous",
  "sampling": "log_uniform",
  "num_samples": 5
}
```

- `parameter`: Name of hyperparameter to vary
- `values`: List of discrete values (for discrete type)
- `range`: [min, max] range (for continuous type)
- `type`: "discrete" or "continuous"
- `sampling`: "uniform", "log_uniform", or "grid" (for continuous)
- `num_samples`: Number of samples to draw (for continuous)

### TrainingConfig

Configuration for training runs.

```json
"training_config": {
  "num_seeds": 5,
  "seeds": [42, 123, 456, 789, 1011],
  "total_timesteps": 500000,
  "num_envs": 8,
  "eval_frequency": 10000,
  "eval_episodes": 10,
  "save_freq": 50000,
  "checkpoint_best": true
}
```

- `num_seeds`: Number of random seeds (≥1)
- `seeds`: Specific seeds to use (optional, length must match num_seeds)
- `total_timesteps`: Training duration (≥1000)
- `num_envs`: Number of parallel environments (≥1)
- `eval_frequency`: Evaluation frequency in timesteps
- `eval_episodes`: Number of episodes per evaluation
- `save_freq`: Checkpoint save frequency (null = disabled)
- `checkpoint_best`: Whether to save best model

### EvaluationCriteria

Criteria for evaluating results.

```json
"evaluation_criteria": {
  "primary_metric": "mean_reward",
  "secondary_metrics": ["episode_length"],
  "aggregation": "mean_last_100",
  "comparison_method": "simple_comparison",
  "confidence_level": 0.95
}
```

- `primary_metric`: Main metric to optimize
- `secondary_metrics`: Additional metrics to track
- `aggregation`: "mean_last_10", "mean_last_100", "max", or "mean_all"
- `comparison_method`: "statistical_test" or "simple_comparison"
- `confidence_level`: Confidence level for statistical tests (0.0-1.0)

### ResourceLimits

Resource constraints for execution.

```json
"resource_limits": {
  "max_concurrent_runs": 4,
  "max_total_runs": 100,
  "max_runtime_hours": 48,
  "gpu_required": false
}
```

### OutputConfig

Configuration for experiment outputs.

```json
"output_config": {
  "base_dir": "experiments/my_experiment",
  "save_models": true,
  "save_logs": true,
  "generate_report": true,
  "tensorboard": true,
  "mlflow_tracking": false,
  "wandb_tracking": false
}
```

## Complete Example

See `src/robot_lab/experiments/spec_templates.py` for complete examples:
- `HYPERPARAM_SWEEP_TEMPLATE`: Hyperparameter search
- `ALGORITHM_COMPARISON_TEMPLATE`: Algorithm comparison
- `QUICK_TEST_TEMPLATE`: Quick test run

## Usage

```python
from robot_lab.experiments import get_template

# Get a template
spec = get_template("hyperparam_sweep")

# Modify as needed
spec["experiment_metadata"]["name"] = "my_custom_experiment"
spec["environments"] = [{"name": "HalfCheetah-v5", "config_overrides": {}}]

# Save to file
import json
with open("my_experiment.json", "w") as f:
    json.dump(spec, f, indent=2)
```

## Future: AI-Generated Specifications

The `AIExperimentPlanner` class (currently stub) will enable:
- Generating specs from natural language descriptions
- Suggesting hyperparameter ranges automatically
- Adaptive experiment design (Bayesian optimization)
- Automated result interpretation and next-step suggestions

```python
# Future usage (not yet implemented)
from robot_lab.experiments import AIExperimentPlanner

planner = AIExperimentPlanner()
spec = planner.generate_from_natural_language(
    "Compare SAC and PPO on Walker2d with 5 seeds, "
    "focusing on sample efficiency"
)
```
