# Metadata System Documentation

The robot_lab system now automatically saves comprehensive metadata for every training and visualization run as JSON files.

## Location

Each run directory contains a `metadata.json` file:
- **Training runs**: `data/logs/<run_id>/metadata.json`
- **Visualization runs**: `data/logs/<run_id>_viz/metadata.json`

## Training Metadata Format

Example from a training run:

```json
{
  "type": "training",
  "run_id": "20260215_124504_7693bfd1_sac_mountaincarcontinuous",
  "timestamp": "2026-02-15T12:45:06.241146",
  "environment": {
    "name": "MountainCarContinuous-v0",
    "base_name": "mountaincarcontinuous"
  },
  "algorithm": "SAC",
  "training": {
    "seed": 42,
    "total_timesteps": 20000,
    "num_envs": 2,
    "eval_freq": 5000,
    "eval_episodes": 10,
    "save_freq": null,
    "use_checkpoints": false
  },
  "hyperparameters": {
    "policy": "MlpPolicy",
    "learning_rate": 0.001,
    "buffer_size": 300000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.02,
    "gamma": 0.99,
    "train_freq": 32,
    "gradient_steps": 32,
    "use_sde": true,
    "sde_sample_freq": 4
  },
  "vec_normalize": {
    "norm_obs": true,
    "norm_reward": true,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "gamma": 0.99
  },
  "output_dir": null,
  "system": {
    "python_version": "3.11.14 (main, Feb  3 2026, 22:51:56) [Clang 21.1.4 ]",
    "platform": "Linux-6.8.0-87-generic-x86_64-with-glibc2.39",
    "processor": "x86_64",
    "machine": "x86_64"
  }
}
```

## Visualization Metadata Format

Example from a visualization run:

```json
{
  "type": "visualization",
  "run_id": "20260215_130000_abc123de_sac_mountaincarcontinuous_viz",
  "timestamp": "2026-02-15T13:00:00.123456",
  "environment": {
    "name": "MountainCarContinuous-v0",
    "base_name": "mountaincarcontinuous"
  },
  "algorithm": "SAC",
  "visualization": {
    "model_path": "/path/to/model.zip",
    "vecnorm_path": "/path/to/vecnorm.pkl",
    "num_episodes": 3,
    "render": true,
    "save_plot": true
  },
  "output_dir": null,
  "system": {
    "python_version": "3.11.14 (main, Feb  3 2026, 22:51:56) [Clang 21.1.4 ]",
    "platform": "Linux-6.8.0-87-generic-x86_64-with-glibc2.39",
    "processor": "x86_64",
    "machine": "x86_64"
  },
  "results": {
    "episode_rewards": [450.2, 478.5, 461.3],
    "episode_lengths": [320, 335, 328],
    "mean_reward": 463.33,
    "std_reward": 11.82,
    "mean_length": 327.67,
    "min_reward": 450.2,
    "max_reward": 478.5
  }
}
```

## Key Features

### Training Metadata
- **Complete reproducibility**: All training parameters saved
- **Hyperparameters**: Full config including algorithm-specific settings
- **VecNormalize config**: Normalization settings for evaluation
- **System info**: Python version, platform for debugging
- **Seed tracking**: Random seed for exact reproducibility

### Visualization Metadata
- **Source tracking**: References to model and vecnorm files
- **Configuration**: Episodes, render, plot settings
- **Results**: Automatically updated with episode rewards and lengths
- **Statistics**: Mean, std, min, max for rewards and lengths

## Usage

### Loading Metadata Programmatically

```python
from robot_lab.utils.metadata import load_metadata

# Load training metadata
metadata = load_metadata(Path("data/logs/20260215_124504_7693bfd1_sac_mountaincarcontinuous"))

# Access fields
print(f"Algorithm: {metadata['algorithm']}")
print(f"Seed: {metadata['training']['seed']}")
print(f"Hyperparameters: {metadata['hyperparameters']}")
```

### Automatic Saving

Metadata is automatically saved:
- **Training**: When training starts (before model training)
- **Visualization**: When visualization starts (before running episodes)
- **Results update**: After visualization completes (with episode stats)

## Benefits

1. **Reproducibility**: Every run fully documented
2. **Experiment tracking**: Easy to compare runs
3. **Debugging**: System info helps identify platform issues
4. **Analysis**: JSON format easy to parse for analysis scripts
5. **Self-documenting**: No need to remember training settings

## Environment Configuration Tracking

**NEW**: As of Phase 0 experiments, control and physics parameters are now tracked separately for reproducibility.

### Environment Config Format

Example `environment_config.yaml`:

```yaml
env_id: A1Quadruped-v0
observation_space:
  shape: [30]
  dtype: float64
action_space:
  shape: [12]
  dtype: float32
  low: [-1.0, -1.0, ...]
  high: [1.0, 1.0, ...]
max_episode_steps: 1000
physics_steps_per_action: 5
control_parameters:
  num_actuators: 12
  actuators:
    - id: 0
      name: fl_hip_act
      type: position_servo
      kp: 100.0
      gear: [1.0, 0.0, ...]
      ctrl_range: [-0.7, 0.7]
physics_parameters:
  timestep: 0.01
  gravity: [0.0, 0.0, -9.81]
  integrator: euler
  solver: pgs
  iterations: 50
  tolerance: 1.0e-06
```

### Why Track Control Parameters?

**Critical for reproducibility and comparison:**
- Different `kp` gains = fundamentally different experiments
- Actuator type (position vs torque) drastically affects learning
- Physics timestep impacts sim2real transfer
- Required for comparing smooth locomotion experiments (exp001)

### Using Environment Config Utilities

```python
from robot_lab.utils.mujoco_config import extract_environment_config
from robot_lab.experiments import ExperimentTracker

# Extract config from environment
env = gym.make("A1Quadruped-v0")
env_config = extract_environment_config(env)

# Log with experiment tracker
tracker = ExperimentTracker("my_experiment", "run_1")
tracker.log_env_config(env_config)
```

## File Organization

Each run directory contains:
```
<run_id>/
├── metadata.yaml            # Run metadata and configuration (YAML)
├── environment_config.yaml  # Control & physics parameters (YAML)
├── hyperparameters.yaml     # Algorithm hyperparameters (YAML)
├── system_info.yaml         # System and environment info (YAML)
├── metrics.json             # Time-series metrics (JSON for append performance)
├── training.log             # Loguru logs
├── <algo>_<env>.zip         # Model weights
├── <algo>_<env>_vecnorm.pkl # Normalization stats
├── evaluations.npz          # Evaluation callback results
└── results.png              # Visualization plot (viz runs only)
```

### Experiment Tracking (Phase 0+)

For structured experiments using ExperimentTracker:
```
data/experiments/<experiment_name>/
├── runs/
│   └── <run_name>/
│       ├── metadata.yaml            # Run metadata (YAML)
│       ├── hyperparameters.yaml     # RL hyperparameters (YAML, validated)
│       ├── environment_config.yaml  # Control/physics params (YAML)
│       ├── system_info.yaml         # System info (YAML)
│       └── metrics.json             # Time-series metrics (JSON)
└── analysis/
    └── ...
```

### YAML vs JSON

**YAML files** (all configuration and metadata):
- `metadata.yaml`: Run metadata, status, tags, artifacts
- `hyperparameters.yaml`: RL algorithm hyperparameters (validated)
- `environment_config.yaml`: Control and physics parameters
- `system_info.yaml`: System and environment info
- Human-readable, easy to edit and review
- Better for version control (cleaner diffs)
- Supports comments for inline documentation

**JSON files** (only for time-series data):
- `metrics.json`: Time-series metrics (appended frequently)
- JSON is more efficient for append operations
- Better performance for programmatic parsing of large datasets
