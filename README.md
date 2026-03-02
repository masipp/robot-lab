# robot-lab

A reinforcement learning playground for robotic environments with experiment automation and AI-driven research capabilities.

## Features

✅ **Modern Package Structure**: Installable with `uv` or `pip`  
✅ **CLI Interface**: Train and visualize with simple commands (powered by Typer)  
✅ **Multiple Algorithms**: SAC (continuous) and PPO (discrete & continuous)  
✅ **Parallel Training**: Multi-environment vectorization for speed  
✅ **Experiment Automation**: JSON-based experiment campaigns with reproducibility  
✅ **Comprehensive Tracking**: JSON-based experiment database with metadata  
✅ **AI Planner (Groundwork)**: Foundation for AI-driven experiment design  
✅ **Custom Environments**: Gripper control and quadruped locomotion  
✅ **TensorBoard Integration**: Real-time training monitoring  

## Installation

### Quick Start with Makefile (Recommended)

**For GTX 1080 (default configuration):**

```bash
# Install everything (includes PyTorch 2.9.1+cu126)
make install

# Verify GPU setup
make verify-gpu
```

**For other GPUs:**

```bash
# Install core dependencies first
make install

# Then install PyTorch for your GPU
make install-torch-cuda126    # For RTX 20xx/30xx/40xx (latest PyTorch)
make install-torch-cpu        # For CPU only

# Verify
make verify-gpu
```

### Manual Installation

#### Using uv (recommended)

```bash
# Install everything (GTX 1080 configuration)
uv sync

# For other GPUs, update pyproject.toml first (see GPU Setup below)
```

### Using pip

```bash
# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev,robot-descriptions]"
```

### GPU Setup (Important!)

**Default Configuration: GTX 1080 (PyTorch 2.9.1+cu126)**

The project is configured by default for GTX 1080 with PyTorch 2.9.1+cu126. Simply run:

```bash
make install && make verify-gpu
```

Or with uv directly:

```bash
uv sync && .venv/bin/python docs/verify_gpu_setup.py
```

**For RTX series or other GPUs:**

Use the Makefile shortcuts to reinstall PyTorch:

```bash
make install-torch-cuda126    # RTX series (latest PyTorch)
make install-torch-cpu        # CPU only
```

Or manually update [pyproject.toml](pyproject.toml):

```toml
# Change these versions and CUDA variant
dependencies = [
    # ... other deps ...
    "torch==2.5.0",           # Latest version for RTX
    "torchvision==0.20.0",
]

# Update the index URL
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"  # Or cu121, cu118, cpu
```

Then reinstall:

```bash
uv sync --reinstall-package torch
```

📖 **See [docs/GPU_SETUP.md](docs/GPU_SETUP.md) for detailed GPU setup instructions**, including:
- How to check your GPU compute capability
- PyTorch version compatibility matrix
- Troubleshooting common issues
- Driver installation guide

## Quick Start

### Basic Training

Train a SAC agent on Walker2d:

```bash
robot-lab train --env Walker2d-v5 --algo SAC
```

Train with custom configuration:

```bash
robot-lab train --env HalfCheetah-v5 --algo SAC \
  --config my_config.json \
  --seed 123 \
  --output-dir ./my_experiment
```

### Visualizing Policies

Visualize a trained policy:

```bash
robot-lab visualize --env Walker2d-v5 --algo SAC
```

Run without rendering (headless):

```bash
robot-lab visualize --env Walker2d-v5 --algo SAC --no-render
```

### TensorBoard Monitoring

Launch TensorBoard to view training progress:

```bash
robot-lab tensorboard --logdir logs --port 6006
```

### Get Information

Display available commands and environments:

```bash
robot-lab info
```

## Project Structure

```
robot-lab/
├── src/robot_lab/              # Main package
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── training.py             # Core training logic
│   ├── visualization.py        # Policy visualization
│   ├── config.py               # Configuration management
│   ├── configs/                # Hyperparameter JSON files
│   │   ├── default_sac.json
│   │   ├── default_ppo.json
│   │   ├── walker2d_sac.json
│   │   └── ...
│   ├── envs/                   # Custom environments
│   │   ├── __init__.py
│   │   ├── gripper.py
│   │   └── locomotion/
│   │       └── quadruped.py
│   ├── experiments/            # Experiment automation
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic models
│   │   ├── tracker.py          # Experiment tracking
│   │   ├── results_db.py       # JSON database
│   │   ├── spec_templates.py   # Example specs
│   │   └── ai_planner.py       # AI planner (stub)
│   └── utils/
│       ├── __init__.py
│       └── paths.py            # Path utilities
├── docs/
│   └── experiment_schema.md    # Experiment spec documentation
├── experiments/                # Experiment outputs (git-ignored)
├── models/                     # Trained models (git-ignored)
├── logs/                       # TensorBoard logs (git-ignored)
├── pyproject.toml              # Project configuration
└── README.md
```

## Supported Environments

### Built-in Gymnasium/MuJoCo
- **Walker2d-v5**: Bipedal walking robot
- **HalfCheetah-v5**: 2D running cheetah
- **MountainCarContinuous-v0**: Continuous mountain car

### Custom Environments
- **GripperEnv-v0**: MuJoCo gripper control
- **A1Quadruped-v0**: Unitree A1 quadruped locomotion (requires `robot_descriptions`)

## Supported Algorithms

| Algorithm | Action Space | Description |
|-----------|-------------|-------------|
| **SAC** | Continuous only | Soft Actor-Critic - Sample efficient, off-policy |
| **PPO** | Discrete & Continuous | Proximal Policy Optimization - Stable, on-policy |

## Experiment Automation

### Using Templates

Create experiments from templates:

```python
from robot_lab.experiments import get_template
import json

# Get hyperparameter sweep template
spec = get_template("hyperparam_sweep")

# Customize
spec["experiment_metadata"]["name"] = "my_sac_sweep"
spec["environments"] = [{"name": "HalfCheetah-v5", "config_overrides": {}}]

# Save
with open("my_experiment.json", "w") as f:
    json.dump(spec, f, indent=2)
```

Available templates:
- `hyperparam_sweep`: Systematic hyperparameter search
- `algorithm_comparison`: Compare multiple algorithms
- `quick_test`: Fast test run with minimal training

### Experiment Tracking

Track experiments with JSON-based metadata:

```python
from robot_lab.experiments import ExperimentTracker

tracker = ExperimentTracker("my_experiment", "run_001")
tracker.log_params({"learning_rate": 0.001, "batch_size": 256})
tracker.log_metrics({"reward": 150.0, "episode_length": 500}, step=10000)
tracker.set_status("completed")
```

### Results Database

Store and query results:

```python
from robot_lab.experiments import ResultsDatabase

db = ResultsDatabase()
best_run = db.get_best_run("my_experiment", metric="final_mean_reward")
stats = db.get_statistics("my_experiment")
print(f"Mean reward: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

## Configuration Files

Hyperparameters are defined in JSON files under `src/robot_lab/configs/`:

```json
{
  "environment": "Walker2d-v5",
  "algorithm": "SAC",
  "num_envs": 8,
  "total_timesteps": 350000,
  "hyperparameters": {
    "policy": "MlpPolicy",
    "learning_rate": 0.001,
    "buffer_size": 300000,
    "batch_size": 256,
    ...
  },
  "vec_normalize": {
    "norm_obs": true,
    "norm_reward": true,
    ...
  }
}
```

See `docs/experiment_schema.md` for complete documentation.

## AI-Driven Experimentation (Future)

The `AIExperimentPlanner` class provides groundwork for AI-driven research:

```python
# Future capability (stub implementation)
from robot_lab.experiments import AIExperimentPlanner

planner = AIExperimentPlanner()

# Generate experiment from natural language
spec = planner.generate_from_natural_language(
    "Compare SAC and PPO on locomotion tasks with 5 seeds each"
)

# Adaptive experiment design
spec = planner.design_adaptive_experiment(
    environment="Walker2d-v5",
    algorithm="SAC",
    budget_hours=24
)
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
ruff check src/
ruff format src/
```

## Output Files

After training, you'll find:

- **Models**: `models/{algo}_{env}_parallel.zip`
- **VecNormalize**: `models/{algo}_{env}_vecnorm.pkl`
- **Best Model**: `models/best/`
- **TensorBoard Logs**: `logs/{algo}_{env}_parallel/`
- **Visualization**: `{env}_learned_policy_results.png`
- **Experiments**: `experiments/{experiment_name}/`

## Customization

### Adding Custom Environments

1. Create your environment in `src/robot_lab/envs/`
2. Register it in `src/robot_lab/envs/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='robot_lab.envs.my_env:MyEnv',
    max_episode_steps=1000,
)
```

3. Create a hyperparameter config in `src/robot_lab/configs/myenv_sac.json`

### Creating Custom Hyperparameter Configs

Copy an existing config and modify:

```bash
cp src/robot_lab/configs/walker2d_sac.json src/robot_lab/configs/myenv_sac.json
# Edit the new file with your parameters
```

## Examples

### Example 1: Quick Training Run

```bash
# Train for quick test (uses MountainCarContinuous)
robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42
```

### Example 2: Full Training with Evaluation

```bash
robot-lab train \
  --env Walker2d-v5 \
  --algo SAC \
  --seed 42 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --checkpoints \
  --save-freq 50000
```

### Example 3: Custom Output Directory

```bash
# Train with custom output location
robot-lab train \
  --env HalfCheetah-v5 \
  --algo PPO \
  --output-dir ./my_experiments/halfcheetah_ppo_001

# Visualize from custom location
robot-lab visualize \
  --env HalfCheetah-v5 \
  --algo PPO \
  --output-dir ./my_experiments/halfcheetah_ppo_001
```

### Example 4: Experiment Automation

```python
from robot_lab.experiments import get_template, ExperimentTracker
import json

# Create experiment spec
spec = get_template("algorithm_comparison")
spec["experiment_metadata"]["name"] = "locomotion_study"
spec["environments"] = [
    {"name": "Walker2d-v5", "config_overrides": {}},
    {"name": "HalfCheetah-v5", "config_overrides": {}}
]

# Save for future use
with open("locomotion_study.json", "w") as f:
    json.dump(spec, f, indent=2)
```

## Troubleshooting

### MuJoCo Installation Issues

If you encounter MuJoCo errors:

```bash
# Ensure mujoco is properly installed
pip uninstall gymnasium mujoco
pip install gymnasium[mujoco]
```

### Import Errors

If you get import errors after installation:

```bash
# Reinstall in editable mode
pip install -e .
```

### Missing Dependencies

```bash
# Install all dependencies including optional ones
pip install -e ".[dev,robot-descriptions]"
```

## Contributing

Contributions are welcome! Areas for future development:

1. **Experiment Manager**: Implement parallel experiment execution
2. **Report Generation**: Automated markdown reports with plots
3. **AI Planner**: LLM integration for experiment generation
4. **Additional Algorithms**: TD3, DDPG, etc.
5. **Curriculum Learning**: Progressive difficulty adjustment
6. **Multi-task Learning**: Train on multiple environments simultaneously

## License

MIT License

## Acknowledgments

- Built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Uses [Gymnasium](https://gymnasium.farama.org/) for environments
- CLI powered by [Typer](https://typer.tiangolo.com/)
- Validation with [Pydantic](https://pydantic-docs.helpmanual.io/)

