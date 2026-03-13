# robot-lab

A reinforcement learning research playground built on Stable-Baselines3 and Gymnasium.  
Designed for rapid experimentation, systematic curriculum learning research, and sim2real transfer work — all runnable on consumer hardware (GTX 1080 / CUDA 12.6).

## Features

- **CLI Interface** — `train`, `visualize`, `tensorboard`, `info` commands via Typer
- **Algorithms** — SAC (continuous) and PPO (discrete & continuous)
- **Parallel Training** — `SubprocVecEnv` with configurable workers; `DummyVecEnv` for evaluation
- **Video Recording** — `robot-lab visualize --record-video` writes MP4 via imageio/ffmpeg
- **Action Wrappers** — `ActionFilterWrapper` base class for custom action post-processing
- **Smoothness Metrics** — `ActionSmoothnessMetricPlugin` tracks ∑‖aₜ−aₜ₋₁‖² per episode automatically
- **TensorBoard Integration** — metrics logged during training, launchable via CLI
- **Experiment Tracking** — JSON-based per-run metadata, hyperparameters, system info, and metrics
- **Config Hierarchy** — `{env}_{algo}.json` → `default_{algo}.json` → `default.json` fallback chain
- **Custom Environments** — `GripperEnv-v0`, `A1Quadruped-v0` (requires `robot_descriptions`)
- **Reproducibility** — seed propagation, git commit capture, VecNorm stats saved alongside every model

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- NVIDIA GPU with appropriate driver (see [docs/setup/GPU_SETUP.md](docs/setup/GPU_SETUP.md))

### Quick Start (GTX 1080 / CUDA 12.6)

```bash
git clone <repo>
cd robot-lab
uv sync
make verify-gpu
```

### Other GPU Configurations

```bash
# RTX 20xx/30xx/40xx (CUDA 12.6, latest PyTorch)
make install-torch-cuda126

# CPU only
make install-torch-cpu
```

To use a different CUDA version, edit the index URL in `pyproject.toml`:

```toml
[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu118"  # e.g. cu118, cu121, cu126
```

Then reinstall: `uv sync --reinstall-package torch`

📖 Full GPU setup instructions: [docs/setup/GPU_SETUP.md](docs/setup/GPU_SETUP.md)

---

## Quick Start

### Train

```bash
# Smoke test (fast, CPU-friendly)
robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42

# Full locomotion run
robot-lab train --env Walker2d-v5 --algo SAC --seed 42 \
  --eval-freq 10000 --eval-episodes 10 \
  --output-dir data/experiments/walker2d_baseline
```

### Visualize & Record Video

```bash
robot-lab visualize \
  --env Walker2d-v5 --algo SAC \
  --model-path data/experiments/walker2d_baseline/models/sac_walker2d_parallel.zip \
  --vecnorm-path data/experiments/walker2d_baseline/models/sac_walker2d_vecnorm.pkl \
  --record-video

# Watch live without recording
robot-lab visualize --env Walker2d-v5 --algo SAC --no-record-video
```

### TensorBoard

```bash
robot-lab tensorboard --logdir data/experiments/walker2d_baseline/logs
robot-lab tensorboard --logdir data/experiments/   # compare all runs
```

### Environment & Command Info

```bash
robot-lab info
```

📖 **Full walkthrough**: [docs/general/getting_started.md](docs/general/getting_started.md)

---

## Project Structure

```
robot-lab/
├── robot_lab/                  # Main package
│   ├── cli.py                  # Typer CLI entry point
│   ├── training.py             # Training pipeline (SubprocVecEnv, callbacks)
│   ├── visualization.py        # visualize() MP4 recording; visualize_policy() plots
│   ├── config.py               # Config hierarchy loader (importlib.resources)
│   ├── wrappers.py             # ActionFilterWrapper base class + built-in filters
│   ├── configs/                # Bundled hyperparameter JSON files
│   │   ├── default.json
│   │   ├── default_sac.json
│   │   ├── default_ppo.json
│   │   ├── walker2d_sac.json
│   │   └── ...
│   ├── envs/                   # Custom environment registration
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── locomotion/         # A1Quadruped
│   │   └── manipulation/       # GripperEnv
│   ├── experiments/            # Tracking & automation
│   │   ├── __init__.py         # ExperimentTracker, ResultsDatabase, get_template
│   │   ├── schemas.py          # Pydantic experiment spec models
│   │   ├── tracker.py          # Per-run JSON tracking
│   │   ├── results_db.py       # Aggregation & query
│   │   ├── spec_templates.py   # hyperparam_sweep, algorithm_comparison, quick_test
│   │   ├── ai_planner.py       # Stub for future LLM-driven experiment generation
│   │   └── plugins/            # Metric & metadata plugin system
│   │       ├── __init__.py     # PluginRegistry with on_step/on_episode_end/on_eval
│   │       ├── base.py         # MetricsPlugin, MetadataPlugin ABC
│   │       └── defaults.py     # Built-in plugins (reward, smoothness, system info)
│   └── utils/
│       ├── paths.py            # get_models_dir, get_logs_dir (respects --output-dir)
│       └── ...
├── experiments/                # Experiment documentation (tracked in git)
│   ├── README.md
│   └── 0_foundations/
│       ├── 001_smooth_locomotion.md
│       └── configs/            # Per-experiment environment YAML overrides
├── data/                       # All runtime output (git-ignored)
│   └── experiments/            # Training outputs, models, logs, videos
├── docs/
│   ├── general/
│   │   ├── getting_started.md  # ← Start here
│   │   ├── experiment_schema.md
│   │   ├── adding_environments.md
│   │   └── metadata_system.md
│   ├── setup/
│   │   └── GPU_SETUP.md
│   └── user/
│       ├── PLAN.md             # 40-week research roadmap
│       └── TODO.md             # Current sprint goals
├── tests/
├── pyproject.toml
└── Makefile
```

---

## Supported Environments

### Gymnasium / MuJoCo (Built-in)

| Environment | Algorithm | Notes |
|---|---|---|
| `MountainCarContinuous-v0` | SAC | Fast smoke test (~30s) |
| `Walker2d-v5` | SAC | Phase 0 locomotion baseline |
| `HalfCheetah-v5` | SAC | Speed-focused locomotion |
| `Hopper-v5` | SAC | Single-leg stability |
| `Ant-v5` | SAC | Quadruped (simplified) |
| `CartPole-v1` | PPO | Classic discrete baseline |

### Custom Environments

| Environment | Notes |
|---|---|
| `A1Quadruped-v0` | Unitree A1 — requires `uv sync --extra robot-descriptions` |
| `GripperEnv-v0` | MuJoCo gripper manipulation |

---

## Algorithms

| Algorithm | Action Space | Description |
|---|---|---|
| **SAC** | Continuous | Soft Actor-Critic — off-policy, sample efficient, entropy-regularised |
| **PPO** | Discrete & Continuous | Proximal Policy Optimization — on-policy, stable, easy to tune |

---

## Output Files

Each training run writes:

```
{output_dir}/
├── models/
│   ├── {algo}_{env}_parallel.zip      ← model (always paired with vecnorm)
│   ├── {algo}_{env}_vecnorm.pkl       ← VecNormalize stats
│   └── best/best_model.zip            ← best eval checkpoint
├── logs/
│   └── {algo}_{env}_parallel/         ← TensorBoard event files
├── experiments/
│   └── {name}/runs/{run_id}/
│       ├── metadata.json              ← git commit, timestamps, env info
│       ├── metrics.json               ← reward, episode length, smoothness series
│       ├── hyperparameters.json
│       └── system_info.json
└── videos/
    └── {algo}_{env}/
        └── {algo}_{env}.mp4           ← recorded rollout (--record-video)
```

> **Note**: `model.zip` and `vecnorm.pkl` are always saved as a pair. `visualize` raises a `ValueError` if the vecnorm file is missing.

---

## Configuration Files

Hyperparameters live in `robot_lab/configs/`. Lookup order for e.g. `Walker2d-v5` + SAC:

1. `walker2d_sac.json` (env-specific)
2. `default_sac.json` (algorithm defaults)
3. `default.json` (universal fallback)

```json
{
  "algorithm": "SAC",
  "num_envs": 8,
  "total_timesteps": 350000,
  "hyperparameters": {
    "policy": "MlpPolicy",
    "learning_rate": 0.001,
    "buffer_size": 300000,
    "batch_size": 256
  },
  "vec_normalize": {
    "norm_obs": true,
    "norm_reward": true
  }
}
```

Pass a custom config with `--config my_overrides.json`. Full schema: [docs/general/experiment_schema.md](docs/general/experiment_schema.md).

---

## Plugin System

Plugins collect metrics and metadata automatically during training without touching training code.

| Plugin | Registry | What it does |
|---|---|---|
| `BasicRewardLogPlugin` | `metrics` | Logs episode reward to TensorBoard and tracker |
| `ActionSmoothnessMetricPlugin` | `metrics` | Tracks ∑‖aₜ−aₜ₋₁‖² → `smoothness/action_delta_norm` in TensorBoard |
| `SystemMetadataPlugin` | `metadata` | Captures GPU info, Python version, git commit at run start |

Adding a plugin:

```python
from robot_lab.experiments.plugins.base import MetricsPlugin
from robot_lab.experiments.plugins import get_plugin_registry

class MyPlugin(MetricsPlugin):
    def on_step(self, context: dict) -> None:
        pass  # context: actions, observations, rewards, dones, sb3_logger, tracker

get_plugin_registry("metrics").register(MyPlugin())
```

---

## Action Wrappers

`ActionFilterWrapper` is a base class for applying custom post-processing to policy outputs:

```python
from robot_lab.wrappers import ActionFilterWrapper

class MyFilter(ActionFilterWrapper):
    def _apply_filter(self, action):
        # your filtering logic here
        return action
```

Built-in wrappers: `ActionRepeatWrapper`, `ExponentialMovingAverageFilter`, `LowPassFilter`.

---

## Experiment Tracking (Programmatic API)

```python
from robot_lab.experiments import ExperimentTracker, ResultsDatabase, get_template

tracker = ExperimentTracker("my_experiment", "run_001")
tracker.update("hyperparameters", {"lr": 3e-4, "batch_size": 256})
tracker.update("metrics", {"mean_reward": [120.0, 180.0, 210.0]})

db = ResultsDatabase()
best = db.get_best_run("my_experiment", metric="final_mean_reward")

spec = get_template("algorithm_comparison")  # also: "hyperparam_sweep", "quick_test"
```

---

## Development

### Testing

```bash
make test-fast      # all tests except slow training tests (~5s)
make test           # full suite
make test-coverage  # with HTML coverage report
```

### Code Quality

```bash
make lint           # ruff check
make format         # ruff format + check --fix
```

### Adding a Custom Environment

See [docs/general/adding_environments.md](docs/general/adding_environments.md). Short version:

1. Implement `gym.Env` subclass in `robot_lab/envs/`
2. Register in `robot_lab/envs/__init__.py`
3. Create `robot_lab/configs/{envbase}_{algo}.json`

---

## Research Roadmap

This project follows a 40-week curriculum learning and sim2real research plan:

| Phase | Weeks | Focus |
|---|---|---|
| **0 — Foundations** | 1–4 | Clean baselines, reproducibility, experiment infrastructure ✅ |
| **0.5 — Smooth Control** | 4–6 | Trajectory generation, action filtering, PD gain exploration |
| **1 — Curriculum Basics** | 5–10 | Manual vs adaptive curriculum, sample efficiency |
| **2 — Representation** | 11–18 | Goal-conditioned RL, curriculum + domain randomisation |
| **3 — Sim2Real** | 19–30 | Morphology transfer, dynamics gap, transfer learning |
| **4 — Scientific** | 31–40 | Curriculum as exploration shaping, publication-grade results |

📖 Full plan: [docs/user/PLAN.md](docs/user/PLAN.md) | Current sprint: [docs/user/TODO.md](docs/user/TODO.md)

---

## License

MIT License

## Acknowledgments

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Typer](https://typer.tiangolo.com/) · [Rich](https://rich.readthedocs.io/) · [Loguru](https://loguru.readthedocs.io/)
- [imageio](https://imageio.readthedocs.io/) + [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) for video recording

---

## BMAD — AI-Assisted Planning

This project uses [BMAD](https://github.com/bmad-code-org) v6.0.4, an AI-assisted methodology for structured product planning, architecture, and development workflows. See the [BMAD documentation](https://github.com/bmad-code-org) for full usage details.

### Prerequisites

BMAD runs entirely inside VS Code via GitHub Copilot Chat — no separate binary to install. BMAD itself is already present in the `_bmad/` directory.

| Requirement | Notes |
|---|---|
| [VS Code](https://code.visualstudio.com/) | Any recent version |
| [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot) | Requires an active Copilot subscription |
| [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) | Enables prompt-file invocation |

Invoke workflows by attaching a prompt file from `.github/prompts/` in Copilot Chat, e.g.:

```
#file:.github/prompts/bmad-bmm-validate-prd.prompt.md
```
