# Getting Started: Running & Evaluating Experiments

This guide walks you through training a policy, monitoring it live, recording a rollout video, and reading the results — end to end.

---

## Prerequisites

```bash
uv sync           # install all dependencies
make verify-gpu   # confirm PyTorch sees your GPU (optional but recommended)
```

---

## 1. Quick Sanity Check

Before running a long experiment, verify everything is wired up with a 500-step smoke run:

```bash
robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42
```

You should see a rich training table printed, then:

```
✓ Training completed successfully!
  Model saved to: models/sac_mountaincarcontinuous_parallel.zip
  VecNormalize saved to: models/sac_mountaincarcontinuous_vecnorm.pkl
```

Both files must exist as a pair — the vecnorm statistics are required for correct evaluation.

---

## 2. Running an Experiment

Experiment plans live in `experiments/`. Each `.md` file contains the exact commands to run each variant.

### Example: Smooth Locomotion (Experiment 001)

The full plan is in [experiments/0_foundations/001_smooth_locomotion.md](../../experiments/0_foundations/001_smooth_locomotion.md). The baseline variant:

```bash
robot-lab train \
  --env A1Quadruped-v0 \
  --algo SAC \
  --seed 42 \
  --env-config experiments/0_foundations/configs/a1_kp100_baseline.yaml \
  --output-dir data/experiments/smooth_locomotion/exp0_baseline
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--seed` | Reproducibility — always set this |
| `--env-config` | Per-experiment environment overrides (control params, physics) |
| `--output-dir` | Where models, logs, and metadata are written |
| `--eval-freq` | How often (in timesteps) to run evaluation episodes |
| `--eval-episodes` | Number of episodes per evaluation |
| `--checkpoints` | Save model snapshots every `--save-freq` steps |

For a robust run with evaluation data:

```bash
robot-lab train \
  --env Walker2d-v5 \
  --algo SAC \
  --seed 42 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --checkpoints \
  --output-dir data/experiments/walker2d_baseline
```

---

## 3. Monitoring Training with TensorBoard

In a second terminal, launch TensorBoard while training is running:

```bash
robot-lab tensorboard --logdir data/experiments/walker2d_baseline/logs
```

Or point at the default log directory:

```bash
robot-lab tensorboard
```

Open [http://localhost:6006](http://localhost:6006). The key scalars to watch:

- `rollout/ep_rew_mean` — mean episode reward (primary performance signal)
- `rollout/ep_len_mean` — mean episode length  
- `train/learning_rate` — should stay constant unless scheduled
- `smoothness/action_delta_norm` — action smoothness metric (tracked automatically)

---

## 4. Output Files

After training, your `--output-dir` (or the default `./`) contains:

```
data/experiments/walker2d_baseline/
├── models/
│   ├── sac_walker2d_parallel.zip      ← trained model
│   ├── sac_walker2d_vecnorm.pkl       ← observation/reward normalisation stats
│   └── best/
│       └── best_model.zip             ← checkpoint with highest eval reward
├── logs/
│   └── sac_walker2d_parallel/         ← TensorBoard event files
└── experiments/
    └── walker2d_baseline/
        └── runs/
            └── <run_id>/
                ├── metadata.json      ← git commit, system info, timestamps
                ├── metrics.json       ← reward, episode length, smoothness series
                ├── hyperparameters.json
                └── system_info.json
```

> **Important**: `model.zip` and `vecnorm.pkl` must stay together. Loading a model without its vecnorm produces garbage observations.

---

## 5. Recording a Rollout Video

Once training is complete, record an MP4 of the policy behaving in the environment:

```bash
robot-lab visualize \
  --env Walker2d-v5 \
  --algo SAC \
  --model-path data/experiments/walker2d_baseline/models/sac_walker2d_parallel.zip \
  --vecnorm-path data/experiments/walker2d_baseline/models/sac_walker2d_vecnorm.pkl \
  --output-dir data/experiments/walker2d_baseline \
  --episodes 3 \
  --record-video
```

The MP4 is written to:

```
data/experiments/walker2d_baseline/videos/sac_walker2d/sac_walker2d.mp4
```

To skip recording and just watch live (requires a display):

```bash
robot-lab visualize \
  --env Walker2d-v5 \
  --algo SAC \
  --no-record-video
```

---

## 6. Reading Experiment Results

### From JSON (programmatic)

```python
import json
from pathlib import Path

run_dir = Path("data/experiments/walker2d_baseline/experiments/walker2d_baseline/runs")
run = sorted(run_dir.iterdir())[-1]   # most recent run

metrics = json.loads((run / "metrics.json").read_text())
smoothness = metrics.get("smoothness_action_delta_norm", [])

print(f"Final mean reward: {metrics['mean_reward'][-1]:.1f}")
print(f"Mean episode smoothness: {sum(smoothness)/len(smoothness):.4f}")
```

### From TensorBoard (visual)

After training, re-point TensorBoard at the log directory:

```bash
robot-lab tensorboard --logdir data/experiments/walker2d_baseline/logs
```

Compare multiple experiments by pointing at the parent directory:

```bash
robot-lab tensorboard --logdir data/experiments/
```

TensorBoard will split experiment variants into separate colour-coded runs.

---

## 7. Comparing Experiment Variants

The smoothness experiment (001) trains 12 variants. Once you have run at least two, compare them:

```python
import json
from pathlib import Path

base_dir = Path("data/experiments/smooth_locomotion")

results = {}
for exp_dir in sorted(base_dir.iterdir()):
    metrics_file = next(exp_dir.glob("**/metrics.json"), None)
    if metrics_file:
        m = json.loads(metrics_file.read_text())
        smoothness = m.get("smoothness_action_delta_norm", [])
        results[exp_dir.name] = {
            "final_reward": m["mean_reward"][-1] if m.get("mean_reward") else None,
            "mean_smoothness": sum(smoothness) / len(smoothness) if smoothness else None,
        }

for name, r in results.items():
    print(f"{name:30s}  reward={r['final_reward']:.1f}  smoothness={r['mean_smoothness']:.4f}")
```

---

## 8. Next Steps

| I want to... | Go to... |
|---|---|
| Run the smooth locomotion variants | [experiments/0_foundations/001_smooth_locomotion.md](../../experiments/0_foundations/001_smooth_locomotion.md) |
| Understand the experiment tracking schema | [docs/general/experiment_schema.md](experiment_schema.md) |
| Add a new environment | [docs/general/adding_environments.md](adding_environments.md) |
| Understand the full research plan | [docs/user/PLAN.md](../user/PLAN.md) |
| See what's planned next | [docs/user/TODO.md](../user/TODO.md) |
