# Running Experiments

This guide covers how to run the Phase 0 smooth locomotion experiments using the
`run-experiment` CLI command and how to monitor, visualize, and interpret results.

---

## Prerequisites

```bash
uv sync
make verify-gpu   # confirm GPU is visible
robot-lab info    # sanity check CLI is working
```

Verify the A1 environment is available:

```bash
robot-lab env-info --env A1Quadruped-v0
```

---

## The `run-experiment` Command

`run-experiment` reads a YAML campaign file and orchestrates one or more training runs.
It handles environment parameter injection (kp, kd), action wrappers (EMA filter,
frameskip), metric collection, and JSON result tracking automatically.

```
robot-lab run-experiment <config.yaml> [OPTIONS]

Arguments:
  config            Path to YAML experiment configuration file  [required]

Options:
  -e, --experiment  Specific experiment ID to run (default: run all)
  -o, --output-dir  Base output directory (default: data/experiments)
  --dry-run         Print execution plan without running
  --list, -l        List all experiments in config and exit
```

---

## Phase 0.5 — Smooth Locomotion Campaign

All 12 experiment variants for Experiment 001 are defined in:

```
experiments/0_foundations/configs/smooth_locomotion_experiments.yaml
```

### Step 1 — Check what will run

```bash
# List all experiments and their status
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  --list

# Preview execution plan for the baseline (no training runs)
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  --dry-run -e exp0_baseline
```

### Step 2 — Run the baseline first

Always run the baseline before any variants. It establishes the reference metrics.

```bash
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp0_baseline
```

Output goes to: `data/experiments/0_foundations/exp0_baseline/`

### Step 3 — Run by group

The experiments are organised into groups. Run one group at a time, review results
between groups, and abort early if a group shows no signal.

#### Group 1 — PD Gain Tuning

Tests the effect of position control stiffness on smoothness and learning.

```bash
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp1a_kp100

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp1b_kp20
```

#### Group 2 — Damping Tuning

Tests joint damping (kd) at the baseline stiffness (kp=50).

```bash
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp2a_kd5

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp2b_kd10

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp2c_kd20
```

#### Group 3 — Control Frequency (Frameskip)

Tests reduced control rates by repeating actions. The policy is queried less
frequently, which enforces temporal consistency.

```bash
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp3a_frameskip1

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp3b_frameskip2

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp3c_frameskip3

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp3d_frameskip4
```

#### Group 4 — EMA Action Filtering

Trains the policy with an exponential moving average filter applied *inside* the
environment. The policy learns the filtered dynamics, so results are meaningful.

```bash
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp4a_filter_alpha07

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp4b_filter_alpha05

robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  -e exp4c_filter_alpha03
```

### Step 4 — Run the entire campaign (unattended)

Once you have confirmed the baseline works, you can queue all enabled experiments
in one shot:

```bash
robot-lab run-experiment \
  experiments/0_foundations/configs/smooth_locomotion_experiments.yaml \
  --output-dir data/experiments/smooth_locomotion
```

> **Note**: Groups 5 (reward engineering) are disabled in the YAML (`enabled: false`)
> and will be skipped automatically.

---

## Monitoring Training

In a second terminal, point TensorBoard at the experiment output directory:

```bash
# Monitor a single run while it trains
robot-lab tensorboard --logdir data/experiments/0_foundations/exp0_baseline/logs

# Compare all smooth locomotion runs side by side
robot-lab tensorboard --logdir data/experiments/0_foundations
```

Open [http://localhost:6006](http://localhost:6006).

Key scalars to watch:

| Scalar | What it tells you |
|--------|-------------------|
| `rollout/ep_rew_mean` | Primary performance signal |
| `rollout/ep_len_mean` | Whether the robot stays upright |
| `smoothness/action_delta_norm` | Action jitter (∑‖aₜ−aₜ₋₁‖² per episode, lower = smoother) |
| `eval/mean_reward` | Deterministic evaluation reward |

---

## Output Structure

Each experiment run produces:

```
data/experiments/0_foundations/<exp_id>/
├── models/
│   ├── sac_a1quadruped_parallel.zip      ← trained model
│   ├── sac_a1quadruped_vecnorm.pkl       ← VecNormalize stats (keep with model)
│   └── best/
│       └── best_model.zip                ← checkpoint with highest eval reward
├── logs/
│   └── sac_a1quadruped_parallel/         ← TensorBoard event files
└── experiments/
    └── 0_foundations/
        └── runs/
            └── <run_id>/
                ├── metadata.json          ← env config, git commit, timestamps
                ├── metrics.json           ← reward / smoothness time series
                ├── hyperparameters.json
                ├── system_info.json
                └── env_config.json        ← the exact control_params used
```

> **Important**: `model.zip` and `vecnorm.pkl` must travel together.
> Loading a model without its vecnorm produces garbage observations.

---

## Recording Videos

After training, record a rollout video to qualitatively assess motion smoothness:

```bash
robot-lab visualize \
  --env A1Quadruped-v0 \
  --algo SAC \
  --model-path data/experiments/0_foundations/exp0_baseline/models/sac_a1quadruped_parallel.zip \
  --vecnorm-path data/experiments/0_foundations/exp0_baseline/models/sac_a1quadruped_vecnorm.pkl \
  --output-dir data/experiments/0_foundations/exp0_baseline \
  --episodes 3 \
  --record-video
```

Video is saved to:
```
data/experiments/0_foundations/exp0_baseline/videos/sac_a1quadruped/sac_a1quadruped.mp4
```

Repeat for each experiment variant to build a visual comparison library.

---

## Reading Computed Metrics

After each run the runner automatically evaluates the 12 smoothness/performance
metrics defined in the YAML and saves them to `metadata.json`. To compare across
all runs:

```python
import json
from pathlib import Path

base = Path("data/experiments/0_foundations")

rows = []
for exp_dir in sorted(base.iterdir()):
    metadata_files = list(exp_dir.glob("**/metadata.json"))
    for mf in metadata_files:
        meta = json.loads(mf.read_text())
        computed = meta.get("computed_metrics", {}).get("computed_metrics", {})
        rows.append({
            "experiment": exp_dir.name,
            "jerk_mean":        computed.get("jerk_mean"),
            "action_delta_mean": computed.get("action_delta_mean"),
            "forward_distance":  computed.get("forward_distance"),
            "fall_rate":         computed.get("fall_rate"),
        })

for r in rows:
    print(
        f"{r['experiment']:25s}  "
        f"jerk={r['jerk_mean']:.3f}  "
        f"delta={r['action_delta_mean']:.3f}  "
        f"dist={r['forward_distance']:.1f}  "
        f"falls={r['fall_rate']:.0%}"
    )
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| List all experiments | `robot-lab run-experiment <config> --list` |
| Preview a run | `robot-lab run-experiment <config> -e <id> --dry-run` |
| Run one experiment | `robot-lab run-experiment <config> -e <id>` |
| Run entire campaign | `robot-lab run-experiment <config>` |
| Watch training live | `robot-lab tensorboard --logdir data/experiments/...` |
| Record rollout video | `robot-lab visualize --env A1Quadruped-v0 --algo SAC --record-video ...` |

---

## Related

- [Experiment 001 plan](../../experiments/0_foundations/001_smooth_locomotion.md) — background, hypotheses, and results tables
- [Experiment YAML config](../../experiments/0_foundations/configs/smooth_locomotion_experiments.yaml) — all 12 variants
- [Getting started guide](getting_started.md) — single-run training walkthrough
- [Research plan](../user/PLAN.md) — 40-week roadmap
