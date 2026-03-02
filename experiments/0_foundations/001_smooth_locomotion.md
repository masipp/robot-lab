# Smooth Locomotion Experiments

**Goal:** Achieve smooth, natural quadruped locomotion via systematic control exploration

**Date:** February 17, 2026 | **Status:** Planning

---

## Background

A1Quadruped shows jerky motion (kp=100 position control). This affects sim2real transfer, energy efficiency, and task performance.

**Research Questions:**
1. What causes jerky motion? (Policy outputs vs control response)
2. Which control approach is smoothest? (Position gain tuning, filtering, torque control)
3. What are the tradeoffs? (Smoothness vs learnability vs compliance)

## Hypotheses

**H1:** High PD gains amplify policy noise → Reducing kp smooths motion without retraining  
**H2:** Policy outputs are noisy → Low-pass filtering smooths motion but adds lag  
**H3:** Torque control provides compliance but requires learning stability  
**H4:** Integration filters noise → Torque control produces smoother trajectories  
**Counter:** Noisy torques without PD damping may be less smooth

---

## Metrics (n=10 episodes)

**Smoothness:** `jerk_mean`, `jerk_max`, `accel_variance`  
**Actions:** `action_delta_mean`, `action_delta_max`, `action_frequency_power` (>10Hz FFT)  
**Performance:** `forward_distance`, `episode_length`, `energy_cost`  
**Stability:** `fall_rate`, `orientation_variance`, `base_height_variance`

---

## Configuration

All experiment variants are defined in a single file:  
**`experiments/0_foundations/configs/smooth_locomotion_experiments.yaml`**

This contains:
- **12 experiment variants** (baseline, 3 PD gains, 3 filters, torque, 3 reward variants)
- **Common parameters** (environment, algorithm, metrics)
- **Control parameters** for each variant
- **Training settings** (timesteps, seed)

Example structure:
```yaml
experiments:
  exp0_baseline:
    control_params: {kp: 100}
    training: {total_timesteps: 350000, seed: 42}
  exp1_kp50:
    control_params: {kp: 50}
    # ... etc
```

---

## Experiments

### Exp 0: Baseline (kp=100, position control)
Establish baseline metrics. Observe which joints are jerkiest.

```bash
robot-lab train --env A1Quadruped-v0 --algo SAC --seed 42 \
  --env-config experiments/0_foundations/configs/a1_kp100_baseline.yaml \
  --output-dir data/experiments/smooth_locomotion/exp0_baseline
```

**Results:** *(TBD)*


### Exp 1: Reduce PD Gain
Test kp ∈ {50, 20, 10}. Hypothesis: Lower gain → softer tracking → smoother motion.

```bash
# kp=50
robot-lab train --env A1Quadruped-v0 --algo SAC --seed 42 \
  --env-config experiments/0_foundations/configs/a1_kp50.yaml \
  --output-dir data/experiments/smooth_locomotion/exp1_kp50

# kp=20
robot-lab train --env A1Quadruped-v0 --algo SAC --seed 42 \
  --env-config experiments/0_foundations/configs/a1_kp20.yaml \
  --output-dir data/experiments/smooth_locomotion/exp1_kp20

# kp=10
robot-lab train --env A1Quadruped-v0 --algo SAC --seed 42 \
  --env-config experiments/0_foundations/configs/a1_kp10.yaml \
  --output-dir data/experiments/smooth_locomotion/exp1_kp10
```

| kp | jerk_mean | action_delta | fwd_dist | fall_rate |
|----|-----------|--------------|----------|-----------|
| 100 | - | - | - | - |
| 50 | - | - | - | - |
| 20 | - | - | - | - |
| 10 | - | - | - | - |

**Conclusion:** *(TBD)*


### Exp 2: Action Filtering
Test exponential moving average: `action_t = α * policy_output + (1-α) * action_{t-1}`  
Use baseline model (no retraining). Test α ∈ {0.7, 0.5, 0.3}.

| α | jerk_mean | action_delta | fwd_dist | fall_rate | lag |
|---|-----------|--------------|----------|-----------|-----|
| 1.0 | - | - | - | - | None |
| 0.7 | - | - | - | - | - |
| 0.5 | - | - | - | - | - |
| 0.3 | - | - | - | - | - |

**Conclusion:** *(TBD)*


### Exp 3: Torque Control
Replace position actuators with torque motors (ctrlrange [-10, 10] Nm). Requires retraining.
*Note: Requires environment modification to support torque control mode*

```bash
# TODO: Create a1_torque_control.yaml config
robot-lab train --env A1Quadruped-v0 --algo SAC --seed 42 \
  --total-timesteps 500000 \
  --env-config experiments/0_foundations/configs/a1_torque_control.yaml \
  --output-dir data/experiments/smooth_locomotion/exp3_torque
```

| Metric | Position (kp=100) | Torque | Δ |
|--------|-------------------|--------|---|
| jerk_mean | - | - | - |
| fwd_dist | - | - | - |
| fall_rate | - | - | - |
| train_time | - | - | - |

**Conclusion:** *(TBD)*


### Exp 4: Reward Engineering
Add smoothness penalties: `reward -= λ_action * ||Δaction||² + λ_jerk * sum(jerk²)`
*Note: Requires environment modification to include smoothness penalties in reward*

```bash
robot-lab train --env A1Quadruped-v0 --algo SAC --seed 42 \
  --env-config experiments/0_foundations/configs/a1_kp100_baseline.yaml \
  --output-dir data/experiments/smooth_locomotion/exp4_reward
```

| λ_action | λ_jerk | jerk_mean | fwd_dist |
|----------|--------|-----------|----------|
| 0.0 | 0.0 | - | - |
| 0.01 | 0.0 | - | - |
| 0.0 | 0.001 | - | - |
| 0.01 | 0.001 | - | - |

**Conclusion:** *(TBD)*

---

## Analysis

**Plots:** Jerk distributions, learning curves, action trajectories, video comparisons  
**Tools:** FFT (frequency), phase portraits (position vs velocity), gait patterns

## Decision Criteria

1. **Smoothness:** >50% jerk reduction
2. **Performance:** Maintain >80% baseline distance
3. **Learnability:** Training time <2x baseline
4. **Preference:** Simplest effective solution

## Timeline

- **Week 1:** Baseline + Exp 1
- **Week 2:** Exp 2 + Exp 4
- **Week 3:** Exp 3 (if needed)
- **Week 4:** Analysis

## Next Steps

- [ ] Run baseline, measure smoothness
- [ ] Create kp variants, run Exp 1
- [ ] Implement SmoothActionWrapper (Exp 2)
- [ ] Modify reward (Exp 4)
- [ ] Generate comparison plots

---

**Notes:** 
- Control parameters auto-tracked via `ExperimentTracker.log_env_config()`  
- Implement metrics in `robot_lab/utils/smoothness_metrics.py`
- **Main config**: `smooth_locomotion_experiments.yaml` (contains all 12 variants)
- **Individual configs**: `a1_kp{100,50,20,10}.yaml` (for direct CLI use)
- Usage: `robot-lab train --env A1Quadruped-v0 --algo SAC --env-config <path>`
- Visualization: `robot-lab visualize --env A1Quadruped-v0 --algo SAC --env-config <path>`
- *Future*: CLI support for `--experiment-variant exp0_baseline` from master YAML

