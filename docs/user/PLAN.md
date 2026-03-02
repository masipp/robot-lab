
# 🦾 Robotics & Reinforcement Learning Comeback Plan

## Focus: Curriculum Learning + Sim2Real

## Goal: Strong Portfolio + Potential Scientific Contribution

---

# Phase 0 — Core Foundations & Infrastructure (Weeks 1–4)

## Objective
Establish strong foundations in modern RL tooling, reproducibility, and experimental rigor using Stable-Baselines3.

## Current Status ✓
**COMPLETED:**
- ✅ Clean PPO and SAC baselines (using Stable-Baselines3)
- ✅ TensorBoard logging integration
- ✅ Multi-seed experiment support
- ✅ JSON-based config management
- ✅ Experiment tracking infrastructure (ExperimentTracker, ResultsDatabase)
- ✅ Custom environments (A1Quadruped, Gripper, Pusher)
- ✅ Vectorized parallel training (SubprocVecEnv)
- ✅ VecNormalize support with proper save/load
- ✅ CLI interface with Typer
- ✅ Metadata tracking (git commit, system info, hyperparameters)

## Remaining Actions
- [ ] **Algorithm Comparison Document**: Create comprehensive SAC vs PPO comparison
  - Mathematical foundations (policy gradient vs Q-learning)
  - When to use each algorithm
  - Empirical comparison on 3-5 environments
  - Hyperparameter sensitivity analysis
  - Failure modes and debugging strategies
- [ ] CSV exports for offline analysis
- [ ] Multi-seed variance band visualization tools
- [ ] Statistical significance testing utilities

## Deliverable
- SAC vs PPO comparison document (for portfolio and interview prep)
- Reproducible locomotion baseline with variance bands
- Clean experiment results ready for Phase 1

---

# Phase 0.5 — Trajectory Generation & Smooth Control (Weeks 4–6)

## Objective
Transition from direct position control to realistic trajectory-based control for smooth, natural robot movement.

## Research Question
How do we generate smooth, dynamically feasible trajectories for robot control instead of raw position outputs?

## Learning Goals
- Understand trajectory parameterization (splines, Bézier curves, minimum jerk)
- Learn action space design for continuous control
- Explore temporal smoothness vs task performance tradeoffs
- Study action filtering and post-processing techniques

## Potential Approaches (To Be Discussed)
- **Action Smoothing**: Low-pass filter, exponential moving average
- **Trajectory Parameterization**: Output trajectory parameters (duration, waypoints) instead of raw positions
- **Temporal Regularization**: Penalty on action changes (∑‖a_t - a_{t-1}‖²)
- **Latent Action Spaces**: Policy outputs in smooth latent space, decoded to actions
- **Model Predictive Control**: Plan smooth trajectories with learned dynamics

## Experiments
1. **Baseline**: Train policy with raw action outputs → measure jerk and smoothness
2. **Action Filtering**: Post-process actions with smoothing filter
3. **Regularized Training**: Add action change penalty to reward
4. **Trajectory Primitives**: Policy selects from pre-defined smooth motion primitives

## KPIs
- Action smoothness: ∑‖a_t - a_{t-1}‖² (lower is better)
- Jerk: Rate of acceleration change
- Task success rate (ensure smoothness doesn't hurt performance)
- Energy efficiency
- Qualitative assessment (does motion look natural?)

## Deliverable
- Trajectory generation framework integrated into codebase
- Smooth motion comparison document
- Video comparisons (raw vs smoothed policies)

---

# Phase 1 — Foundational Curriculum Experiments (Weeks 7–12)

## Experiment 1 — Manual Curriculum vs No Curriculum

### Research Question
Does structured terrain progression improve sample efficiency and stability?

### Design
Compare:
- Direct training on hardest level.
- Progressive curriculum (Flat → Friction → Obstacles → Climbing).

### KPIs
- Timesteps to 80% success.
- Final performance.
- Learning stability (variance across seeds).
- State visitation entropy.

---

## Experiment 2 — Adaptive Curriculum Advancement

### Research Question
Is performance-based advancement superior to fixed scheduling?

### Design
Compare:
- Fixed episode transitions.
- Reward-threshold advancement.
- Regression-based progression (move back if performance drops).

### KPIs
- Convergence speed.
- Catastrophic forgetting events.
- Reward slope consistency.

---

# Phase 2 — Representation & Generalization (Weeks 13–20)

## Experiment 3 — Goal-Conditioned Curriculum

### Research Question
Does explicit goal representation improve compositional generalization?

### Design
Compare:
- With goal vector in observation.
- Without explicit goal representation.

Test on unseen waypoint configurations.

### KPIs
- Zero-shot success rate.
- Path efficiency.
- Action smoothness (∑‖action‖²).
- Generalization gap.

---

## Experiment 4 — Curriculum + Domain Randomization

### Research Question
Is layered curriculum + domain randomization better than either alone?

### Design
Compare:
- Curriculum only.
- Domain randomization only.
- Curriculum + randomization.

### KPIs
- Robustness score across perturbations.
- Success under friction/slope variation.
- Energy efficiency.
- Stability across seeds.

---

# Phase 3 — Sim2Real-Oriented Research (Weeks 21–32)

## Experiment 5 — Morphology Transfer

### Research Question
Does curriculum pretraining accelerate adaptation to altered robot morphology?

### Variants
- +15% leg length.
- +20% torso mass.
- 70% actuator strength.

### Compare
- Scratch training.
- Zero-shot transfer.
- Fine-tuning from curriculum pretraining.
- Optional: Domain-randomized pretraining.

### KPIs
- Zero-shot success.
- Fine-tuning speed.
- Relative improvement over scratch.
- Robustness under perturbations.

---

## Experiment 6 — Dynamics Gap Sensitivity

### Research Question
Does curriculum reduce sensitivity to dynamics mismatch?

### Simulated Gaps
- Actuator latency.
- Contact noise.
- Motor strength perturbations.

### KPIs
- Success vs perturbation magnitude curve.
- Recovery after pushes.
- Energy efficiency degradation.

---

# Phase 4 — Toward Scientific Contribution (Weeks 33–44)

## Potential Knowledge Gap
Limited understanding of how curriculum affects:
- State visitation distribution.
- Exploration shaping.
- Transfer robustness.

---

## Contribution Direction A — Curriculum as Exploration Shaping

### Hypothesis
Curriculum modifies exploration distribution similar to entropy regularization.

### Measure
- State visitation density.
- Contact frequency maps.
- Action entropy evolution.

---

## Contribution Direction B — Curriculum vs Domain Randomization Tradeoff

### Hypothesis
There exists an optimal balance between structured curriculum and randomization.

### Study
Vary:
- Curriculum depth.
- Randomization magnitude.

Map:
- Robustness.
- Asymptotic performance.

---

## Contribution Direction C — Transfer-Optimal Curriculum

### Hypothesis
Curricula optimized for base-task learning are not optimal for transfer.

### Approach
Design:
- Fast-learning curriculum.
- Diversity-maximizing curriculum.

Compare transfer efficiency.

---

# Stretch Goals (High Impact)

## Stretch 1 — Automatic Teacher Curriculum
- Implement performance-driven environment parameter sampling.
- Compare against manual design.

## Stretch 2 — Curriculum-Aware Representation Learning
- Auxiliary loss predicting terrain difficulty.
- Learn latent terrain embeddings.

## Stretch 3 — Minimal Real Robot Validation
- Evaluate flat-trained vs curriculum-trained policy on hardware.
- Even limited validation increases credibility significantly.

---

# Expected Outcome After 6–9 Months

You will have:

- Reproducible RL infrastructure.
- Structured curriculum framework.
- Robustness benchmarking tools.
- Transfer experiments.
- Potential workshop or conference submission.
- Strong portfolio for robotics RL roles.

---

# Execution Principles

- Always use ≥5 seeds.
- Log everything.
- Save rollout videos.
- Write markdown summaries for every experiment.
- Treat this like a focused research project.

---

End of Plan.
