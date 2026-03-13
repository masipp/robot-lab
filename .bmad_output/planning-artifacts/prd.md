---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-02b-vision', 'step-02c-executive-summary', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type', 'step-08-scoping', 'step-09-extensibility-framework', 'step-10-nonfunctional', 'step-11-polish', 'step-e-01-discovery', 'step-e-02-review', 'step-e-03-edit']
lastEdited: '2026-03-12'
editHistory:
  - date: '2026-03-12'
    changes: 'Added ## Functional Requirements section (FR-001–FR-009 + FR-S01–FR-S02); split stretch goal capabilities in Journey Requirements Summary; added traceability anchor to Extensibility Framework Architecture section'
inputDocuments:
  - 'docs/user/PLAN.md'
  - '.bmad_output/project-context.md'
workflowType: 'prd'
briefCount: 0
researchCount: 0
projectDocsCount: 2
classification:
  projectType: 'personal_learning_research_environment'
  domain: 'scientific_ml_robotics'
  complexity: 'medium'
  projectContext: 'brownfield'
  primaryPurpose: 'Career development through hands-on RL/robotics research — author learns by doing'
vision:
  statement: 'A personal RL research lab that grows with the author — from clean baselines to curriculum learning to sim2real, with every experiment building career-ready expertise in robotics RL'
  differentiator: 'Intentional friction: infrastructure handles everything except the learning-critical logic, which the author must implement themselves'
  coreInsight: 'The value is not the trained models — it is the understanding earned by implementing, debugging, and iterating on core RL concepts'
  targetUser: 'Marco (sole author) — intermediate programmer reskilling toward top-company robotics RL roles'
---

# Product Requirements Document - robot-lab

**Author:** Marco
**Date:** March 5, 2026

---

## Executive Summary

**robot-lab** is a personal reinforcement learning research environment built by and for its sole author (Marco) as a structured 40-week reskilling program toward senior robotics RL roles at top companies. The project runs on consumer hardware (GTX 1080) using Stable-Baselines3 + Gymnasium, and is organized as a progressive research curriculum: clean baselines → trajectory control → curriculum learning → sim2real transfer → scientific contribution.

The project is deliberately **brownfield**: Phase 0 infrastructure (PPO/SAC baselines, vectorized training, experiment tracking, CLI, config system) is complete. This PRD governs Phases 0.5–4 — the research and feature work that remains.

### What Makes This Special

The defining design decision is **intentional friction**: the infrastructure layer (CLI, config system, experiment tracker, logging, visualization, test scaffolding) is fully automated and maintained by tooling, while every learning-critical implementation — curriculum advancement logic, trajectory generation, domain randomization, goal conditioning, sim2real transfer — must be authored by Marco himself. An AI agent actively enforces this boundary (YouAreLazy protocol), blocking shortcuts that would undermine the learning objective.

This makes robot-lab uniquely suited to its purpose: it is not a framework for running RL experiments — it is a *curriculum for becoming an RL engineer*, where the codebase is the artifact that proves competency.

The second foundational design decision is the **plugin/registry architecture** governing all observability concerns: evaluation metrics, visualization outputs, and experiment metadata are each independently extensible by dropping in an interface implementation — no modifications to the core training loop, tracker, or visualization pipeline required. This is not an optional convenience pattern; it is the contract that keeps the infrastructure stable across all four research phases as new analytical needs emerge.

### Project Classification

| Attribute | Value |
|---|---|
| **Project Type** | Personal learning & research environment (solo author) |
| **Domain** | Scientific ML / Robotics Reinforcement Learning |
| **Complexity** | Medium (no regulatory overhead; requires statistical validity, reproducibility, domain expertise) |
| **Project Context** | Brownfield (Phase 0 complete; Phases 0.5–4 in scope) |
| **Primary Stakeholder** | Marco (author, sole user, subject of skill development) |
| **Target Outcome** | Portfolio of reproducible research + demonstrated RL/robotics expertise for career transition |

---

## Success Criteria

### User Success

Marco succeeds when he can:

1. **Implement core RL concepts independently** — the full breadth of learning targets below authored from scratch (not copied from AI), debuggable, and explainable in a technical interview:
   - **Trajectory generation & smooth control**: action filtering, parameterization, temporal regularization
   - **Curriculum learning**: advancement conditions, adaptive scheduling, regression strategies
   - **Goal-conditioned RL**: goal representation, HER variants, goal sampling
   - **Domain randomization**: distribution design, curriculum-DR interaction, ADR
   - **Custom environments**: reward functions, observation spaces, success criteria (GripperEnv + A1Quadruped)
   - **Morphology transfer & sim2real**: fine-tuning strategies, dynamics gap compensation
   - **Scientific analysis**: state visitation entropy, action entropy evolution, transfer efficiency metrics, statistical significance
2. **Run any experiment in one command** — `robot-lab train --env X --algo Y --seed Z` launches a complete, tracked, reproducible run with zero manual setup
3. **Explain experiment results clearly** — every completed experiment has a markdown summary with methodology, results, and conclusions suitable for a portfolio or paper appendix
4. **Compare variants systematically** — multi-seed runs with variance bands prove statistical validity of any observed difference

### Business/Career Success

| Milestone | Target | Timeframe |
|---|---|---|
| Phase 0.5 complete | Smooth locomotion baseline + trajectory comparison document | Week 6 |
| Phase 1 complete | Curriculum vs no-curriculum paper-quality result | Week 12 |
| Phase 2 complete | Goal-conditioned + DR integration results | Week 20 |
| Phase 3 complete | Morphology transfer + dynamics gap experiments | Week 32 |
| Phase 4 complete | ≥1 workshop/conference-ready contribution | Week 44 |
| Portfolio ready | 5+ reproducible experiments with video renders + markdown summaries | Ongoing |

### Technical Success

- **Reproducibility**: Any published result re-producible from seed + config alone (no manual steps)
- **Infrastructure stability**: All tests pass (`make test`); CLI commands work end-to-end; no broken imports
- **Hardware efficiency**: Full training runs complete on GTX 1080 within reasonable wall-clock time (≤48h per experiment)
- **Result integrity**: ≥5 seeds per experiment; variance bands on all learning curves; no p-hacking
- **Codebase health**: Ruff clean; type hints on all public functions; Google-style docstrings

### Measurable Outcomes (Per-Phase KPIs)

| Phase | Primary KPI | Target |
|---|---|---|
| 0.5 — Trajectory | Action smoothness ∑‖aₜ−aₜ₋₁‖² | Measurable reduction vs raw baseline |
| 1 — Curriculum | Timesteps to 80% success | Curriculum ≤ 70% of no-curriculum |
| 1 — Adaptive | Convergence speed vs fixed schedule | Performance-based advancement ≥ fixed |
| 2 — Goal-conditioned | Zero-shot success rate on novel configs | Statistically significant improvement |
| 2 — DR | Robustness score vs perturbations | Curriculum+DR ≥ either alone |
| 3 — Morphology | Fine-tuning speed from curriculum pretraining | ≤ 50% of scratch training time |
| 3 — Dynamics | Success vs perturbation magnitude curve | Curriculum reduces degradation slope |
| 4 — Exploration | State visitation entropy over curriculum stages | Measurable distribution shift |

---

## Product Scope

### MVP — Minimum Viable Research Platform (Phases 0.5–1)

*Must be true for the project to prove its core thesis (curriculum learning improves sample efficiency)*

- Trajectory generation framework integrated (Phase 0.5): smooth action experiment infrastructure, video comparison tooling
- Curriculum callback infrastructure: base class, logging hooks, stage tracking in ExperimentTracker
- Manual curriculum experiment (Exp 1): direct vs progressive training, ≥5 seeds, variance bands, markdown summary
- Adaptive advancement experiment (Exp 2): reward-threshold vs fixed-schedule comparison
- CSV export + multi-seed variance visualization utilities

### Growth Features — Research Depth (Phases 2–3)

*What makes this competitive as a portfolio and credible as research*

- Goal-conditioned experiment infrastructure: observation augmentation, zero-shot evaluation harness
- Domain randomization integration: parameter sampling hooks, curriculum-DR combination runner
- Morphology transfer experiment: variant environment configs + fine-tuning runner
- Dynamics gap sensitivity: perturbation sweep tooling, success-vs-magnitude plotting
- TensorBoard enhanced dashboards: state visitation heatmaps, action entropy over time

### Vision — Scientific Contribution (Phase 4 + Stretch Goals)

*Dream version — potential workshop/conference paper*

- State visitation density analysis pipeline
- Transfer-optimal curriculum comparison framework
- Automatic teacher curriculum (PLR/ALP implementation — by Marco)
- Curriculum-aware representation learning auxiliary loss
- **Gazebo sim2real validation stage**: deploy curriculum-trained policy into Gazebo with calibrated reality measures — actuator dynamics (motor torque curves, inertia), observation latency (simulated sensor delay), contact noise, and friction perturbations; compare performance degradation vs pure MuJoCo baseline to quantify the sim2real gap attributable to dynamics mismatch
- Minimal real-robot validation (physical hardware, video evidence of transfer — even single-environment proof is publication-credible)
- Published workshop or conference paper
- **Stretch: Advanced experiment extensions** *(if Phases 1–4 complete ahead of schedule)*:
  - Cross-environment curriculum transfer: train curriculum on GripperEnv, evaluate zero-shot on novel manipulation tasks
  - Multi-task curriculum: simultaneous locomotion + manipulation training with shared representation
  - Curriculum meta-learning: few-shot adaptation of curriculum parameters across morphology variants
  - Online curriculum with real-time difficulty estimation (no fixed stage boundaries)
  - Curriculum distillation: compress multi-stage curriculum-trained policy into a single-stage deployment model

---

## User Journeys

### Journey 1 — The Experimenter (Primary Happy Path)

*Marco sits down on a Saturday to run his next curriculum experiment.*

He opens the terminal and fires off one command. The training run starts immediately — parallel envs spin up, TensorBoard logs begin streaming, and the experiment tracker writes metadata to a JSON file. 45 minutes later, the run finishes. He opens the results folder: a markdown summary template is waiting, pre-filled with config, system info, and seed. He fills in his observations, commits to git.

Next he runs the no-curriculum baseline with the same command. Then `make plot` generates side-by-side learning curves with variance bands across seeds. He can already see the curriculum variant converges faster. The research question has a provisional answer.

**Capabilities revealed:** one-command training, automatic experiment tracking, multi-seed runner, variance band plotting, pre-filled markdown summary template.

---

### Journey 2 — The Implementer (Core Learning Path)

*Marco wants to implement reward-threshold curriculum advancement (Experiment 2).*

He opens `robot_lab/utils/callbacks.py` and finds a `CurriculumCallback` base class already scaffolded — `on_step()` hook, stage tracking logged to ExperimentTracker, placeholder `_should_advance()` method left intentionally empty. The docstring says: *"Implement `_should_advance()` to define your advancement condition."*

He reads the SB3 callback docs, sketches the logic on paper, and writes the condition function himself. He runs the unit test: `test_curriculum_advancement` — it passes. He runs the experiment. Mid-training he notices the agent never advances past stage 1. He fires up TensorBoard, inspects the curriculum stage metric, and spots the threshold is too high. He adjusts, re-runs, and this time the stages advance cleanly.

The debugging loop is the education. The infrastructure got out of the way.

**Capabilities revealed:** scaffolded base class with empty core hooks, per-run curriculum stage metrics in TensorBoard, unit test harness for callbacks, actionable error messages.

---

### Journey 3 — The Analyst (Post-Experiment Review)

*A week after Phase 1 experiments, Marco wants to compare all results and write a portfolio summary.*

He runs `robot-lab results --phase 1` and gets a table: all runs, seeds, final rewards, timesteps-to-threshold, variance. He exports to CSV, opens in a notebook, and generates the variance band plot. He notices Experiment 2 (adaptive advancement) has high variance on seed 3 — he digs into the TensorBoard log and finds a catastrophic forgetting event.

He writes this up in the experiment markdown doc under "Anomalies." The finding becomes a research insight for Phase 2: regression-based advancement might be necessary. The result database was queryable, the unexpected result was visible, the insight feeds forward.

**Capabilities revealed:** results query CLI, CSV export, ResultsDatabase schema, per-run TensorBoard logs, experiment markdown template with anomaly section.

---

### Journey 4 — The Sim2Real Validator (Phase 4 Vision)

*Marco has a curriculum-trained quadruped policy and wants to evaluate how it holds up against reality-like conditions.*

He launches the Gazebo validation harness — a configured simulation environment with injected actuator dynamics, observation latency, and contact noise tuned to match known hardware specs. He runs the same policy that scored well in MuJoCo. Gazebo logs success rate, energy consumption, and recovery-after-push metrics.

The performance drops 22% vs MuJoCo baseline — but only 8% vs the domain-randomized variant. The gap is quantified. He exports a degradation curve (success vs perturbation magnitude), adds it to the experiment summary, and has concrete evidence that curriculum + DR reduces the sim2real gap. This figure goes into the paper.

**Capabilities revealed:** Gazebo integration harness (infrastructure), perturbation sweep runner, sim2real gap metrics (success degradation curve), comparison table vs MuJoCo baseline.

---

### Journey Requirements Summary

| Capability | Revealed By |
|---|---|
| One-command experiment launch | Journey 1 |
| Automatic experiment tracking + metadata | Journey 1 |
| Multi-seed runner + variance band plotting | Journeys 1, 3 |
| Pre-filled markdown summary template | Journeys 1, 3 |
| Scaffolded base classes with empty learning-critical hooks | Journey 2 |
| Per-run curriculum stage metrics in TensorBoard | Journey 2 |
| Unit test harness for user-implemented callbacks | Journey 2 |
| Results query CLI + CSV export | Journey 3 |
| ResultsDatabase queryable schema | Journey 3 |

*(Capabilities above correspond to FRs FR-001–FR-009 — confirmed Phase 0.5–3 deliverables.)*

**Phase 4 Stretch Goal Capabilities:**

| Capability | Revealed By |
|---|---|
| Gazebo sim2real validation harness (infrastructure only) | Journey 4 |
| Perturbation sweep runner + degradation curve plotting | Journey 4 |

*(Stretch capabilities correspond to FRs FR-S01–FR-S02.)*

---

## Functional Requirements

> Derived from and traceable to User Journeys above. Every FR maps to at least one journey. Capabilities marked **Stretch** are Phase 4 goals; all others are confirmed Phase 0.5–3 deliverables.

### Core Capabilities (Phases 0.5–3)

| ID | Marco can... | Test Criteria | Journey |
|---|---|---|---|
| FR-001 | Launch a complete, tracked training run with a single CLI command | `robot-lab train --env X --algo Y --seed Z` executes end-to-end with no manual setup steps | 1 |
| FR-002 | Retrieve full experiment metadata (config, system info, git commit, timestamps) automatically after each run | Experiment directory contains `metadata.json`, `hyperparameters.json`, `system_info.json` with populated fields at run completion | 1 |
| FR-003 | Run identical experiments across ≥5 seeds and generate variance band plots | `--seed` parameter accepted; variance band plot generated from multi-seed results without manual scripting | 1, 3 |
| FR-004 | Access a pre-filled markdown experiment summary template after each run | Template populated with config, system info, and seed present in experiment directory at completion | 1, 3 |
| FR-005 | Extend experiment tracking with custom RL logic via base class hooks without modifying core training code | `CurriculumCallback`, `ActionFilterWrapper`, `GoalConditionedWrapper`, `DomainRandomizationWrapper` base classes present with intentionally empty core hook methods | 2 |
| FR-006 | Inspect per-run curriculum stage progression in TensorBoard during and after training | Curriculum stage metric logged at each stage transition; visible in TensorBoard as a scalar series | 2 |
| FR-007 | Validate user-implemented callback logic with unit tests before running full experiments | Test scaffolding in `tests/` accepts mock environment context; passes without live training run | 2 |
| FR-008 | Query all experiment results by phase and export to CSV | `robot-lab results --phase N` returns table with seeds, final rewards, KPI values; CSV export functional | 3 |
| FR-009 | Query ResultsDatabase for any completed run by run ID, seed, phase, or date range | `ResultsDatabase` query interface accessible programmatically and via CLI | 3 |

### Stretch Goal Capabilities (Phase 4)

| ID | Marco can... | Test Criteria | Journey |
|---|---|---|---|
| FR-S01 | Evaluate a trained policy in Gazebo with injected actuator dynamics, observation latency, and contact noise | `robot-lab gazebo-eval` produces success rate, energy, and recovery metrics; conditional `try: import rclpy` import ensures base package works without ROS 2 | 4 |
| FR-S02 | Run a perturbation sweep and generate a success-vs-perturbation-magnitude degradation curve | Sweep runner accepts parameter range config; outputs matplotlib degradation curve and CSV export | 4 |

---

## Domain-Specific Requirements

### Reproducibility Standards

Scientific credibility depends on strict reproducibility:
- Every result must be reproducible from `(seed, config, git commit)` alone — no manual environment setup steps
- Floating-point non-determinism must be explicitly acknowledged (GPU ops may vary across CUDA versions); document known sources of variance
- All configs immutably copied into the experiment directory at run start — modifying configs mid-experiment is a protocol violation
- Minimum ≥5 seeds per reported result; single-seed results are not reportable

### Computational Constraints (GTX 1080 / sm_61)

- PyTorch hard-capped at `<2.10`; sm_61 (GTX 1080) support dropped in 2.10+ — see NFRs for version pinning requirements
- Experiment designs must account for VRAM limits (~8 GB) — batch sizes, policy network sizes, and number of parallel envs constrained accordingly
- `SubprocVecEnv` incompatible with MuJoCo rendering on Windows (GL context sharing) — `DummyVecEnv` required for all visualization paths
- Training wall-clock target: ≤48 hours per experiment on current hardware; experiments exceeding this are flagged as hardware-constrained and deferred, not discarded

### Simulation Fidelity & Sim2Real Gap

- MuJoCo is the primary physics backend (via `gymnasium[mujoco]`); contact dynamics and actuator models are simplified vs real hardware
- Gazebo validation stage (Phase 4) must inject: actuator latency, motor torque curves, contact noise, and friction perturbations calibrated to target hardware specs
- Results comparing MuJoCo vs Gazebo performance must document all injected perturbation parameters for reproducibility
- Domain randomization parameters must be grounded in physically plausible ranges — unbounded randomization is not scientifically valid

### Statistical Validity

- Variance bands on all learning curves are required before any comparative claim is made
- "Curriculum outperforms baseline" requires: overlapping seeds, consistent hyperparameters across variants, and statistical separation visible in variance bands
- Anomalies (unexpected variance spikes, catastrophic forgetting events) must be documented and investigated — not discarded

### Risk Mitigations

| Risk | Mitigation |
|---|---|
| PyTorch version incompatibility (sm_61 dropped) | Hard lock `<2.10` in `pyproject.toml`; upgrade only with explicit GPU compat check |
| Experiment results not reproducible | Immutable config copy + git commit hash logged at run start |
| AI-generated core logic bypasses learning objective | YouAreLazy enforcement protocol; scaffolded hooks with intentionally empty implementations |
| Gazebo-MuJoCo gap miscommunicated in results | All Gazebo runs document exact perturbation parameters in experiment metadata |
| Training run corruption (power loss, OOM) | Checkpoint callbacks at configurable intervals; `ExperimentTracker` marks incomplete runs as `INTERRUPTED` |

---

## Research Platform Specific Requirements

### Project-Type Overview

robot-lab is a Python package (`robot_lab`) that functions simultaneously as:
- A **CLI tool** (`robot-lab train`, `visualize`, `tensorboard`, `info`, `results`) for running experiments
- A **library** (`from robot_lab.training import train`, `from robot_lab.experiments import ExperimentTracker`) for programmatic use
- A **research scaffold** — structured directories, templates, and base classes that Marco fills in with core RL logic

All three modes must remain consistent; CLI commands are thin wrappers around importable library functions.

### Language & Package Management

- **Language**: Python ≥3.12 exclusively
- **Package manager**: `uv` — `uv sync`, `uv add`, `uv run` are the only supported workflows; no bare `pip` or `conda`
- **Build backend**: `hatchling` — must not change
- **Installation**: editable install via `uv sync`; CLI exposed as `robot-lab` script via `[project.scripts]`

### API Surface (Library Interface)

Public API must remain stable within each phase to avoid breaking experiment scripts mid-research:

| Module | Public Functions |
|---|---|
| `robot_lab.training` | `train()` |
| `robot_lab.visualization` | `visualize()` |
| `robot_lab.config` | `load_hyperparameters()` |
| `robot_lab.experiments` | `ExperimentTracker`, `ResultsDatabase`, `ExperimentRunner`, `get_template()` |
| `robot_lab.envs` | `make_env()`, `register_custom_envs()` |
| `robot_lab.utils.paths` | `get_models_dir()`, `get_logs_dir()`, `get_experiments_dir()` |

New public functions require type hints + Google-style docstrings before merge.

### Configuration & Code Examples

- Hyperparameter configs: JSON in `robot_lab/configs/`, accessed via `importlib.resources.files()` — never `__file__`
- Experiment runner configs: YAML in `experiments/` root directory
- Every new environment requires a `{env}_{algo}.json` config and a usage example in `experiments/`
- README must have a working example for every new CLI command before merge

### Extension Points — Infrastructure Scaffolds, Marco Implements

The infrastructure provides hooks, harnesses, and base classes; Marco authors the core logic. See Extensibility Framework Architecture for the plugin registration pattern governing how these hooks integrate with the observability stack.

#### Curriculum Learning

| Component | Infrastructure Provided | Marco Implements |
|---|---|---|
| `CurriculumCallback` base class | `on_step()` hook, stage tracking in ExperimentTracker, TensorBoard logging | `_should_advance()` — advancement condition |
| Regression strategy | Stage state tracking, rollback hook | When and how to move backwards |
| Adaptive scheduler | Episode window buffering, metric aggregation | Threshold adaptation logic |

#### Trajectory Generation & Smooth Control

| Component | Infrastructure Provided | Marco Implements |
|---|---|---|
| `ActionFilterWrapper` base class | Gym wrapper boilerplate, obs/act passthrough | Filtering logic (low-pass, EMA, splines, etc.) |
| Smoothness metrics in ExperimentTracker | `∑‖aₜ−aₜ₋₁‖²` logging hook | Threshold criteria and training-time penalty |
| Video render pipeline | Episode recording wrapper, MP4 export | Nothing — fully auto-implemented |

#### Goal-Conditioned RL

| Component | Infrastructure Provided | Marco Implements |
|---|---|---|
| `GoalConditionedWrapper` base class | Observation augmentation hook, goal logging | Goal sampling strategy, goal relabeling (HER variant) |
| Zero-shot evaluation harness | Novel goal config loader, success rate tracker | Goal representation design |

#### Domain Randomization

| Component | Infrastructure Provided | Marco Implements |
|---|---|---|
| `DomainRandomizationWrapper` base class | Parameter logging, curriculum-DR combo runner | `_sample_params()` — distribution design |
| Robustness sweep runner | Parallel eval across perturbation magnitudes | Perturbation distribution ranges |

#### Custom Environments

| Environment | Infrastructure Provided | Marco Implements |
|---|---|---|
| `GripperEnv-v0` (manipulation) | Gym API skeleton, env registration, config JSON | Reward function, observation space, success criterion |
| `A1Quadruped-v0` (locomotion) | Physics config, env registration | Reward shaping, terrain curriculum stages |
| Morphology variant envs | Env factory with param injection scaffolding | Physics parameter ranges for morphology variants |

**Note on GripperEnv**: Included alongside locomotion to validate that curriculum and DR insights generalize across task types, to demonstrate portfolio breadth beyond single-task RL, and to test goal-conditioned RL in a context where goal representation is more explicit.

#### Sim2Real & Transfer

| Component | Infrastructure Provided | Marco Implements |
|---|---|---|
| Gazebo validation harness | ROS/Gazebo launch configs, perturbation injector, metrics logger | Perturbation calibration to target hardware specs |
| Fine-tuning runner | Checkpoint loader, eval harness for morphology variants | Fine-tuning strategy, learning rate schedule |

#### Analysis & Scientific Methods

| Component | Infrastructure Provided | Marco Implements |
|---|---|---|
| State visitation logger | Env step interceptor, heatmap plotter scaffold | Entropy calculation, interpretation |
| Action entropy tracker | Per-step action logging, evolution plotter | Statistical significance test |
| Transfer efficiency reporter | Cross-experiment comparison table scaffolding | Transfer metric definition |

### Implementation Considerations

- **No side effects on import**: `import robot_lab` must not trigger training, file creation, or network calls
- **Lazy imports**: `torch` and `stable_baselines3` imported only inside functions that use them
- **Windows compatibility**: `SubprocVecEnv` forbidden with MuJoCo rendering on Windows — `DummyVecEnv` for all visualization paths
- **Absolute imports only**: `from robot_lab.x import y` everywhere except `__init__.py` re-exports

---

### Extensibility Framework Architecture (Plugin/Registry Pattern)

> *This pattern directly enables all four user journeys — Experimenter, Implementer, Analyst, and Sim2Real Validator — by providing a stable, extensible observability layer that scales across all research phases without requiring modifications to core training, tracking, or visualization modules. See Journey Requirements Summary for the capabilities this enables.*

This is a **foundational architectural principle**, not an optional add-on. The entire observability stack — metrics evaluation, visualization, and experiment metadata — is designed around a plugin/registry pattern. New capabilities are added by implementing a defined interface and registering the plugin; the infrastructure invokes all registered plugins at the appropriate lifecycle points automatically. The core modules (`training.py`, `tracker.py`, `visualization.py`) are not modified when new analytical concerns are introduced.

This principle applies consistently across three concerns:

#### Evaluation Metrics (MetricsPlugin)

New evaluation metrics are implemented as `MetricsPlugin` subclasses and registered with the `MetricsRegistry`. The `ExperimentTracker` discovers and executes all registered plugins at the correct lifecycle hooks (on-step, on-episode-end, on-eval). Adding a smoothness metric, a curriculum efficiency metric, or a transfer efficiency score requires no changes to `tracker.py` or the training loop.

| What infrastructure provides | What Marco implements |
|---|---|
| `MetricsPlugin` base class with lifecycle hooks (`on_step`, `on_episode_end`, `on_eval`) | Metric calculation logic inside the hook |
| `MetricsRegistry` — auto-invokes all registered plugins | When and how to aggregate (window, cumulative, per-stage) |
| Automatic JSON + TensorBoard export for every registered metric | Threshold criteria and scientific interpretation |

#### Visualization (VisualizationPlugin)

New plot types are implemented as `VisualizationPlugin` subclasses and registered with the `VisualizationRegistry`. The visualization pipeline invokes all registered plugins with the current result set. Adding a state visitation heatmap, an action entropy evolution chart, or a curriculum stage timeline requires no changes to `visualization.py`.

| What infrastructure provides | What Marco implements |
|---|---|
| `VisualizationPlugin` base class with `render(results)` hook | Plot construction logic (matplotlib, seaborn, etc.) |
| `VisualizationRegistry` — batches all registered plugins per run | Output format choices and figure layout |
| Consistent output path injection and MP4/PNG export wiring | Scientific interpretation and annotation |

#### Experiment Metadata (MetadataPlugin)

New metadata categories are implemented as `MetadataPlugin` subclasses and registered with the `MetadataRegistry`. The `ExperimentTracker` collects all registered providers at run-start, mid-run checkpoints, and run-end. Adding curriculum stage snapshots, domain randomization parameter logs, or hardware telemetry requires no changes to `tracker.py` or `ExperimentTracker`.

| What infrastructure provides | What Marco implements |
|---|---|
| `MetadataPlugin` base class with `collect(context)` hook | What data to capture and how to structure it |
| `MetadataRegistry` — merges all plugin outputs into `metadata.json` | Domain-specific context (e.g., curriculum stage number, DR parameter values) |
| Schema validation (Pydantic) before write to disk | Ensuring captured values are scientifically meaningful |

#### Plugin Registration Convention

All three plugin types follow the same registration pattern so new capabilities are discoverable and composable without coupling:

```python
from robot_lab.experiments.plugins import (
    register_metric_plugin,
    register_visualization_plugin,
    register_metadata_plugin,
)

register_metric_plugin(ActionSmoothnessMetric())          # custom metric
register_visualization_plugin(CurriculumProgressPlot())   # custom plot
register_metadata_plugin(CurriculumStageMetadata())       # custom metadata
```

Plugins registered in an experiment script are active for that run only. Plugins registered in `robot_lab/experiments/plugins/__init__.py` are active globally.

#### Architectural Guarantees

- **Zero core-code changes**: every new observability capability is an additive registration, never a patch to internals
- **Phase isolation**: plugins from Phase 1 (curriculum metrics) and Phase 3 (sim2real gap metrics) coexist without conflict — they share the registry, not the implementation
- **YouAreLazy-compatible**: the *plugin infrastructure* (interfaces, registries, lifecycle hooks, export wiring) is scaffolded by the agent; the *plugin logic* (what to measure, how to visualize, what metadata is meaningful) is Marco's responsibility
- **Testable in isolation**: each plugin implements a narrow interface and can be unit-tested with mock context without requiring a live training run

This plugin/registry pattern is the third architectural pillar alongside YouAreLazy enforcement and one-command reproducibility.

---

## Innovation & Novel Patterns

### Detected Innovation Areas

**1. Intentional Friction as Pedagogy (Primary Innovation)**
The YouAreLazy enforcement protocol is a novel application of constraint-based learning to software tooling. Rather than optimizing for user productivity, the system deliberately withholds automation in learning-critical areas. The AI agent actively identifies and refuses to implement core RL logic, redirecting the user to implement it themselves. This inverts the standard developer-tool value proposition: productive friction is the feature.

**2. AI-Enforced Learning Boundaries**
Embedding a learning boundary protocol directly into the AI agent's instructions (copilot-instructions.md) creates a persistent enforcement layer that persists across sessions. The agent maintains a taxonomy of "YouAreLazy" tasks vs infrastructure tasks and enforces this distinction without user intervention. This is a novel pattern for AI-assisted skill development tools.

**3. Curriculum Research Platform on Consumer Hardware**
Running a complete Phase 0–4 curriculum learning and sim2real research program on a GTX 1080 (8 GB VRAM, sm_61) is non-trivially constrained. The project demonstrates that publication-credible RL research is achievable outside academic GPU clusters — relevant to practitioners without institutional compute access.

**4. MuJoCo → Gazebo Sim2Real Validation Pipeline**
The Phase 4 Gazebo harness injects calibrated reality measures (actuator dynamics, latency, contact noise) to quantify the sim2real gap attributable to dynamics mismatch, without requiring real hardware. This creates a reproducible, hardware-free sim2real validation methodology.

### Validation Approach

- **YouAreLazy protocol**: validated when Marco can independently debug, explain, and extend his own implemented RL logic in technical interviews
- **Curriculum research**: validated via statistically significant KPI improvements across ≥5 seeds vs baselines
- **Gazebo sim2real harness**: validated when perturbation sweep curves are reproducible across runs with fixed perturbation parameters
- **Consumer hardware feasibility**: validated when Phase 1–2 experiments complete within the ≤48h wall-clock budget

### Risk Mitigation

| Innovation Risk | Mitigation |
|---|---|
| YouAreLazy boundary ambiguity (gray areas) | Explicit taxonomy in copilot-instructions.md; agent errs toward not implementing |
| Gazebo fidelity insufficient for meaningful gap measurement | Perturbation parameters grounded in published hardware specs; gaps documented quantitatively |
| Consumer hardware bottleneck blocks Phase 3–4 | Experiment designs pre-scoped to VRAM budget; fallback to lighter environments (HalfCheetah vs A1Quadruped) |

---

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** *Platform MVP* — the minimum that makes the research thesis testable. The core claim is “structured curriculum learning improves sample efficiency.” The MVP is complete when that claim can be tested, measured, and reported with statistical validity across ≥5 seeds.

**Resource requirements:** Solo developer (Marco), GTX 1080, ~12 weeks for Phases 0.5–1.

**What the MVP is NOT:**
- It is not a publishable paper (that’s Phase 4)
- It is not a generalizable framework (only needs to work for Marco’s specific experiments)
- It is not complete until there’s a reproducible baseline to compare against

### MVP Feature Set (Phase 1 — Weeks 1–12)

**Core journeys supported:** Experimenter + Implementer (Journeys 1 & 2)

**Must-have capabilities:**

| Capability | Rationale |
|---|---|
| Trajectory smoothness experiment infrastructure (Phase 0.5) | Required baseline before curriculum work — establishes smooth-action foundation |
| `CurriculumCallback` scaffolded base class | Marco can’t implement advancement logic without the hook infrastructure |
| Multi-seed experiment runner | No result is reportable without ≥5 seeds |
| Variance band plotting (`make plot`) | Required to visually compare curriculum vs no-curriculum |
| Experiment markdown template (pre-filled) | Required for portfolio-ready output |
| CSV export from ResultsDatabase | Required for offline analysis in notebooks |
| `GripperEnv-v0` skeleton (Gym boilerplate only) | Needed for Phase 2 transferability validation; skeleton in Phase 1, logic in Phase 2 |

**Deferred from MVP:**
- Results query CLI (`robot-lab results --phase 1`) — direct JSON inspection sufficient for Phase 1
- Enhanced TensorBoard dashboards — standard dashboards sufficient for MVP

### Feature Dependencies (Build Order)

```
Phase 0 (done)
  └── Phase 0.5: Trajectory framework + smoothness baseline
        └── Phase 1a: CurriculumCallback scaffold + manual curriculum experiment
              └── Phase 1b: Adaptive advancement + regression strategy
                    └── Phase 2a: GoalConditionedWrapper + GripperEnv logic
                          └── Phase 2b: DomainRandomizationWrapper + curriculum-DR combination
                                └── Phase 3a: Morphology variant envs + fine-tuning runner
                                      └── Phase 3b: Dynamics gap + Gazebo harness
                                            └── Phase 4: State visitation + transfer analysis + paper
                                                  └── Stretch: Advanced experiments (if ahead of schedule)
```

**Critical dependency**: Phase 0.5 trajectory work must complete before Phase 1 — the smoothness baseline is the reference point for all subsequent action quality comparisons.

### Risk Mitigation Strategy

**Technical Risks:**

| Risk | Severity | Mitigation |
|---|---|---|
| VRAM exhaustion on complex MuJoCo envs (A1Quadruped) | High | Fall back to `HalfCheetah-v4` for Phase 1–2; A1 reserved for Phase 3+ |
| Curriculum advancement logic never converges in Phase 1 | Medium | Implement fixed-schedule variant first as sanity check; adaptive after |
| Gazebo-MuJoCo integration complexity blocks Phase 4 | Medium | Treat Gazebo as stretch goal; MuJoCo perturbation injection is the fallback |

**Research Risks:**

| Risk | Severity | Mitigation |
|---|---|---|
| Curriculum shows no improvement vs baseline | Medium | Expected outcome; documents null result — still publishable with good methodology |
| Results don’t transfer from locomotion to GripperEnv | Low | Exactly what we’re testing; null transfer is a valid research finding |
| Phase 4 scope too broad for solo researcher | High | Commit to one contribution direction (A, B, or C) by end of Phase 3; scope the paper accordingly |

**Resource Risks:**

| Risk | Severity | Mitigation |
|---|---|---|
| Weekly time availability inconsistent | Medium | Each phase designed as independent experiment units; partial completion is still publishable |
| Phase slippage compresses Phase 4 | Low | Phase 4 stretch goals explicitly optional; workshop paper possible from Phase 2–3 results alone |

---

## Non-Functional Requirements

### Performance

- **Training throughput**: full-phase experiments (Phase 0.5–Phase 2) are designed to complete within the 48h wall-clock budget on GTX 1080 (8 GB VRAM); experiments exceeding this ceiling are flagged as **hardware-constrained** in the results database and deferred — not discarded — pending a hardware upgrade if development progresses to that point
- **CLI cold-start latency**: `robot-lab --help` and config-loading commands must respond within 3 seconds; achieved via lazy imports (`torch`, `stable_baselines3` not imported at CLI entry point)
- **TensorBoard logging overhead**: step-level logging must not add >5% wall-clock overhead vs unlogged training; default frequency is every 100 steps (not every step)
- **Vectorized environment requirement**: training runs use `SubprocVecEnv` with `num_envs ≥ 4` by default; single-env training is explicitly not the performance baseline

### Reliability

- **Experiment fault tolerance**: checkpoint callbacks save intermediate models at configurable intervals (default: every 50k steps); a run interrupted after the first checkpoint can be diagnosed from saved state — complete loss of a multi-hour run is not acceptable
- **Experiment completeness tracking**: `ExperimentTracker` marks runs as `COMPLETED`, `INTERRUPTED`, or `FAILED`; any non-`COMPLETED` run is flagged in the results database — incomplete runs must never be silently included in comparative analysis
- **VecNormalize integrity**: model checkpoints and VecNormalize statistics are always saved as atomic pairs; a model checkpoint without its matching VecNorm stats file is a corrupt artifact — the save path logic must enforce this invariant
- **Config immutability**: the JSON config active at run-start is copied into the experiment directory before the first training step; any mutation to the original config file after this point has no effect on the running experiment

### Maintainability

- **Lint cleanliness**: `ruff check robot_lab/` and `ruff format robot_lab/` must pass with zero violations before any merge; max line length 100 characters
- **Type hint coverage**: all public functions in all public modules require full type hints (args + return type); private helpers are encouraged but not enforced
- **Test baseline**: core infrastructure modules must remain covered by at least smoke tests; `make test` must pass before any implementation PR merges
- **Docstring standard**: Google-style docstrings with Args/Returns/Raises sections on all public API functions; inline comments required where architectural decisions are non-obvious
- **Import safety**: `import robot_lab` and `from robot_lab.x import y` must never trigger I/O, training, or network calls — validated by the smoke test suite

### Integration Compatibility

- **PyTorch version ceiling**: hard-capped at `<2.10` in `pyproject.toml`; sm_61 (GTX 1080) support is dropped in 2.10+ — any bump must be gated on explicit hardware compatibility verification
- **SB3 API stability**: Stable-Baselines3 and Gymnasium versions pinned in `pyproject.toml`; upgrades require running the full training smoke test before merging
- **Gymnasium API conformance**: all custom environments (`GripperEnv-v0`, `A1Quadruped-v0`) must pass Gymnasium's `check_env()` validation at registration time
- **MuJoCo rendering constraint**: `SubprocVecEnv` is forbidden on all visualization paths on Windows (GL context sharing incompatibility); `DummyVecEnv` is required — this is a hard architectural constraint, not a configuration option
- **Gazebo integration (Phase 4)**: Gazebo harness must be isolated behind a conditional import (`try: import rclpy`) so the absence of ROS 2 never breaks the base package install


