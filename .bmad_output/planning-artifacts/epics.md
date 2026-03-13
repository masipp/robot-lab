---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-design-epics', 'step-03-create-stories', 'step-04-final-validation']
inputDocuments:
  - '.bmad_output/planning-artifacts/prd.md'
  - '.bmad_output/planning-artifacts/architecture.md'
---

# robot-lab - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for robot-lab, decomposing the requirements from the PRD and Architecture document into implementable stories.

## Requirements Inventory

### Functional Requirements

FR-001: Marco can launch a complete, tracked training run with a single CLI command — `robot-lab train --env X --algo Y --seed Z` executes end-to-end with no manual setup steps.

FR-002: Marco can retrieve full experiment metadata (config, system info, git commit, timestamps) automatically after each run — experiment directory contains `metadata.json` with `run`, `config`, `system`, `metrics`, and `custom` sections populated at run completion.

FR-003: Marco can run identical experiments across ≥5 seeds and generate variance band plots — `--seed` parameter accepted; variance band plot generated from multi-seed results without manual scripting.

FR-004: Marco can access a pre-filled markdown experiment summary template after each run — template populated with config, system info, and seed, written to the experiment run directory at completion.

FR-005: Marco can extend experiment tracking with custom RL logic via base class hooks without modifying core training code — `CurriculumCallback`, `ActionFilterWrapper`, `GoalConditionedWrapper`, and `DomainRandomizationWrapper` base classes are present with intentionally empty core hook methods.

FR-006: Marco can inspect per-run curriculum stage progression in TensorBoard during and after training — curriculum stage metric logged at each stage transition; visible in TensorBoard as a `curriculum/stage` scalar series.

FR-007: Marco can validate user-implemented callback logic with unit tests before running full experiments — test scaffolding in `tests/test_callbacks.py` accepts mock environment context; passes without live training run.

FR-008: Marco can query all experiment results by phase and export to CSV — `robot-lab results --phase N` returns a table with seeds, final rewards, and KPI values; CSV export functional.

FR-009: Marco can query `ResultsDatabase` for any completed run by run ID, seed, phase, or date range — query interface accessible both programmatically and via CLI.

FR-S01 [Stretch — Phase 4]: Marco can evaluate a trained policy in Gazebo with injected actuator dynamics, observation latency, and contact noise — `robot-lab gazebo-eval` produces success rate, energy, and recovery metrics; conditional `try: import rclpy` import guard ensures base package installs and runs without ROS 2.

FR-S02 [Stretch — Phase 4]: Marco can run a perturbation sweep and generate a success-vs-perturbation-magnitude degradation curve — sweep runner accepts parameter range config; outputs matplotlib degradation curve and CSV export.

### NonFunctional Requirements

NFR-001 [Performance]: Training throughput — full-phase experiments (Phase 0.5–Phase 2) complete within the 48h wall-clock budget on GTX 1080 (8 GB VRAM); experiments exceeding this are flagged as hardware-constrained in the results database and deferred — not discarded.

NFR-002 [Performance]: CLI cold-start latency — `robot-lab --help` and config-loading commands respond within 3 seconds; achieved via lazy imports (`torch` and `stable_baselines3` not imported at CLI entry point).

NFR-003 [Performance]: TensorBoard logging overhead — step-level logging adds no more than 5% wall-clock overhead vs unlogged training; default logging frequency is every 100 steps, not every step.

NFR-004 [Performance]: Vectorized environment requirement — training runs use `SubprocVecEnv` with `num_envs ≥ 4` by default; single-env training is not the performance baseline.

NFR-005 [Reliability]: Experiment fault tolerance — checkpoint callbacks save intermediate model+VecNorm pairs at configurable intervals (default: every 50k steps); a run interrupted after the first checkpoint is recoverable from saved state.

NFR-006 [Reliability]: Experiment completeness tracking — `ExperimentTracker` marks every run as `COMPLETED`, `INTERRUPTED`, or `FAILED`; non-`COMPLETED` runs are flagged in the results database and never silently included in comparative analysis.

NFR-007 [Reliability]: VecNormalize integrity — model checkpoints and VecNormalize statistics are always saved as atomic pairs; a model checkpoint without its matching VecNorm `.pkl` file is a corrupt artifact — save path logic enforces this invariant.

NFR-008 [Reliability]: Config immutability — the JSON config active at run-start is copied into the experiment directory before the first training step; mutations to the source config after this point have no effect on the running experiment.

NFR-009 [Maintainability]: Lint cleanliness — `ruff check robot_lab/` and `ruff format robot_lab/` pass with zero violations before any merge; max line length 100 characters.

NFR-010 [Maintainability]: Type hint coverage — all public functions in all public modules have full type hints (args + return type).

NFR-011 [Maintainability]: Test baseline — core infrastructure modules are covered by at least smoke tests; `make test` passes before any implementation PR merges.

NFR-012 [Maintainability]: Docstring standard — Google-style docstrings with Args/Returns/Raises sections on all public API functions; inline comments required for non-obvious architectural decisions.

NFR-013 [Maintainability]: Import safety — `import robot_lab` and `from robot_lab.x import y` never trigger I/O, training, or network calls; validated by the smoke test suite.

NFR-014 [Integration]: PyTorch version ceiling — hard-capped at `<2.10` in `pyproject.toml`; sm_61 (GTX 1080) support drops in 2.10+; any version bump gated on explicit hardware compatibility verification.

NFR-015 [Integration]: SB3 API stability — Stable-Baselines3 and Gymnasium versions pinned in `pyproject.toml`; upgrades require running the full training smoke test before merging.

NFR-016 [Integration]: Gymnasium API conformance — all custom environments (`GripperEnv-v0`, `A1Quadruped-v0`) pass Gymnasium's `check_env()` validation at registration time.

NFR-017 [Integration]: MuJoCo rendering constraint — `SubprocVecEnv` is forbidden on all visualization paths on Windows (GL context sharing incompatibility); `DummyVecEnv` is required — this is a hard architectural constraint, not a configuration option.

NFR-018 [Integration]: Gazebo integration isolation — Gazebo harness isolated behind a conditional import (`try: import rclpy`) so absence of ROS 2 never breaks base package install.

### Additional Requirements

From Architecture:

- AR-001: Plugin/Registry Architecture — Three separate module-level singleton registries (`MetricsRegistry`, `VisualizationRegistry`, `MetadataRegistry`) in `robot_lab/experiments/plugins/`; each with distinct lifecycle hooks (`on_step`/`on_episode_end`/`on_eval`, `render(results)`, `collect(context)`).
- AR-002: Two-layer JSON storage — per-run `metadata.json` (single nested dict with `run`, `config`, `system`, `metrics`, `custom` sections) + `results_index.jsonl` append-only index at `data/experiments/results_index.jsonl` (one line per run for fast phase queries).
- AR-003: `RobotLabCallbackMixin` — thin mixin over SB3 `BaseCallback` that adds ExperimentTracker + TensorBoard wiring only; `CurriculumCallback` subclass with `_should_advance()` intentionally empty (YouAreLazy boundary).
- AR-004: `RobotLabCheckpointCallback` — SB3 callback subclass that overrides `_on_step` to always save model `.zip` + VecNormalize `.pkl` as an atomic pair; wired into `train()` by default.
- AR-005: Scaffolded wrappers in `robot_lab/wrappers.py` — `RobotLabWrapperMixin`, `ActionFilterWrapper` (empty `_apply_filter()`), `GoalConditionedWrapper` (empty `_sample_goal()`), `DomainRandomizationWrapper` (empty `_sample_params()`).
- AR-006: `ExperimentTracker` brownfield migration — refactor from multi-file schema to consolidated `metadata.json` (breaking change; must be sequenced before Phase 0.5 experiment runs).
- AR-007: `ResultsDatabase` extension — add JSONL index append at `end_run()`, streaming phase query from index, full `metadata.json` load on demand.
- AR-008: `results` CLI command — `robot-lab results --phase N` → thin wrapper over `ResultsDatabase.query()`.
- AR-009: Wrapper chaining order (mandatory) — `gym.make()` → `Monitor` → `[RobotLabWrappers]` → `VecEnv` → `VecNormalize`; deviations break SB3 callback statistics.
- AR-010: `make_env()` as sole env factory — `gym.make()` is forbidden in training, evaluation, or experiment code; `make_env()` is the single point of Monitor wrapping and seeding.
- AR-011: Experiment summary template — written to `data/experiments/{name}/runs/{run_id}/experiment_summary.md` by `tracker.end_run()`; populated with `metadata.json["run"]` and `metadata.json["config"]`.
- AR-012: Multi-seed runner — `ExperimentRunner` is sequential by default (`parallel_seeds: false`); parallel opt-in with explicit VRAM warning; GTX 1080 cannot reliably run two concurrent MuJoCo processes.
- AR-013: Plugin registration discipline — plugins must NOT register themselves on import; global plugins registered via `_register_defaults()` called lazily on first registry access; run-scoped plugins registered in experiment script before `train()`.
- AR-014: Seeding protocol — at training start: `random.seed(seed)`, `np.random.seed(seed)`, `model = SAC(..., seed=seed)`, `env.reset(seed=seed)` — all four required.
- AR-015: `generate_run_id()` — the only permitted way to create run IDs; format `{YYYYMMDD_HHMMSS}_{8char_hash}_{suffix}`; never constructed manually.
- AR-016: Implementation sequence — plugins → metadata.json/JSONL/ResultsDB → CurriculumCallback scaffold → RobotLabCheckpointCallback → scaffolded wrappers.

### FR Coverage Map

| FR | Epic | Rationale |
|---|---|---|
| FR-001 | Epic 1 | `train()` wired with new tracker + checkpoint callback |
| FR-002 | Epic 1 | `metadata.json` schema + `ExperimentTracker` migration |
| FR-003 | Epic 3 | Multi-seed `ExperimentRunner` + variance band plotting |
| FR-004 | Epic 1 | `experiment_summary.md` written by `tracker.end_run()` |
| FR-005 | Epics 2, 4, 5 | ActionFilterWrapper (Epic 2), CurriculumCallback (Epic 4), GoalConditioned + DR (Epic 5) |
| FR-006 | Epic 4 | `RobotLabCallbackMixin` TensorBoard `curriculum/stage` scalar |
| FR-007 | Epic 4 | `tests/test_callbacks.py` mock env harness |
| FR-008 | Epic 3 | `robot-lab results --phase N` CLI + CSV export |
| FR-009 | Epic 3 | `ResultsDatabase.query()` over JSONL index |
| FR-S01 | Epic 6 | `gazebo-eval` CLI with `try: import rclpy` guard |
| FR-S02 | Epic 6 | Perturbation sweep runner + degradation curve plotter |

## Epic List

### Epic 1: Core Experiment Infrastructure
Marco can run reproducible, fault-tolerant experiments with automatic metadata capture, a unified `metadata.json` schema, plugin-extensible observability, and atomic model+VecNorm checkpoints — replacing the current multi-file tracker schema.
**FRs covered:** FR-001, FR-002, FR-004
**ARs addressed:** AR-001, AR-002, AR-004, AR-005 (partial), AR-006, AR-007, AR-009 through AR-016

### Epic 2: Trajectory Smoothness Research Platform
Marco can run Phase 0.5 trajectory comparison experiments with a ready-to-fill `ActionFilterWrapper` scaffold, automated smoothness metric hooks (`∑‖aₜ−aₜ₋₁‖²`), and video render pipeline — enabling the first research checkpoint.
**FRs covered:** FR-005 (ActionFilterWrapper)
**Builds on:** Epic 1

### Epic 3: Results Analysis & Reporting
Marco can query all experiment results by phase, export CSV, run multi-seed experiments from a YAML config, and generate variance band plots — making comparative claims statistically valid.
**FRs covered:** FR-003, FR-008, FR-009
**Builds on:** Epic 1

### Epic 4: Curriculum Learning Scaffold
Marco can implement curriculum advancement logic in a ready-to-fill `CurriculumCallback` scaffold with TensorBoard stage tracking already wired, and validate his implementation with unit tests before running full experiments.
**FRs covered:** FR-005 (CurriculumCallback), FR-006, FR-007
**Builds on:** Epic 1

### Epic 5: Goal-Conditioned & Domain Randomization Scaffolds
Marco can extend experiments with goal-conditioned RL (`GoalConditionedWrapper`, zero-shot eval harness) and domain randomization (`DomainRandomizationWrapper`, curriculum-DR combo runner) — unlocking Phase 2 research.
**FRs covered:** FR-005 (GoalConditionedWrapper, DomainRandomizationWrapper)
**Builds on:** Epics 1, 4

### Epic 6: Sim2Real Validation Harness *(Stretch — Phase 4)*
Marco can run perturbation sweeps, generate success-vs-magnitude degradation curves, and evaluate policies in Gazebo with injected dynamics — all without requiring ROS 2 for the base package install.
**FRs covered:** FR-S01, FR-S02
**Builds on:** Epics 1–5

---

## Epic 1: Core Experiment Infrastructure

Marco can run reproducible, fault-tolerant experiments with automatic metadata capture, a unified `metadata.json` schema, plugin-extensible observability, and atomic model+VecNorm checkpoints — replacing the current multi-file tracker schema.

### Story 1.1: Plugin/Registry Infrastructure

As a developer (Marco),
I want a `robot_lab/experiments/plugins/` module with three separate registries (`MetricsRegistry`, `VisualizationRegistry`, `MetadataRegistry`) and their abstract base classes,
So that I can extend experiment observability by registering an interface implementation — with zero changes to `training.py`, `tracker.py`, or `visualization.py`.

**Acceptance Criteria:**

**Given** the package is installed,
**When** I run `from robot_lab.experiments.plugins import register_metric_plugin, register_visualization_plugin, register_metadata_plugin, MetricsPlugin, VisualizationPlugin, MetadataPlugin`,
**Then** all six names are importable with no errors.

**Given** I execute `import robot_lab` with no experiment running,
**When** the import completes,
**Then** no plugins are auto-registered and no files are created (no side effects on import).

**Given** any registry is accessed for the first time,
**When** `_register_defaults()` is triggered lazily,
**Then** the built-in default plugins (basic reward logging, system metadata) are registered; repeated calls are idempotent.

**Given** a plugin is registered via `register_metric_plugin()` in an experiment script,
**When** a second, independent experiment runs in the same process,
**Then** run-scoped plugins from the first experiment do not affect the second run.

---

### Story 1.2: Consolidated Experiment Metadata Schema

As a researcher (Marco),
I want `ExperimentTracker` to write a single `metadata.json` per run with nested `run`, `config`, `system`, `metrics`, and `custom` sections (replacing the current multi-file schema),
So that each experiment run is fully self-contained and human-readable without cross-referencing separate files.

**Acceptance Criteria:**

**Given** `tracker.start_run()` is called before the first training step,
**When** the file is written,
**Then** `metadata.json["run"]["status"]` equals `"RUNNING"`, `metadata.json["config"]` contains an immutable full-config snapshot, and `metadata.json["system"]` contains Python version, GPU info, and git commit hash.

**Given** `tracker.end_run("COMPLETED")` is called at run completion,
**When** the file is updated,
**Then** `metadata.json["run"]["status"]` equals `"COMPLETED"` and `metadata.json["run"]["finished_at"]` is a valid ISO timestamp.

**Given** an unhandled exception occurs during training,
**When** the `finally` block in `train()` executes,
**Then** `metadata.json["run"]["status"]` equals `"FAILED"`.

**Given** a `KeyboardInterrupt` occurs,
**When** the `finally` block in `train()` executes,
**Then** `metadata.json["run"]["status"]` equals `"INTERRUPTED"`.

**Given** a `MetadataPlugin` calls `tracker.update("custom", data)`,
**When** data is merged into the file,
**Then** plugin outputs appear under `metadata.json["custom"]` only — the `run`, `config`, `system`, and `metrics` keys are unmodified.

**Given** `make test` is run after this story is implemented,
**When** `test_tracker.py` executes,
**Then** all tracker schema tests pass.

---

### Story 1.3: Atomic Model + VecNorm Checkpointing

As a researcher (Marco),
I want a `RobotLabCheckpointCallback` that always saves model `.zip` and VecNormalize `.pkl` as an atomic pair, wired into every training run by default,
So that no checkpoint can become a corrupt artifact (a model without its matching VecNorm stats).

**Acceptance Criteria:**

**Given** `train()` is called by any caller (CLI or library),
**When** the training callback list is assembled,
**Then** `RobotLabCheckpointCallback` is present without the caller explicitly adding it.

**Given** `RobotLabCheckpointCallback._on_step()` fires at a checkpoint interval,
**When** the model `.zip` is saved,
**Then** the VecNorm `.pkl` save immediately follows within the same method — they are never written independently.

**Given** a training run completes normally,
**When** I inspect the `data/models/best/` directory,
**Then** both `best_model.zip` and `best_model_vecnorm.pkl` are present as a matched pair.

**Given** `make test` is run after implementation,
**When** `test_callbacks.py` executes the checkpoint tests,
**Then** the atomic-pair invariant is verified without running a full live training session.

---

### Story 1.4: Results Index + Experiment Summary Template

As a researcher (Marco),
I want `tracker.end_run()` to append a lightweight summary line to `data/experiments/results_index.jsonl` and write a pre-filled `experiment_summary.md` to the run directory,
So that I can query runs across experiments without loading full metadata files, and have a portfolio-ready document waiting after every run.

**Acceptance Criteria:**

**Given** `tracker.end_run()` is called on run completion,
**When** the JSONL index is updated,
**Then** a single valid JSON line is appended to `data/experiments/results_index.jsonl` containing at minimum: `run_id`, `experiment`, `seed`, `phase`, `final_reward`, `status`, `timestamp`.

**Given** `tracker.end_run()` is called,
**When** the summary template is written,
**Then** `experiment_summary.md` is present in the run directory with `metadata.json["run"]` and `metadata.json["config"]` values substituted into the template placeholders.

**Given** 10 runs have been recorded across sessions,
**When** I read `data/experiments/results_index.jsonl`,
**Then** it contains exactly 10 newline-delimited JSON lines in append order, each independently valid JSON.

**Given** `import robot_lab` is executed with no runs started,
**When** the import completes,
**Then** `results_index.jsonl` is not created and no disk writes occur.

---

### Story 1.5: End-to-End Training Pipeline Smoke-Test

As a researcher (Marco),
I want `robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42` to execute a complete tracked run with the new pipeline fully wired (plugin registry, `ExperimentTracker`, `RobotLabCheckpointCallback`, atomic VecNorm saves, JSONL index),
So that I can confirm all Epic 1 components work together and the one-command experiment launch is fully functional.

**Acceptance Criteria:**

**Given** `robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42` is run,
**When** the run completes,
**Then** the run directory contains: `metadata.json` (nested schema with all five sections), `experiment_summary.md`, at least one checkpoint `.zip`+`.pkl` pair, and `results_index.jsonl` is updated.

**Given** `train()` is called with `seed=42`,
**When** training begins,
**Then** `random.seed(42)`, `np.random.seed(42)`, and `model.seed=42` are all set before the first training step.

**Given** `make test` is run after Epic 1 is complete,
**When** all tests execute,
**Then** `test_smoke.py`, `test_training.py`, `test_tracker.py`, `test_plugins.py`, and `test_callbacks.py` all pass with zero failures.

---

## Epic 2: Trajectory Smoothness Research Platform

Marco can run Phase 0.5 trajectory comparison experiments with a ready-to-fill `ActionFilterWrapper` scaffold, automated smoothness metric hooks, and video render pipeline — enabling the first research checkpoint.

### Story 2.1: ActionFilterWrapper Scaffold

As a researcher (Marco),
I want an `ActionFilterWrapper` base class in `robot_lab/wrappers.py` with `RobotLabWrapperMixin` providing tracker parameter logging, and an intentionally empty `_apply_filter(action)` method,
So that I have a Gym wrapper skeleton where I can implement any action filtering strategy (low-pass, EMA, splines) without modifying any infrastructure code.

**Acceptance Criteria:**

**Given** `from robot_lab.wrappers import ActionFilterWrapper` is executed,
**When** the import completes,
**Then** `ActionFilterWrapper` is importable, subclasses `gym.Wrapper`, and mixes in `RobotLabWrapperMixin`; no error is raised.

**Given** `ActionFilterWrapper._apply_filter(action)` is inspected,
**When** the method body is read,
**Then** the method raises `NotImplementedError` with the message `"Implement _apply_filter() to define your filtering logic."` — confirming the YouAreLazy boundary.

**Given** a subclass of `ActionFilterWrapper` implements `_apply_filter(action)` and wraps an environment,
**When** `env.step(action)` is called,
**Then** the wrapper's `_apply_filter(action)` is called with the raw action before passing the result to the inner environment.

**Given** the wrapper is constructed with an `ExperimentTracker` instance passed in,
**When** the wrapper is initialized,
**Then** `RobotLabWrapperMixin` logs the wrapper class name and any init parameters to the tracker under `metadata.json["config"]`.

**Given** the wrapper chaining order `Monitor → ActionFilterWrapper → VecEnv → VecNormalize` is used,
**When** `make_env()` builds the environment stack,
**Then** `Monitor` remains the first wrapper applied and `VecNormalize` remains the last.

---

### Story 2.2: Smoothness Metric Plugin

As a researcher (Marco),
I want an `ActionSmoothnessMetricPlugin` registered as a built-in default plugin that logs `∑‖aₜ−aₜ₋₁‖²` per episode to TensorBoard under `smoothness/action_delta_norm` and into `metadata.json["metrics"]`,
So that smoothness is automatically tracked for any training run without requiring me to write metric collection code.

**Acceptance Criteria:**

**Given** `plugins/defaults.py` is loaded via lazy `_register_defaults()`,
**When** `ActionSmoothnessMetricPlugin` is registered,
**Then** it appears in `MetricsRegistry.list_plugins()` without any manual registration call in experiment scripts.

**Given** a training run is active and `MetricsRegistry.on_episode_end(context)` fires,
**When** the plugin processes the episode,
**Then** `smoothness/action_delta_norm` is logged as a TensorBoard scalar for that episode.

**Given** `tracker.end_run("COMPLETED")` is called,
**When** the final `metadata.json` is written,
**Then** `metadata.json["metrics"]["smoothness_action_delta_norm"]` contains the per-episode series (or at minimum the final mean value).

**Given** `ActionSmoothnessMetricPlugin` is imported at the module level in `defaults.py`,
**When** `import robot_lab` is executed without running an experiment,
**Then** no metric computation occurs and no files are written (plugin logic is lifecycle-gated, not import-triggered).

---

### Story 2.3: Video Render Pipeline

As a researcher (Marco),
I want `robot-lab visualize --env MountainCarContinuous-v0 --algo SAC` to record a policy rollout and export an MP4 to the experiment directory,
So that I can generate visual evidence of policy behavior for portfolio documentation and qualitative trajectory comparison.

**Acceptance Criteria:**

**Given** a trained model `.zip` and matching `_vecnorm.pkl` exist in `data/models/`,
**When** `robot-lab visualize --env MountainCarContinuous-v0 --algo SAC` is run,
**Then** an MP4 file is written to `data/experiments/{experiment_name}/runs/{run_id}/` with a non-zero file size.

**Given** the visualization path is executed on Windows,
**When** the environment is constructed for rollout,
**Then** `DummyVecEnv` is used — `SubprocVecEnv` is never instantiated on the visualization path.

**Given** a VecNorm `.pkl` file is missing from the expected paired path,
**When** `visualize()` is called,
**Then** a `ValueError` is raised with the message `"[Visualize] VecNorm stats file not found at {path}. Ensure model and VecNorm were saved as a pair."` — rather than silently loading an unnormalized policy.

**Given** `make test-fast` is run after this story,
**When** the smoke tests execute,
**Then** `test_smoke.py` visualization path passes without GPU (uses CPU + DummyVecEnv).

---

## Epic 3: Results Analysis & Reporting

Marco can query all experiment results by phase, export CSV, run multi-seed experiments from a YAML config, and generate variance band plots — making comparative claims statistically valid.

### Story 3.1: ResultsDatabase Phase Query

As a researcher (Marco),
I want `ResultsDatabase.query(phase=N)` to stream `results_index.jsonl` for fast filtering and load full `metadata.json` on demand for matched runs,
So that I can retrieve all runs for a given research phase without scanning individual experiment directories.

**Acceptance Criteria:**

**Given** `results_index.jsonl` contains 20 run entries across phases 0, 1, and 2,
**When** `ResultsDatabase.query(phase=1)` is called,
**Then** only runs with `"phase": 1` are returned; the JSONL file is streamed (not fully loaded into memory) for the initial filter pass.

**Given** `ResultsDatabase.query(run_id="20260313_abc12345_sac")` is called,
**When** the query executes,
**Then** the full `metadata.json` for that run is loaded and returned as a dict.

**Given** `ResultsDatabase.query(phase=1, seed=3)` is called,
**When** the query executes,
**Then** only entries matching both `phase=1` AND `seed=3` are returned.

**Given** `results_index.jsonl` does not exist yet,
**When** `ResultsDatabase.query(phase=0)` is called,
**Then** an empty list is returned — no `FileNotFoundError` is raised.

**Given** `make test` is run after implementation,
**When** `test_results_db.py` executes,
**Then** all query filter tests pass using a temporary JSONL fixture (no live training required).

---

### Story 3.2: Results CLI Command + CSV Export

As a researcher (Marco),
I want `robot-lab results --phase N` to print a formatted table of all runs for that phase and `robot-lab results --phase N --export results.csv` to write the table to CSV,
So that I can review experiment outcomes at a glance and load them into notebooks for analysis.

**Acceptance Criteria:**

**Given** `robot-lab results --phase 1` is run with three completed runs in the index,
**When** the command executes,
**Then** a Rich-formatted table is printed to stdout with columns: `run_id`, `seed`, `final_reward`, `status`, `timestamp`; non-`COMPLETED` runs are visually flagged (e.g., dimmed or marked `⚠`).

**Given** `robot-lab results --phase 1 --export results.csv` is run,
**When** the command completes,
**Then** `results.csv` is written to the current working directory with a header row and one data row per matched run.

**Given** no runs exist for `--phase 99`,
**When** `robot-lab results --phase 99` is run,
**Then** the command prints `"No results found for phase 99."` and exits with code 0 (not an error).

**Given** `robot-lab --help` is run after implementation,
**When** the help text is displayed,
**Then** the `results` command appears in the command list with a one-line description.

---

### Story 3.3: Multi-Seed Experiment Runner

As a researcher (Marco),
I want `ExperimentRunner` to accept a YAML experiment config specifying `seeds: [42, 43, 44, 45, 46]` and run each seed sequentially (with optional parallel opt-in),
So that I can launch a statistically valid multi-seed experiment from a single command rather than manually re-running five times.

**Acceptance Criteria:**

**Given** an experiment YAML config with `seeds: [42, 43, 44, 45, 46]` and `algorithm: SAC`,
**When** `robot-lab run --config experiments/1_curriculum/configs/manual_curriculum.yaml` is executed,
**Then** five sequential training runs are launched, each with a distinct seed, each producing its own run directory and `metadata.json`.

**Given** `parallel_seeds: true` is set in the YAML config,
**When** the runner starts,
**Then** a `⚠` warning is printed: `"Parallel seed execution enabled. VRAM budget (~8GB) may be exceeded on GTX 1080."` before launching parallel processes.

**Given** one seed in a multi-seed run raises an unhandled exception,
**When** the runner catches the error,
**Then** that run's `metadata.json["run"]["status"]` is set to `"FAILED"`, the runner logs the error, and execution continues with the remaining seeds — the entire batch is not aborted.

**Given** `ExperimentRunner` is imported as a library,
**When** `ExperimentRunner.run(config_path=...)` is called programmatically,
**Then** the same sequential execution and per-seed tracking behaviour applies as the CLI path.

---

### Story 3.4: Variance Band Plot Generation

As a researcher (Marco),
I want `robot-lab plot --phase 1` (or `make plot`) to generate a matplotlib learning curve plot with variance bands (mean ± std across seeds) for all `COMPLETED` runs in a phase,
So that I can visually compare curriculum vs no-curriculum variants with statistical validity in one command.

**Acceptance Criteria:**

**Given** `robot-lab plot --phase 1` is run with five completed runs (one per seed) for two experiment variants,
**When** the plot is generated,
**Then** a PNG is saved to `data/experiments/phase_1_comparison.png` showing mean reward curves with shaded ±1 std band for each variant, with a legend labelling each variant by experiment name.

**Given** only a single seed exists for a variant,
**When** the plot is generated for that variant,
**Then** the curve is plotted without a variance band and a `⚠` warning is logged: `"Variant '{name}' has only 1 seed — variance band omitted."`.

**Given** `robot-lab plot --phase 1 --export plot.png --show`,
**When** the command completes,
**Then** `plot.png` is written to the specified path AND the plot is displayed interactively via `matplotlib.pyplot.show()`.

**Given** no `COMPLETED` runs exist for the given phase,
**When** `robot-lab plot --phase 0` is run,
**Then** the command prints `"No completed runs found for phase 0."` and exits with code 0.

---

## Epic 4: Curriculum Learning Scaffold

Marco can implement curriculum advancement logic in a ready-to-fill `CurriculumCallback` scaffold with TensorBoard stage tracking already wired, and validate his implementation with unit tests before running full experiments.

### Story 4.1: RobotLabCallbackMixin + CurriculumCallback Scaffold

As a researcher (Marco),
I want a `RobotLabCallbackMixin` that wires `ExperimentTracker` and TensorBoard logging into SB3 callbacks, and a `CurriculumCallback` subclass with an intentionally empty `_should_advance()` method and stage state tracking,
So that I have a scaffold where I can implement my own curriculum advancement condition without touching any infrastructure wiring.

**Acceptance Criteria:**

**Given** `from robot_lab.utils.callbacks import RobotLabCallbackMixin, CurriculumCallback` is executed,
**When** the import completes,
**Then** both names are importable with no errors; `CurriculumCallback` inherits from both `RobotLabCallbackMixin` and SB3's `BaseCallback`.

**Given** `CurriculumCallback._should_advance()` is inspected,
**When** the method body is read,
**Then** it raises `NotImplementedError` with the message `"Implement _should_advance() to define your curriculum advancement condition."` — confirming the YouAreLazy boundary.

**Given** a subclass of `CurriculumCallback` implements `_should_advance()` and the callback fires a stage transition,
**When** the stage advances from N to N+1,
**Then** `RobotLabCallbackMixin` logs the scalar `curriculum/stage` with value N+1 to TensorBoard, and calls `tracker.update("custom", {"curriculum_stage": N+1})` — with no RL logic in the mixin itself.

**Given** `CurriculumCallback` is initialized with a stage count of 3,
**When** `_should_advance()` is never called (because the subclass isn't implemented),
**Then** the callback runs silently without raising on each `on_step()` call — the `NotImplementedError` is only raised when `_should_advance()` is explicitly invoked.

**Given** `make test` is run after implementation,
**When** `test_callbacks.py` executes,
**Then** the mixin wiring tests pass using a mock `ExperimentTracker` and mock TensorBoard writer — no live training required.

---

### Story 4.2: Curriculum Callback Unit Test Harness

As a researcher (Marco),
I want `tests/test_callbacks.py` scaffolded with fixtures for a mock SB3 environment context, a mock `ExperimentTracker`, and a helper to instantiate `CurriculumCallback` subclasses for testing,
So that I can write unit tests for my `_should_advance()` implementation and verify it behaves correctly before committing a multi-hour training run.

**Acceptance Criteria:**

**Given** `test_callbacks.py` exists in `tests/`,
**When** the file is inspected,
**Then** it contains: a `mock_env_context` fixture (provides a minimal SB3 `locals` dict with episode rewards and step count), a `mock_tracker` fixture (in-memory `ExperimentTracker` that writes to `temp_output_dir`), and at least one example test demonstrating how to subclass `CurriculumCallback` for testing.

**Given** the example test in `test_callbacks.py` instantiates a trivial `CurriculumCallback` subclass where `_should_advance()` always returns `True`,
**When** the test calls `callback.on_step()` with the mock context,
**Then** the test asserts that `curriculum/stage` increments and the mock tracker received an update — demonstrating the harness works end-to-end.

**Given** the example test exercises a subclass where `_should_advance()` always returns `False`,
**When** the test calls `callback.on_step()` 100 times,
**Then** the stage never increments and `tracker.update()` is never called with a stage change — verifying the advancement condition gate works.

**Given** `make test-fast` is run,
**When** `test_callbacks.py` executes,
**Then** all tests complete in under 5 seconds with no live environment or GPU required.

---

### Story 4.3: Curriculum Stage Metrics in TensorBoard

As a researcher (Marco),
I want the `curriculum/stage` TensorBoard scalar to be visible during a live training run — updating in real time as stages advance — so that I can monitor curriculum progression without waiting for the run to complete.

**Acceptance Criteria:**

**Given** a `CurriculumCallback` subclass is active in a training run and `_should_advance()` returns `True` at step 10,000,
**When** I open TensorBoard while training is in progress,
**Then** the `curriculum/stage` scalar is visible and its value is `1` (advanced from stage 0).

**Given** a training run logs `curriculum/stage` across multiple transitions,
**When** the TensorBoard scalar tab is viewed after run completion,
**Then** the `curriculum/stage` series shows a monotonically non-decreasing step function over training timesteps.

**Given** the TensorBoard log directory follows the naming convention `{output_dir}/logs/{algo}_{env}_parallel/`,
**When** `robot-lab tensorboard` is run after a curriculum training run,
**Then** TensorBoard opens and `curriculum/stage` appears under the correct run tag without manual path configuration.

**Given** no `CurriculumCallback` subclass is registered for a training run,
**When** the run completes,
**Then** no `curriculum/stage` scalar appears in TensorBoard — the metric is absent, not zero.

---

## Epic 5: Goal-Conditioned & Domain Randomization Scaffolds

Marco can extend experiments with goal-conditioned RL and domain randomization using ready-to-fill wrapper scaffolds, a zero-shot evaluation harness, and a curriculum-DR combo runner — unlocking Phase 2 research.

### Story 5.1: GoalConditionedWrapper Scaffold

As a researcher (Marco),
I want a `GoalConditionedWrapper` base class in `robot_lab/wrappers.py` with `RobotLabWrapperMixin`, an intentionally empty `_sample_goal()` method, and observation augmentation hooks,
So that I have a Gym wrapper skeleton where I can implement goal representation and sampling strategy without modifying any infrastructure code.

**Acceptance Criteria:**

**Given** `from robot_lab.wrappers import GoalConditionedWrapper` is executed,
**When** the import completes,
**Then** `GoalConditionedWrapper` is importable, subclasses `gym.Wrapper`, and mixes in `RobotLabWrapperMixin`; no error is raised.

**Given** `GoalConditionedWrapper._sample_goal()` is inspected,
**When** the method body is read,
**Then** it raises `NotImplementedError` with the message `"Implement _sample_goal() to define your goal sampling strategy."`.

**Given** a subclass implements `_sample_goal()` returning a goal vector,
**When** `env.reset()` is called,
**Then** the wrapper calls `_sample_goal()`, and the returned observation is the concatenation of the base observation and the goal vector — with the new `observation_space` correctly reflecting the augmented shape.

**Given** `RobotLabWrapperMixin` is active,
**When** the wrapper is initialized with an `ExperimentTracker`,
**Then** the wrapper class name and goal space shape are logged to `metadata.json["config"]`.

**Given** `make test` is run after implementation,
**When** `test_wrappers.py` exercises `GoalConditionedWrapper`,
**Then** the observation augmentation shape test passes using a mock environment — no MuJoCo required.

---

### Story 5.2: Zero-Shot Evaluation Harness

As a researcher (Marco),
I want an `evaluate_zero_shot(model, env_configs: list[dict], n_episodes: int)` utility function that evaluates a trained policy across a list of novel goal configurations and returns per-config success rates,
So that I can measure generalization to unseen goals after Phase 2 training without writing custom evaluation loops.

**Acceptance Criteria:**

**Given** `from robot_lab.experiments.runner import evaluate_zero_shot` is executed,
**When** the import completes,
**Then** the function is importable with the signature `evaluate_zero_shot(model, env_configs: list[dict], n_episodes: int = 10) -> list[dict]`.

**Given** a trained model and a list of 3 novel goal configs are passed,
**When** `evaluate_zero_shot()` runs,
**Then** it returns a list of 3 dicts, each containing at minimum: `env_config`, `success_rate` (float 0–1), `mean_reward` (float), `n_episodes` (int).

**Given** the evaluation harness runs `n_episodes=10` per config,
**When** any single episode raises an exception,
**Then** that episode is counted as a failure (success=False), logged as `⚠`, and evaluation continues — the harness never crashes on a single bad episode.

**Given** results are returned from `evaluate_zero_shot()`,
**When** they are passed to `tracker.update("metrics", {"zero_shot_eval": results})`,
**Then** the results are persisted under `metadata.json["metrics"]["zero_shot_eval"]` in the run's output.

---

### Story 5.3: DomainRandomizationWrapper Scaffold

As a researcher (Marco),
I want a `DomainRandomizationWrapper` base class in `robot_lab/wrappers.py` with an intentionally empty `_sample_params()` method and parameter logging hooks,
So that I have a wrapper skeleton where I can implement physics parameter distributions for domain randomization without modifying infrastructure code.

**Acceptance Criteria:**

**Given** `from robot_lab.wrappers import DomainRandomizationWrapper` is executed,
**When** the import completes,
**Then** `DomainRandomizationWrapper` is importable, subclasses `gym.Wrapper`, and mixes in `RobotLabWrapperMixin`.

**Given** `DomainRandomizationWrapper._sample_params()` is inspected,
**When** the method body is read,
**Then** it raises `NotImplementedError` with the message `"Implement _sample_params() to define your domain randomization parameter distributions."`.

**Given** a subclass implements `_sample_params()` returning `{"gravity": -9.2, "friction": 0.8}`,
**When** `env.reset()` is called,
**Then** the wrapper calls `_sample_params()` and the returned parameter dict is applied to the environment's physics model before the episode begins.

**Given** `RobotLabWrapperMixin` is active during a training run,
**When** `_sample_params()` is called at each reset,
**Then** the sampled parameter dict is logged to TensorBoard under `domain_randomization/{param_name}` and appended into `metadata.json["custom"]["dr_params"]`.

**Given** `make test` is run after implementation,
**When** `test_wrappers.py` exercises `DomainRandomizationWrapper`,
**Then** the parameter sampling hook test passes with a mock environment.

---

### Story 5.4: Curriculum-DR Combo Runner

As a researcher (Marco),
I want `ExperimentRunner` to support a YAML config field `wrappers: [...]` that composes wrappers in the correct chaining order alongside a curriculum callback for a single training run,
So that I can run Phase 2 curriculum+DR combination experiments from a single config without writing custom setup scripts.

**Acceptance Criteria:**

**Given** an experiment YAML config specifies `wrappers: ["DomainRandomizationWrapper"]` and a `CurriculumCallback` subclass in the callbacks list,
**When** `ExperimentRunner` builds the training run,
**Then** the environment stack follows: `Monitor → DomainRandomizationWrapper → VecEnv → VecNormalize`, and the curriculum callback is wired into the SB3 callback list.

**Given** both `DomainRandomizationWrapper` and `GoalConditionedWrapper` are specified in the YAML wrappers list,
**When** the environment stack is built,
**Then** the wrappers are applied in the order they appear in the YAML config (innermost first after `Monitor`), and a `logger.info` line confirms the applied stack order.

**Given** an invalid wrapper class name is specified in the YAML config,
**When** `ExperimentRunner` attempts to resolve it,
**Then** a `ValueError` is raised with the message `"[Runner] Unknown wrapper '{name}'. Available wrappers: {list}."` — the run does not silently start with a broken stack.

**Given** the combo runner YAML config is used for a multi-seed run with `seeds: [42, 43, 44, 45, 46]`,
**When** all five runs complete,
**Then** each run's `metadata.json["config"]` records the applied wrapper stack — enabling full reproducibility from config alone.

---

## Epic 6: Sim2Real Validation Harness *(Stretch — Phase 4)*

Marco can run perturbation sweeps, generate success-vs-magnitude degradation curves, and evaluate policies in Gazebo with injected dynamics — all without requiring ROS 2 for the base package install.

### Story 6.1: Perturbation Sweep Runner + Degradation Curve

As a researcher (Marco),
I want a `PerturbationSweepRunner` that accepts a YAML config specifying a parameter name, value range, and step count, evaluates a trained policy at each perturbation level, and exports a success-vs-magnitude CSV and matplotlib degradation curve PNG,
So that I can quantify how policy robustness degrades as a function of perturbation magnitude and produce a publication-ready figure.

**Acceptance Criteria:**

**Given** a YAML sweep config specifying `param: gravity`, `range: [-12.0, -7.0]`, `steps: 10`, `n_episodes: 20`,
**When** `robot-lab sweep --config experiments/3_sim2real/configs/gravity_sweep.yaml` is run,
**Then** the policy is evaluated at 10 evenly-spaced gravity values, producing a results dict with `param_value`, `success_rate`, and `mean_reward` per step.

**Given** the sweep completes,
**When** results are exported,
**Then** a CSV file is written to `data/experiments/{experiment_name}/runs/{run_id}/perturbation_sweep.csv` with one row per perturbation level, and a PNG degradation curve is saved to the same directory showing success rate vs perturbation magnitude with axis labels and title.

**Given** a sweep run is executed,
**When** `tracker.end_run("COMPLETED")` fires,
**Then** all sweep results are persisted under `metadata.json["metrics"]["perturbation_sweep"]` for programmatic access via `ResultsDatabase`.

**Given** two sweep runs exist for the same environment — one with curriculum training, one without —
**When** `robot-lab plot --phase 3 --type sweep` is run,
**Then** both degradation curves are overlaid on the same figure with a legend identifying each variant.

**Given** `make test` is run after implementation,
**When** `test_results_db.py` exercises the sweep results schema,
**Then** the sweep CSV and metadata structure tests pass using a fixture — no live training required.

---

### Story 6.2: Gazebo Eval CLI with Conditional ROS 2 Import

As a researcher (Marco),
I want a `robot-lab gazebo-eval --model data/models/sac_a1quadruped_parallel.zip --config experiments/3_sim2real/configs/gazebo_eval.yaml` command that evaluates a trained policy in a Gazebo simulation with injected perturbations, producing success rate, energy, and recovery metrics,
So that I can quantify the MuJoCo-to-Gazebo performance gap without requiring real hardware, and the absence of ROS 2 never breaks the base package install.

**Acceptance Criteria:**

**Given** `import robot_lab` is executed on a machine without ROS 2 installed,
**When** the import completes,
**Then** no `ImportError` or `ModuleNotFoundError` is raised — the `rclpy` import is fully isolated behind a `try/except ImportError` guard in the `gazebo-eval` command module.

**Given** `robot-lab gazebo-eval --help` is run on a machine without ROS 2,
**When** the help text is displayed,
**Then** the command appears in the CLI help with a description and a note that ROS 2 is required at runtime; the command does not crash on `--help` alone.

**Given** `robot-lab gazebo-eval` is run on a machine without ROS 2,
**When** the command attempts to execute,
**Then** a clear error is raised: `"[GazeboEval] ROS 2 (rclpy) is not installed. Install ROS 2 Humble and source the workspace before running gazebo-eval."` — not a raw Python traceback.

**Given** `robot-lab gazebo-eval` is run on a machine with ROS 2 and a running Gazebo instance,
**When** the evaluation completes over `n_episodes` as specified in the YAML config,
**Then** the results dict contains `success_rate` (float), `mean_energy` (float), `mean_recovery_time` (float), and all injected perturbation parameters — written to `metadata.json["metrics"]["gazebo_eval"]` and printed as a summary table.

**Given** the Gazebo eval YAML config specifies `actuator_latency_ms: 20`, `contact_noise_std: 0.02`, `friction_multiplier: 0.8`,
**When** results are saved to `metadata.json`,
**Then** all three perturbation parameter values are recorded verbatim under `metadata.json["custom"]["gazebo_perturbations"]` — enabling full reproducibility from config alone.

