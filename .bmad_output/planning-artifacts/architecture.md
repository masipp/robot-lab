---
stepsCompleted: ['step-01-init', 'step-02-context', 'step-03-starter', 'step-04-decisions', 'step-05-patterns', 'step-06-structure', 'step-07-validation', 'step-08-complete']
lastStep: 8
status: 'complete'
completedAt: '2026-03-13'
inputDocuments:
  - '.bmad_output/planning-artifacts/prd.md'
  - '.bmad_output/project-context.md'
workflowType: 'architecture'
project_name: 'robot-lab'
user_name: 'Marco'
date: '2026-03-12'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (9 confirmed + 2 stretch):**

| Category | FRs | Architecture Implication |
|---|---|---|
| Experiment execution | FR-001–FR-004 | CLI + library dual-mode API; ExperimentTracker lifecycle hooks |
| Extensibility & scaffolds | FR-005–FR-007 | Plugin/registry pattern; scaffolded base classes with intentionally empty hooks |
| Results analysis | FR-008–FR-009 | ResultsDatabase queryable schema; CSV export path through tracker |
| Stretch/Phase 4 | FR-S01–FR-S02 | Conditional ROS 2 import guard; perturbation sweep runner |

**Non-Functional Requirements driving architecture:**

| NFR | Architectural Implication |
|---|---|
| Training throughput (≤48h/experiment on GTX 1080) | SubprocVecEnv for training; VRAM-aware env/batch size defaults |
| CLI cold-start ≤3s | Lazy imports — torch and SB3 not imported at CLI entry point |
| Experiment fault tolerance (checkpoint every 50k steps) | CheckpointCallback wired by default; atomic model+VecNorm saves |
| Config immutability at run-start | Config copied to experiment dir before first step |
| No side effects on import | Package __init__.py is pure re-export; no I/O at import time |
| Ruff clean + type hints on all public functions | Enforced by CI (`make lint`, `make test`) |

**Scale & Complexity:**

- Primary domain: Scientific ML / RL research tooling (Python package)
- Complexity level: Medium (solo user, no multi-tenancy, but reproducibility and extensibility are hard constraints)
- Estimated architectural components: ~12 (CLI, training pipeline, config system, env registry, experiment tracker, results DB, experiment runner, plugin registry ×3, visualization pipeline, callback scaffold layer)

### Technical Constraints & Dependencies

- **PyTorch hard ceiling**: ≤2.9.x (sm_61/GTX 1080 dropped in 2.10+) — never bump without explicit GPU compat check
- **Windows rendering constraint** *(secondary — best-effort, can be broken if needed)*: `SubprocVecEnv` on visualization paths is discouraged due to GL context issues; `DummyVecEnv` preferred for viz, but not a hard architectural gate
- **Package manager**: `uv` exclusively — no bare pip, no conda
- **Gymnasium API**: v0.26+ (`reset()` → `(obs, info)`, `step()` → `(obs, reward, terminated, truncated, info)`)
- **SB3 algorithms**: SAC and PPO only — no other algorithms supported
- **Pydantic v2 API**: `@field_validator` + `@classmethod`; v1 `@validator` forbidden in new code
- **Config access**: `importlib.resources.files()` for package data — never `__file__`-relative paths
- **Import rule**: absolute imports from `robot_lab.*` everywhere except `__init__.py` re-exports
- **Gazebo/ROS 2**: must stay behind `try: import rclpy` guard — base package must install without ROS 2

### Cross-Cutting Concerns Identified

1. **Reproducibility** — seeding (NumPy, Python random, SB3 model), config immutability, git commit logging, VecNorm paired saves; affects training pipeline, tracker, and all custom environments
2. **Observability plugin/registry pattern** — MetricsPlugin, VisualizationPlugin, MetadataPlugin share a common registration/lifecycle pattern; must be architecturally consistent across all three
3. **YouAreLazy enforcement boundary** — base classes must provide scaffolded hooks with intentionally empty core methods; shapes all `CurriculumCallback`, `ActionFilterWrapper`, `GoalConditionedWrapper`, `DomainRandomizationWrapper` class designs
4. **Dual-mode API surface** — CLI commands are thin wrappers over library functions; every user-facing capability must be importable and callable programmatically
5. **Hardware-aware defaults** — VRAM budget (~8 GB), PyTorch version ceiling, and wall-clock targets must be reflected in default configs and experiment design guardrails
6. **Windows compatibility** *(secondary / best-effort)* — prefer `DummyVecEnv` on visualization paths and document Windows-specific workarounds, but this concern does not block architectural decisions or require dedicated solutions

## Starter Template Evaluation

### Primary Technology Domain

Python scientific ML / RL research tooling — brownfield package (Phase 0 infrastructure complete).
No scaffold initialization required; technology stack is already established and locked.

### Starter Options Considered

Not applicable — this is a brownfield project. Phase 0 infrastructure (`training.py`, `cli.py`,
`config.py`, `ExperimentTracker`, env registry, plugin/registry stubs) is already implemented.
The architecture document governs Phases 0.5–4 additions to this existing foundation.

### Existing Foundation (Phase 0 Complete)

**Initialization Command:**

```bash
uv sync  # installs all dependencies from lock file into managed .venv
```

**Architectural Decisions Already Established:**

**Language & Runtime:**
- Python ≥3.12; `pathlib.Path` for all paths; typing generics directly (`list[str]` etc.)
- Absolute imports from `robot_lab.*` everywhere (no relative imports outside `__init__.py`)

**Package Management:**
- `uv` exclusively — `uv sync`, `uv add`, `uv run`; `.venv` managed by uv, never manually activated

**Deep Learning Stack:**
- PyTorch 2.9.x+cu126 hard-pinned (sm_61/GTX 1080 support; drops in 2.10)
- SB3 ≥2.0 with SAC and PPO only; VecNormalize mandatory for SAC

**Configuration System:**
- JSON for hyperparams (via `importlib.resources.files()`, never `__file__`)
- YAML for experiment runner configs
- 4-level config fallback: custom → `{env}_{algo}.json` → `default_{algo}.json` → `default.json`

**CLI Framework:**
- Typer app at `robot_lab/cli.py`; exposed as `robot-lab` via `[project.scripts]`
- CLI commands are thin wrappers — all logic in library functions

**Testing Infrastructure:**
- pytest with `--strict-markers`; markers: `slow`, `fast`, `smoke`
- Run via Makefile (`make test`, `make test-fast`, `make test-smoke`) with ROS plugin isolation
- `temp_output_dir` fixture for all file-writing tests; never write to `data/` or project root

**Code Quality:**
- Ruff (E, F, I); 100-char line limit; Google-style docstrings; type hints on all public functions

**Logging:**
- Loguru everywhere (`from loguru import logger`); `print()` forbidden for diagnostic output

**Note:** No new project initialization is needed. All new Phases 0.5–4 work builds on this foundation.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
1. Plugin/Registry implementation structure
2. ResultsDatabase schema and query interface
3. Callback scaffold class hierarchy

**Important Decisions (Shape Architecture):**
4. Multi-seed runner execution model
5. Checkpoint + VecNorm atomic save pattern

**Deferred Decisions (Post-Phase 1):**
- Gazebo/ROS 2 harness design (Phase 4 stretch)
- State visitation logger architecture (Phase 4)

### Plugin/Registry Architecture

**Decision**: Three separate module-level singleton registries — `MetricsRegistry`, `VisualizationRegistry`, `MetadataRegistry` — each a class instance exported from `robot_lab/experiments/plugins/__init__.py`.

**Rationale**: The three plugin types have distinct lifecycles (`on_step`/`on_episode_end`/`on_eval` for metrics; `render(results)` for visualization; `collect(context)` for metadata). Unifying them into one registry couples three legitimately independent concerns and reduces testability.

**Registration API:**
```python
from robot_lab.experiments.plugins import (
    register_metric_plugin,
    register_visualization_plugin,
    register_metadata_plugin,
)
register_metric_plugin(ActionSmoothnessMetric())
register_visualization_plugin(CurriculumProgressPlot())
register_metadata_plugin(CurriculumStageMetadata())
```

**Plugin module location**: `robot_lab/experiments/plugins/` with `__init__.py` exporting the three registries and registration functions.

**Global vs run-scoped**: plugins registered in `plugins/__init__.py` are active globally; plugins registered in an experiment script are active for that run only (run-scoped context passed at invocation).

### Data Architecture (Experiment Storage)

**Decision**: Two-layer JSON storage — one `metadata.json` per run (nested dict, all run data) + JSONL append-only index for fast multi-run queries.

**Per-run artifact** (in `data/experiments/{experiment_name}/runs/{run_id}/`):
- `metadata.json` — single file, all run data as a nested dict:
  ```json
  {
    "run": { "run_id": "...", "experiment": "...", "seed": 42, "phase": 1, "status": "COMPLETED", "started_at": "...", "finished_at": "..." },
    "config": { /* full hyperparameter config snapshot — immutable copy taken at run start */ },
    "system": { "python_version": "...", "gpu": "...", "cuda_version": "...", "git_commit": "..." },
    "metrics": { /* episode rewards, eval scores, custom plugin outputs */ },
    "custom": { /* MetadataPlugin outputs merged here */ }
  }
  ```

**Rationale**: One file eliminates semantic fragmentation across `metadata.json` / `hyperparameters.json` / `metrics.json` / `system_info.json`. A nested dict is still human-readable and git-friendly, while making partial loading via key access trivial. Plugin-contributed metadata lands in `"custom"` without polluting core keys.

**Index file** (`data/experiments/results_index.jsonl`): one JSON line per run containing lightweight summary fields (run_id, experiment, seed, phase, final_reward, status, timestamp). `ResultsDatabase` streams this for `--phase N` queries; full `metadata.json` loaded on demand.

**Config immutability**: full config snapshot written into `metadata.json["config"]` before the first training step; modifications to source config after this point have no effect on the run.

### API & Communication Patterns (Internal Module Interface)

**Decision**: CLI as thin wrapper over importable library functions; all public functions have stable signatures within a phase.

**Public API surface** (must not break within a phase):
| Module | Public Functions |
|---|---|
| `robot_lab.training` | `train()` |
| `robot_lab.visualization` | `visualize()` |
| `robot_lab.config` | `load_hyperparameters()` |
| `robot_lab.experiments` | `ExperimentTracker`, `ResultsDatabase`, `ExperimentRunner`, `get_template()` |
| `robot_lab.envs` | `make_env()`, `register_custom_envs()` |
| `robot_lab.experiments.plugins` | `register_metric_plugin()`, `register_visualization_plugin()`, `register_metadata_plugin()` |

**Error handling standard**: actionable error messages with `✘` prefix; warnings with `⚠`; success with `✔`; all via loguru — no `print()`.

### Callback Scaffold Class Hierarchy

**Decision**: Thin `RobotLabCallbackMixin` for shared ExperimentTracker + TensorBoard wiring; core hook methods intentionally empty (YouAreLazy boundary).

**Hierarchy:**
```
BaseCallback (SB3)
  └── RobotLabCallbackMixin  (adds tracker/TB wiring only)
        └── CurriculumCallback
              _should_advance() → intentionally empty (Marco implements)

gym.Wrapper
  └── RobotLabWrapperMixin  (adds tracker param logging only)
        ├── ActionFilterWrapper
        │     _apply_filter(action) → intentionally empty
        ├── GoalConditionedWrapper
        │     _sample_goal() → intentionally empty
        └── DomainRandomizationWrapper
              _sample_params() → intentionally empty
```

**Location**: `robot_lab/utils/callbacks.py` for SB3 callbacks; `robot_lab/wrappers.py` for Gym wrappers.

**Mixin responsibility**: log stage transitions to `ExperimentTracker` + TensorBoard scalar; nothing else. The mixin must NOT implement any RL logic.

### Infrastructure & Artifact Management

**Decision**: `RobotLabCheckpointCallback` subclass that overrides `_on_step` to always save model `.zip` + VecNormalize `.pkl` as an atomic pair.

**Rationale**: The invariant "model checkpoint without VecNorm stats is a corrupt artifact" must be enforced structurally, not by convention. A custom callback is the only way to guarantee this.

**Multi-seed runner**: Sequential by default (`parallel_seeds: false` in YAML config); parallel execution available as opt-in with explicit VRAM warning. GTX 1080 (8 GB) cannot reliably run two concurrent MuJoCo training processes.

### Decision Impact Analysis

**Implementation Sequence:**
1. Plugin registry module (`robot_lab/experiments/plugins/`) — required by all Phase 0.5+ work
2. `metadata.json` schema + JSONL index + `ResultsDatabase` query interface — required for FR-008, FR-009
3. `RobotLabCallbackMixin` + `CurriculumCallback` scaffold — required for Phase 1
4. `RobotLabCheckpointCallback` + atomic save — required for fault tolerance NFR
5. Scaffolded wrappers (`ActionFilterWrapper`, etc.) — phased with research phases

**Cross-Component Dependencies:**
- `ExperimentTracker` writes to `metadata.json` (nested dict) and appends to `results_index.jsonl` at run end
- `train()` in `training.py` must wire `RobotLabCheckpointCallback` by default
- `make_env()` in `robot_lab/envs` is the single env factory — all wrappers applied here, after `Monitor`, before `VecNormalize`
- `MetadataPlugin` outputs merged into `metadata.json["custom"]` at run end by the tracker

## Implementation Patterns & Consistency Rules

### Critical Conflict Points Identified

10 areas where AI agents could make different choices — all resolved below.

### Naming Patterns

**Python Naming:**
- All files: `snake_case.py`
- All classes: `PascalCase`
- All functions and variables: `snake_case`
- All constants: `UPPER_SNAKE_CASE`
- Gym environment IDs: `PascalCase-vN` (e.g., `GripperEnv-v0`, `A1Quadruped-v0`)
- Run IDs: `{YYYYMMDD_HHMMSS}_{8char_hash}_{suffix}` — generated by `generate_run_id()` only, never constructed manually

**`metadata.json` Key Naming:**
- All keys at all nesting levels: `snake_case`
- Top-level sections: `run`, `config`, `system`, `metrics`, `custom`
- Plugin-contributed keys land under `custom` only — no plugin may write to `run`, `config`, `system`, or `metrics` directly

**TensorBoard Scalar Key Format:**
- Pattern: `{category}/{metric_name}` with forward slash separator
- Examples: `curriculum/stage`, `smoothness/action_delta_norm`, `eval/mean_reward`
- Never: `curriculum_stage`, `smoothnessDeltaNorm` (no camelCase, no underscored namespacing)

**Configuration File Naming:**
- Hyperparameter configs: `{env_base}_{algo}.json` where `env_base` = `env_name.split('-')[0].lower()`
- Experiment runner configs: `{experiment_shortname}.yaml` in `experiments/{phase_folder}/configs/`

### Structure Patterns

**Module Placement Rules:**
| What | Where |
|---|---|
| SB3 custom callbacks | `robot_lab/utils/callbacks.py` |
| Gym wrappers | `robot_lab/wrappers.py` |
| Custom environments | `robot_lab/envs/manipulation/` or `robot_lab/envs/locomotion/` |
| Plugin implementations | `robot_lab/experiments/plugins/` |
| Path helpers | `robot_lab/utils/paths.py` — always use these, never `Path("data/...")` directly |
| Experiment runner YAML configs | `experiments/{phase_folder}/configs/` |
| Experiment result data | `data/experiments/{experiment_name}/runs/{run_id}/` |

**Test File Placement:**
- All test files in `tests/` — never co-located with source
- Naming: `test_{module_name}.py`
- All file-writing tests use `temp_output_dir` fixture — never write to `data/` or project root

### Format Patterns

**Environment Wrapper Chaining Order (MANDATORY):**
```
gym.make(env_id)          # raw environment
  → Monitor(env)          # SB3 episode tracking — MUST be first wrapper
  → [RobotLabWrapper]     # ActionFilter, GoalConditioned, DomainRandomization — after Monitor
  → VecEnv([...])         # SubprocVecEnv (training) or DummyVecEnv (eval/viz)
  → VecNormalize(venv)    # ALWAYS last — applied to the VecEnv, not individual envs
```
Never deviate from this order. Applying `VecNormalize` before `Monitor`, or wrapping after `VecEnv`, breaks SB3 callback statistics.

**Atomic Model Save Pattern:**
```python
# ALWAYS save model and VecNorm together — never one without the other (SAC)
model.save(model_path)
vec_norm.save(vecnorm_path)  # must immediately follow model.save()
```

**`train()` Return Signature:**
```python
def train(...) -> tuple[Any, Path, Path]:
    """Returns (model, model_path, vecnorm_path)."""
```
Always a 3-tuple — CLI, experiment runner, and tests all expect this shape.

**YAML Loading:**
- Always `yaml.safe_load()` — never bare `yaml.load()` (arbitrary code execution risk)

### Communication Patterns

**ExperimentTracker Write Protocol:**
- `tracker.start_run()` — called once, before first training step; writes initial `metadata.json` with `status: RUNNING`
- `tracker.update(section, data)` — merges data into the named top-level section (`metrics`, `custom`)
- `tracker.end_run(status)` — called exactly once at run termination; writes final `metadata.json` and appends to `results_index.jsonl`
- Never write directly to `metadata.json` — always go through the tracker API
- Status values: `RUNNING`, `COMPLETED`, `INTERRUPTED`, `FAILED` — no other values

**ExperimentTracker Status Transitions:**
```
start_run() → RUNNING
  → normal completion   → COMPLETED
  → KeyboardInterrupt   → INTERRUPTED
  → unhandled exception → FAILED
```
The `try/except/finally` pattern in `train()` must cover all three exit paths.

**Plugin Registration:**
- Plugins must NOT register themselves on import — registration is always an explicit `register_*()` call
- Global plugins: registered via `_register_defaults()` called lazily on first registry access
- Run-scoped plugins: registered in the experiment script, before `train()` is called

**Pydantic Model API (v2 only):**
- Serialization: `.model_dump()` (not `.dict()`)
- Copying: `.model_copy()` (not `.copy()`)
- Validation: `Model.model_validate(data)` (not `Model.parse_obj(data)`)
- Validators: `@field_validator` with `@classmethod` (not `@validator`)

### Process Patterns

**Error Handling:**
- Actionable CLI errors: raise `typer.BadParameter` or `typer.Exit(code=1)` with a clear message
- Library errors: raise `ValueError` with format `"[Module] <what went wrong>. <what to do>."` — e.g., `"[Config] No config found for 'walker2d_ppo'. Create robot_lab/configs/walker2d_ppo.json."`
- Log levels: `logger.info` (progress), `logger.warning` (`⚠` recoverable), `logger.error` (`✘` failed), `logger.success` (`✔` completed)
- Never `except Exception: pass` — always log and re-raise or handle explicitly

**Seeding Protocol:**
```python
import random, numpy as np
random.seed(seed)
np.random.seed(seed)
model = SAC(..., seed=seed)  # SB3 model seed
# gym env seed passed through make_env(seed=seed) → env.reset(seed=seed)
```
All three must be set at the start of every training run.

**`make_env()` Usage:**
- Always use `make_env()` from `robot_lab.envs` — never `gym.make()` directly in training, evaluation, or experiment code
- `make_env()` is the single point of Monitor wrapping and seeding — bypassing it breaks both

### Enforcement Guidelines

**All AI Agents MUST:**
- Use `make_env()` — never `gym.make()` directly
- Use `yaml.safe_load()` — never `yaml.load()`
- Use Pydantic v2 API (`.model_dump()`, `@field_validator`) — never v1 aliases
- Follow wrapper chaining order: `Monitor → [RobotLab wrappers] → VecEnv → VecNormalize`
- Save model + VecNorm atomically as a pair for SAC
- Go through `ExperimentTracker` API for all `metadata.json` writes
- Use `generate_run_id()` for run IDs — never construct them manually
- Register plugins explicitly — never as import side effects
- Use `snake_case` for all `metadata.json` keys
- Use `{category}/{metric}` for all TensorBoard scalar keys

**Anti-Patterns (Never Do):**
- `gym.make()` in training/experiment code
- `yaml.load()` without `Loader=yaml.SafeLoader`
- Writing `metadata.json` directly (bypass tracker)
- `SubprocVecEnv` for a single env — use `DummyVecEnv` for eval/viz
- Saving model without immediately saving paired VecNorm `.pkl` (SAC)
- Plugin `__init__.py` with registration calls at module level (side effects on import)
- `Path("data/models/...")` hardcoded — always use path helpers from `robot_lab.utils.paths`

## Project Structure & Boundaries

### Complete Project Directory Structure

Annotated with `[exists]` for Phase 0 files already present and `[new]` for items to be added in Phases 0.5–4.

```
robot-lab/
├── pyproject.toml                          [exists] uv/hatchling build config; PyTorch hard-pinned here
├── Makefile                                [exists] make test / make test-fast / make test-smoke / make lint
├── pytest.ini                              [exists] strict-markers, ROS plugin exclusions
├── README.md                               [exists]
├── AGENTS.md                               [exists] AI agent instructions (YouAreLazy protocol, groundrules)
│
├── robot_lab/                              [exists] installable package
│   ├── __init__.py                         [exists] pure re-exports; no I/O on import
│   ├── cli.py                              [exists] Typer app; thin wrappers over library functions
│   │                                                new commands: `results`, `gazebo-eval` [new Phase 3/4]
│   ├── training.py                         [exists] train() — wire RobotLabCheckpointCallback by default [new]
│   ├── visualization.py                    [exists] visualize() — DummyVecEnv only
│   ├── config.py                           [exists] load_hyperparameters(); 4-level fallback
│   ├── wrappers.py                         [exists] ActionRepeatWrapper + create_action_wrapper()
│   │                                                add: RobotLabWrapperMixin, ActionFilterWrapper,
│   │                                                     GoalConditionedWrapper, DomainRandomizationWrapper [new]
│   │
│   ├── configs/                            [exists] bundled JSON hyperparameter configs
│   │   ├── default.json
│   │   ├── default_sac.json
│   │   ├── default_ppo.json
│   │   ├── {env}_{algo}.json               [exists/new] one per env+algo combo
│   │   ├── builtin_envs.json
│   │   └── custom_envs.json               [exists] env registry (add entries here, not Python code)
│   │
│   ├── envs/                               [exists] environment layer
│   │   ├── __init__.py                     [exists] make_env(), register_custom_envs() re-exports
│   │   ├── registry.py                     [exists] JSON-driven env registration
│   │   ├── locomotion/
│   │   │   ├── __init__.py
│   │   │   ├── quadruped.py               [exists] A1Quadruped-v0 skeleton (Marco fills reward/obs)
│   │   │   └── {future_env}.py            [new as needed] additional locomotion envs added here
│   │   └── manipulation/
│   │       ├── __init__.py
│   │       ├── gripper.py                 [exists] GripperEnv-v0 skeleton (Marco fills reward/obs)
│   │       └── {future_env}.py            [new as needed] additional manipulation envs added here
│   │
│   ├── experiments/                        [exists] experiment orchestration layer
│   │   ├── __init__.py                     [exists] public re-exports
│   │   ├── tracker.py                      [exists] ExperimentTracker — refactor for metadata.json schema [new]
│   │   ├── results_db.py                   [exists] ResultsDatabase — add JSONL index + query API [new]
│   │   ├── runner.py                       [exists] ExperimentRunner (YAML-driven, multi-seed sequential)
│   │   ├── schemas.py                      [exists] Pydantic v2 experiment specs
│   │   ├── spec_templates.py               [exists] get_template()
│   │   ├── ai_planner.py                   [exists] stub — LLM integration (future)
│   │   └── plugins/                        [new] plugin/registry layer
│   │       ├── __init__.py                 [new] MetricsRegistry, VisualizationRegistry,
│   │       │                                      MetadataRegistry singletons + register_*() functions
│   │       ├── base.py                     [new] MetricsPlugin, VisualizationPlugin, MetadataPlugin ABCs
│   │       ├── defaults.py                 [new] built-in plugins (basic reward logging, system metadata)
│   │       └── contrib/                    [new] Marco's phase-specific plugins
│   │           └── __init__.py
│   │
│   └── utils/                              [exists] shared utilities
│       ├── __init__.py
│       ├── callbacks.py                    [exists] add: RobotLabCallbackMixin, CurriculumCallback [new]
│       │                                            add: RobotLabCheckpointCallback [new]
│       ├── paths.py                        [exists] get_models_dir(), get_logs_dir(), get_experiments_dir()
│       ├── run_utils.py                    [exists] generate_run_id()
│       ├── logger.py                       [exists] loguru setup
│       ├── metadata.py                     [exists] system info collection — integrate with tracker [new]
│       ├── smoothness_metrics.py           [exists] action smoothness helpers
│       ├── mujoco_config.py               [exists]
│       ├── debug_config.py                [exists]
│       └── run_selector.py                [exists]
│
├── experiments/                            [exists] experiment docs + YAML runner configs
│   ├── README.md
│   ├── 0_foundations/
│   │   ├── 001_smooth_locomotion.md       [exists]
│   │   └── configs/                       [exists] YAML runner configs for Phase 0 experiments
│   ├── 1_curriculum/                      [new]
│   │   ├── 010_manual_curriculum.md       [new]
│   │   ├── 011_adaptive_advancement.md    [new]
│   │   └── configs/
│   ├── 2_representation/                  [new]
│   ├── 3_sim2real/                        [new]
│   └── 4_scientific/                      [new]
│
├── data/                                   [exists] all runtime output (gitignored)
│   ├── experiments/
│   │   ├── results_index.jsonl            [new] append-only multi-run query index
│   │   └── {experiment_name}/
│   │       └── runs/
│   │           └── {run_id}/
│   │               └── metadata.json      [new schema] single nested-dict (run/config/system/metrics/custom)
│   ├── models/                            [exists]
│   ├── logs/                              [exists]
│   └── tensorboard/                       [exists]
│
├── tests/                                  [exists]
│   ├── __init__.py
│   ├── conftest.py                         [exists] temp_output_dir, test_seed fixtures
│   ├── test_smoke.py                       [exists]
│   ├── test_training.py                    [exists]
│   ├── test_env_registry.py               [exists]
│   ├── test_yaml_tracking.py              [exists]
│   ├── test_plugins.py                    [new] plugin registry unit tests
│   ├── test_tracker.py                    [new] ExperimentTracker + metadata.json schema tests
│   ├── test_results_db.py                 [new] JSONL index + ResultsDatabase query tests
│   └── test_callbacks.py                  [new] CurriculumCallback scaffold + mixin tests
│
└── docs/
    ├── general/
    │   ├── experiment_schema.md            [exists] update for new metadata.json structure [new]
    │   ├── metadata_system.md             [exists] update for plugin/registry pattern [new]
    │   ├── environment_registry.md         [exists]
    │   └── adding_environments.md         [exists]
    ├── setup/
    │   └── GPU_SETUP.md                   [exists]
    └── user/
        ├── PLAN.md                        [exists]
        ├── TODO.md                        [exists]
        └── USER_SKILL.md                  [exists]
```

### Architectural Boundaries

**Package Boundary (`robot_lab/`):**
- Everything inside is importable as a library; no I/O, training, or network calls at import time
- CLI (`cli.py`) is the only entry point for user-facing commands; all logic delegated to modules
- `__init__.py` at package root re-exports public API only — nothing instantiated

**Plugin Boundary (`robot_lab/experiments/plugins/`):**
- Plugins cross no other internal module boundaries at definition time — they only reference the plugin base classes
- Registries are module-level singletons in `plugins/__init__.py`; no global state elsewhere
- `contrib/` is the designated location for Marco's phase-specific plugin implementations — kept separate from built-in defaults

**Environment Boundary (`robot_lab/envs/`):**
- `make_env()` is the only function external code may call to create environments
- Env registration is data-driven via `custom_envs.json` — Python registration code is not modified directly
- `locomotion/` and `manipulation/` are open-ended directories — multiple environments per folder is expected and correct; each env is its own module file (e.g., `gripper.py`, `push_block.py`)
- Custom env logic (reward, obs, success criterion) lives exclusively in `locomotion/` or `manipulation/` submodules
- Each new env module must be exported from the subpackage `__init__.py` and registered in `custom_envs.json`

**Experiment Data Boundary (`data/experiments/`):**
- `ExperimentTracker` is the only writer to `metadata.json` and `results_index.jsonl`
- `ResultsDatabase` is the only reader for cross-run queries — no direct file scanning by other modules
- Model artifacts (`.zip`, `.pkl`) live in `data/models/` — separate from metadata

### Requirements to Structure Mapping

| FR | Location |
|---|---|
| FR-001 one-command launch | `cli.py` `train` command → `training.py` `train()` |
| FR-002 automatic metadata | `tracker.py` `start_run()`/`end_run()` + `metadata.py` system info |
| FR-003 multi-seed + variance plots | `runner.py` `ExperimentRunner` + visualization plugin |
| FR-004 markdown summary template | `experiments/{phase}/` template, populated at run end |
| FR-005 base class hooks | `callbacks.py` (`CurriculumCallback`), `wrappers.py` (all scaffold wrappers) |
| FR-006 TensorBoard curriculum metrics | `RobotLabCallbackMixin` in `callbacks.py` |
| FR-007 unit test harness | `tests/test_callbacks.py` with mock env context |
| FR-008 results query CLI | `cli.py` `results` command → `results_db.py` `ResultsDatabase.query()` |
| FR-009 ResultsDatabase query | `results_db.py` — JSONL index scan + full metadata.json load on demand |
| FR-S01 Gazebo eval | `cli.py` `gazebo-eval` (Phase 4); `try: import rclpy` guard |
| FR-S02 perturbation sweep | `runner.py` sweep extension + visualization plugin |

### Integration Points

**Training Pipeline Data Flow:**
```
cli.py train
  → training.py train()
      → config.py load_hyperparameters()
      → envs/__init__.py make_env()          # Monitor + wrappers applied here
      → SubprocVecEnv / DummyVecEnv
      → VecNormalize
      → SAC/PPO.learn()
          ⇕ RobotLabCallbackMixin             # logs to tracker + TensorBoard
          ⇕ RobotLabCheckpointCallback        # atomic model+VecNorm saves
          ⇕ MetricsPlugin registry            # custom metric collection
      → tracker.end_run(COMPLETED)
          → writes metadata.json
          → appends results_index.jsonl
          → MetadataPlugin registry merges into metadata["custom"]
```

**Results Query Data Flow:**
```
cli.py results --phase N
  → ResultsDatabase.query(phase=N)
      → scan results_index.jsonl             # fast, lightweight
      → load metadata.json on demand         # full data for matched runs
      → CSV export / table display
```

**Plugin Lifecycle Integration:**
```
plugins/__init__.py loaded
  → _register_defaults() called lazily       # built-in plugins registered on first access
  → experiment script calls register_*()     # run-scoped plugins registered
  → train() wires registries into callbacks
      → MetricsRegistry.on_step(context)     # each training step
      → MetricsRegistry.on_episode_end()     # each episode
      → MetricsRegistry.on_eval()            # each eval interval
      → VisualizationRegistry.render()       # post-run
      → MetadataRegistry.collect()           # at run end → metadata["custom"]
```

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:** All technology choices are mutually compatible. Plugin/registry singletons
have distinct, non-overlapping lifecycles. `RobotLabCallbackMixin` is fully compatible with SB3's
callback API. Sequential multi-seed execution is consistent with GTX 1080 VRAM constraints. Single
`metadata.json` with nested dict is consistent with JSON-only storage and git-friendly requirements.

**Pattern Consistency:** `snake_case` key naming, `{category}/{metric}` TensorBoard format, and
wrapper chaining order are all internally consistent and aligned with existing project conventions.

**Brownfield Migration Note:** `ExperimentTracker` currently uses a multi-file schema. The
consolidated `metadata.json` decision is a breaking change to `tracker.py` and `results_db.py`.
Implementation stories for these modules must be sequenced before Phase 0.5 experiment runs.

### Requirements Coverage Validation ✅

All 9 confirmed FRs and 2 stretch FRs are architecturally supported. All performance, reliability,
and maintainability NFRs are addressed. See Requirements to Structure Mapping table in Project
Structure section for per-FR traceability.

**FR-004 Clarification (pre-filled markdown summary):** Template written to
`data/experiments/{experiment_name}/runs/{run_id}/experiment_summary.md` by `tracker.end_run()`.
Template populated with run metadata from `metadata.json["run"]` and `metadata.json["config"]`.

### Implementation Readiness Validation ✅

- All critical architectural decisions documented with rationale
- 10 agent conflict points fully resolved with explicit, enforceable patterns
- Mandatory wrapper chaining order specified
- Three-exit-path `try/except/finally` pattern for `train()` specified
- Plugin registration side-effect prohibition explicitly documented
- Complete FR → file mapping provided
- Data flow diagrams for training pipeline, results query, and plugin lifecycle

### Gap Analysis Results

**Critical Gaps:** None identified.

**Important Gaps Resolved:**
- FR-004 template output path explicitly specified (above)
- ExperimentTracker migration sequencing documented as brownfield note
- `locomotion/` and `manipulation/` clarified as open-ended multi-env directories

**Nice-to-Have (deferred, not blocking):**
- Gazebo harness detailed design (Phase 4 — deferred by design)
- State visitation logger detailed architecture (Phase 4 — deferred by design)
- `results_index.jsonl` compaction strategy for very large result sets (not needed at current scale)

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context thoroughly analyzed (47 project context rules incorporated)
- [x] Scale and complexity assessed (Medium — solo researcher, GTX 1080, O(100)–O(1000) runs lifetime)
- [x] Technical constraints identified (PyTorch ceiling, VRAM budget, uv-only, no side-effects on import)
- [x] Cross-cutting concerns mapped (reproducibility, plugin/registry, YouAreLazy boundary, dual-mode API)

**✅ Architectural Decisions**
- [x] Plugin/Registry: three separate singletons with explicit registration API
- [x] Data storage: single `metadata.json` (nested dict) + `results_index.jsonl` index
- [x] Callback hierarchy: `RobotLabCallbackMixin` + intentionally-empty scaffold methods
- [x] Multi-seed runner: sequential default, parallel opt-in with VRAM warning
- [x] Checkpoint saves: `RobotLabCheckpointCallback` enforces atomic model+VecNorm pairs

**✅ Implementation Patterns**
- [x] Naming conventions: files, classes, functions, constants, env IDs, run IDs, JSON keys, TensorBoard keys
- [x] Structure patterns: module placement table, test file placement
- [x] Format patterns: wrapper chaining order (mandatory), atomic save, `train()` return signature, YAML loading
- [x] Communication patterns: tracker write protocol, status transitions, plugin registration, Pydantic v2 API
- [x] Process patterns: error handling format, seeding protocol, `make_env()` usage rule

**✅ Project Structure**
- [x] Complete annotated directory tree (`[exists]` vs `[new]`)
- [x] Component boundaries (package, plugin, environment, data)
- [x] `locomotion/` and `manipulation/` explicitly open-ended (multiple envs expected, each a separate module file)
- [x] Integration points (training pipeline, results query, plugin lifecycle — all with data flow)
- [x] FR → file mapping table

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence Level:** High

**Key Strengths:**
- Plugin/registry pattern directly matches PRD’s “foundational architectural principle” — zero ambiguity
- Single `metadata.json` eliminates semantic fragmentation and simplifies tracker implementation
- YouAreLazy boundary encoded structurally (empty abstract methods) — not just by convention
- Brownfield migration path clearly flagged — no surprise refactors for implementing agents
- Open-ended env directories support multi-environment experiment design without structural changes

**Implementation Priority Order:**
1. `robot_lab/experiments/plugins/` — plugin registry layer (unblocks all Phase 0.5+ observability)
2. `tracker.py` + `results_db.py` refactor — `metadata.json` schema + JSONL index (unblocks FR-002, FR-008, FR-009)
3. `RobotLabCheckpointCallback` in `callbacks.py` — atomic save enforcement (unblocks fault tolerance NFR)
4. `RobotLabCallbackMixin` + `CurriculumCallback` scaffold (unblocks Phase 1)
5. Scaffolded wrappers in `wrappers.py` (phased with research phases)
