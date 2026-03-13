---
project_name: 'robot-lab'
user_name: 'Marco'
date: '2026-03-02'
sections_completed: ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'code_quality', 'workflow_rules', 'critical_rules']
status: 'complete'
rule_count: 47
optimized_for_llm: true
last_updated: '2026-03-05'
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

- **Python**: >=3.12
- **Package manager**: `uv` (NOT pip directly; use `uv run`, `uv sync`)
- **gymnasium**: >=0.28.0 with `[classic-control,mujoco]` extras
- **stable-baselines3**: >=2.0.0 (SAC & PPO are the only supported algorithms)
- **torch**: ==2.9.1+cu126 (LOCKED — see GPU constraint below)
- **torchvision**: ==0.24.1
- **numpy**: >=2.0.0
- **pydantic**: >=2.0.0 (v2 API — use `@field_validator`, `@model_validator`, NOT deprecated v1 `@validator` style)
- **typer**: >=0.9.0 (CLI framework)
- **rich**: >=13.0.0 (console output/tables)
- **loguru**: >=0.7.0 (all logging — do NOT use stdlib `logging`)
- **tensorboard**: >=2.20.0
- **pyyaml**: >=6.0.0
- **matplotlib**: >=3.7.0
- **pytest**: >=7.0.0 / **ruff**: >=0.1.0 (dev only)

### ⚠️ Critical GPU Constraint

- **GTX 1080 = Compute Capability 6.1 (sm_61)**
- **PyTorch 2.10+ dropped sm_61 support** — never bump torch beyond 2.9.x without verifying GPU compat
- PyTorch must always be installed from the CUDA 12.6 index (`https://download.pytorch.org/whl/cu126`) via `tool.uv.sources` in `pyproject.toml`
- To change PyTorch version: update `pyproject.toml` versions AND `tool.uv.index` URL, then run `uv sync --reinstall-package torch`

---

## Critical Implementation Rules

### Language-Specific Rules (Python)

- **Python 3.12+ features are available** — use `pathlib.Path` (never string paths), `typing` generics directly (e.g., `list[str]` not `List[str]`), but keep `from typing import ...` for complex types like `Optional`, `Dict`, `Tuple` for readability consistency with existing code
- **All imports use absolute paths** from `robot_lab.*` — no relative imports (except inside `__init__.py` re-exports)
- **Use `from pathlib import Path`** for all file paths — never use `os.path` string manipulation
- **Use `loguru`'s `logger`** imported as `from loguru import logger` — never use `print()` for diagnostic output, never use Python stdlib `logging`
- **Type hints are required** on all function signatures; return types must be annotated
- **Pydantic v2 API only**: use `@field_validator` with `@classmethod`, `@model_validator(mode='after')`, and `Field(...)` — the deprecated `@validator` from v1 must not be used in new code (existing schemas already mixing both; follow the v2 pattern in new code)
- **`importlib.resources.files()`** is used to access bundled package data files (e.g., JSON configs inside `robot_lab/configs/`) — never use `__file__`-relative paths for package resources

### Framework-Specific Rules

#### Stable-Baselines3

- **Only `SAC` and `PPO`** are supported algorithms; both are imported directly from `stable_baselines3`
- **`VecNormalize` is mandatory for SAC, optional for PPO**: SAC depends on normalized rewards and observations during training — the running statistics are baked into training; loading a SAC model without its matching `VecNormalize` `.pkl` will produce wrong/random behavior. PPO can train without `VecNormalize` but may benefit from it. Always save the `.pkl` alongside the `.zip` for SAC.
- **`SubprocVecEnv`** is used for multi-env training (real parallelism); **`DummyVecEnv`** is used for evaluation and visualization (single env, deterministic)
- **`SubprocVecEnv` caveat on Windows**: environments that use OpenGL/MuJoCo rendering cannot use `SubprocVecEnv` on Windows (Python `spawn` multiprocessing cannot share GL contexts) — use `DummyVecEnv` if the environment has rendering dependencies or if `SubprocVecEnv` raises GL/display errors
- **Always wrap each sub-environment with `Monitor`** before creating a `VecEnv` — `Monitor` is required for SB3 callbacks to track episode rewards
- Callbacks follow SB3's `BaseCallback` interface; custom callbacks live in `robot_lab/utils/callbacks.py`
- **`EvalCallback`** must use `best_model_save_path` to save the best model — the best model `.zip` file is what gets used for visualization

#### Gymnasium Environments

- **All environments must inherit from `gym.Env`** and implement `reset(*, seed=None, options=None)`, `step(action)` — use the v0.26+ API: `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`
- **Use `super().reset(seed=seed)` in every `reset()`** to initialize `self.np_random` properly
- **Seeding is the caller's responsibility** — there is no global seed utility; the `seed` parameter passed to `train()` sets NumPy, Python `random`, and the SB3 model seed for that run; each call site must explicitly pass a seed for reproducibility
- **Action and observation spaces must use `np.float32` dtype** in `spaces.Box` — `np.float64` will silently cause dtype mismatch issues with SB3
- **New environments must be registered via JSON** — add an entry to `robot_lab/configs/custom_envs.json` (do NOT modify Python registration code directly); the registry reads this file at runtime via `importlib.resources`
- **Always use `make_env()` from `robot_lab.envs`** as the factory for environment creation — never call `gym.make()` directly in training/experiment code; `make_env()` handles seeding and `Monitor` wrapping automatically
- Custom environments live in `robot_lab/envs/manipulation/` or `robot_lab/envs/locomotion/` and must be exported from the subpackage's `__init__.py` and re-exported from `robot_lab/envs/__init__.py`

#### Configuration / Hyperparameters

- **Hyperparameters are loaded via `load_hyperparameters(env_name, algorithm, custom_config_path)`** — this applies a 4-level fallback: custom path → `{env}_{algo}.json` → `default_{algo}.json` → `default.json`; if ALL fallbacks fail a `ValueError` is raised — always create at minimum a `default_{algo}.json` when adding support for a new algorithm
- **Config files are JSON**, stored in `robot_lab/configs/` and accessed as package resources
- **Environment name parsing**: the config lookup uses `env_name.split('-')[0].lower()` to derive the base name (e.g., `"Walker2d-v5"` → `"walker2d"`)
- **YAML is used for experiment runner configs** (`ExperimentRunner` loads YAML from `experiments/`), while JSON is used for hyperparameter configs

#### Experiment System

- **`ExperimentRunner`** orchestrates YAML-defined experiment campaigns from `experiments/`; instantiate with `ExperimentRunner(config_path, output_dir)`
- **Pydantic schemas** in `robot_lab/experiments/schemas.py` validate all experiment specs — always pass data through these schemas before using in experiments
- **`ExperimentTracker`** handles per-run metadata; use `tracker.start_run()` / `tracker.end_run()` pattern
- **Environment kwargs validation is non-strict by default** in `ExperimentRunner` — unsupported kwargs are filtered with warnings (not errors); pass `strict=True` only when debugging

#### Action Wrappers

- **`ActionRepeatWrapper`** (frameskip) and action filter wrappers live in `robot_lab/wrappers.py`
- Use `create_action_wrapper()` factory from `robot_lab.wrappers` — never instantiate wrappers directly in experiment code
- Wrappers must be applied **after** `Monitor` but **before** `VecNormalize`

### Testing Rules

- **Test files** must be named `test_*.py`, test classes `Test*`, test functions `test_*`
- **All tests live in `tests/`**; the test root is configured in `pytest.ini`
- **Markers**: use `@pytest.mark.slow` for tests >10s, `@pytest.mark.fast` for <5s, `@pytest.mark.smoke` for import/dependency checks — all markers are declared in `pytest.ini` and `--strict-markers` is active (undeclared markers cause errors)
- **Always use the `temp_output_dir` fixture** (from `conftest.py`) for any test that writes files — never write to `data/` or project root in tests
- **`test_seed` fixture** provides `42` — use this for all reproducibility-sensitive tests
- **Avoid importing ROS/system pytest plugins** — `pytest.ini` already disables them via `-p no:ament_*`; do NOT remove these exclusions
- **Run tests via Makefile** (`make test`, `make test-fast`, `make test-smoke`) or via `PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest` — do NOT use bare `pytest` on this system (ROS plugin conflicts)

### Code Quality & Style Rules

- **Line length**: 100 characters (configured in `[tool.ruff]`)
- **Ruff rules**: `E`, `F`, `I` (pycodestyle errors, pyflakes, isort) — run `make lint` before committing
- **Docstrings are required** on all public functions, classes, and modules — use Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- **Module-level docstrings** must describe what the module provides, not just restate the filename
- **Naming conventions**:
  - Files: `snake_case.py`
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Gym environment IDs: `PascalCase-vN` (e.g., `GripperEnv-v0`)
- **Run ID format**: `{YYYYMMDD_HHMMSS}_{8char_hash}_{suffix}` — generated by `generate_run_id()` in `robot_lab/utils/run_utils.py`; never construct run IDs manually

### Development Workflow Rules

- **Package manager**: always use `uv` — key commands:
  - `uv sync` — install all deps from lock file (use after pulling changes)
  - `uv add <package>` — add a new dependency (updates `pyproject.toml` and lock file)
  - `uv run python <script>` — run a script using the managed `.venv`
  - `uv run pytest ...` — run tests through the managed env
  - **Never manually create or activate the `.venv`** — `uv` manages it entirely
  - **Always run scripts from the workspace root** — `robot_lab/utils/paths.py` uses `Path.cwd()` to locate `data/`; running from a subdirectory will create a `data/` folder in the wrong location
- **Entry point**: CLI is exposed as `robot-lab` script via `[project.scripts]` in `pyproject.toml`, backed by `robot_lab.cli:app`
- **Build backend**: `hatchling` — do NOT change to setuptools or other backends
- **Data directory**: all runtime output (models, logs, tensorboard, experiments) goes under `data/` at workspace root by default; path helpers in `robot_lab/utils/paths.py` always `mkdir(parents=True, exist_ok=True)` — never pre-check directory existence
- **TensorBoard logs** land in `data/tensorboard/`; models in `data/models/`; per-run logs in `data/logs/<run_id>/`

### Critical Don't-Miss Rules

- **NEVER call `gym.make()` directly** in training or experiment code — always use `make_env()` from `robot_lab.envs`
- **NEVER load a SAC model without its paired `VecNormalize` `.pkl` file** — the normalization statistics are essential for correct inference; for PPO this is optional but recommended if VecNormalize was used during training
- **NEVER use `SubprocVecEnv` with environments that have rendering dependencies on Windows** — use `DummyVecEnv` instead to avoid GL/display multiprocessing errors
- **NEVER use `print()` for logging** — always use `from loguru import logger` and appropriate level methods (`logger.info`, `logger.warning`, `logger.success`, `logger.error`)
- **NEVER change the PyTorch version beyond 2.9.x** without explicit GPU compatibility verification for sm_61 (GTX 1080)
- **NEVER use relative imports** outside of `__init__.py` files
- **NEVER add new environments by modifying Python registration code** — add entries to `robot_lab/configs/custom_envs.json` and export the class from the subpackage `__init__.py`
- **NEVER use `os.path` string manipulation** for file paths — always use `pathlib.Path`
- **NEVER use numpy scalar types in observation/action spaces with `float64`** — explicitly set `dtype=np.float32` in all `spaces.Box` definitions
- **NEVER skip `super().reset(seed=seed)`** in custom environment `reset()` methods — this initializes `self.np_random` for reproducible randomness
- **NEVER write test output to `data/` or project root** — always use the `temp_output_dir` fixture
- **NEVER run bare `pytest`** on this system — use Makefile targets or the full invocation with plugin isolation flags to avoid ROS plugin conflicts
- **NEVER use `@validator` (Pydantic v1)** in new schema code — use `@field_validator` with `@classmethod` (Pydantic v2)

---

## Usage Guidelines

**For AI Agents:**

- Read this file before implementing any code in this project
- Follow ALL rules exactly as documented
- When in doubt, prefer the more restrictive option
- If a new pattern emerges during implementation, flag it for addition here

**For Humans:**

- Keep this file lean and focused on agent needs — remove rules that become obvious over time
- Update when technology stack changes (especially PyTorch version and GPU constraints)
- Review when adding new environments, algorithms, or experiment patterns
- Run `make lint` and `make test-smoke` after any structural changes to verify nothing is broken

_Last Updated: 2026-03-05_

