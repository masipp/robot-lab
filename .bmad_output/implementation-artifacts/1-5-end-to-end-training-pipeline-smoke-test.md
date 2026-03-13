# Story 1.5: End-to-End Training Pipeline Smoke-Test

Status: review

## Story

As a researcher (Marco),
I want `robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42` to execute a complete
tracked run with the new pipeline fully wired (plugin registry, `ExperimentTracker`,
`RobotLabCheckpointCallback`, atomic VecNorm saves, JSONL index),
so that I can confirm all Epic 1 components work together and the one-command experiment launch
is fully functional.

## Acceptance Criteria

1. **Given** `robot-lab train --env MountainCarContinuous-v0 --algo SAC --seed 42` is run,
   **When** the run completes,
   **Then** the run directory contains: `metadata.json` (nested schema with all five sections),
   `experiment_summary.md`, at least one checkpoint `.zip`+`.pkl` pair (best model), and
   `results_index.jsonl` is updated.

2. **Given** `train()` is called with `seed=42`,
   **When** training begins,
   **Then** `random.seed(42)`, `np.random.seed(42)`, and `model.seed=42` are all set before
   the first training step.

3. **Given** `make test` is run after Epic 1 is complete,
   **When** all tests execute,
   **Then** `test_smoke.py`, `test_training.py`, `test_tracker.py`, `test_plugins.py`, and
   `test_callbacks.py` all pass with zero failures.

## Tasks / Subtasks

- [x] Task 1: Add global seed setting to `train()` in `robot_lab/training.py` (AC: 2)
  - [x] `import random` and `import numpy as np` at top of file
  - [x] `random.seed(seed)` and `np.random.seed(seed)` called immediately before `model.learn()`

- [x] Task 2: Add `TestEpic1Pipeline` class to `tests/test_training.py` (AC: 1, 3)
  - [x] `test_full_pipeline_artifacts` — runs short training (10k steps, eval_freq=5000),
    then asserts:
    - ExperimentTracker `metadata.json` exists with all five sections (`run`, `config`,
      `system`, `metrics`, `custom`)
    - `metadata.json["run"]["status"] == "COMPLETED"`
    - `experiment_summary.md` is present in the same tracker run dir
    - `results_index.jsonl` exists and has at least 1 valid JSON line with required keys
    - `best_model.zip` + `best_model_vecnorm.pkl` exist in `models/best/`
  - [x] `@pytest.mark.slow` (requires real training)

- [x] Task 3: Ruff + full tests (AC: 3)
  - [x] `uv tool run ruff check robot_lab/training.py tests/test_training.py` — zero violations
  - [x] `pytest tests/test_training.py::TestEpic1Pipeline -v` — passes in 63s

## Dev Notes

### Seed strategy
`random.seed()` and `numpy.random.seed()` affect global Python/NumPy RNG state — setting them
before `model.learn()` covers any SB3 internals that pull from these pools.
The `model.seed=seed` is already set via `SAC/PPO(seed=seed)` in the constructor.

### Finding the ExperimentTracker run dir
The tracker writes to `{output_dir}/experiments/{exp_name}/runs/{run_id}/`.
In tests, glob with `(output_dir / "experiments").glob("*/runs/*")` to find it dynamically
(run_id is timestamp-based and not predictable in tests).

### eval_freq / best model
With `eval_freq=5000` and `total_timesteps=10000`, the eval callback fires at step 5000.
`best_mean_reward` starts at `-inf`, so the first evaluation always writes `best_model.zip`
+ `best_model_vecnorm.pkl` through `RobotLabEvalCallback`.

## Dev Agent Record

### Implementation
- `import random`, `import numpy as np` added to `robot_lab/training.py`
- `random.seed(seed)` and `np.random.seed(seed)` called before `model.learn()` in `train()`
- `TestEpic1Pipeline.test_full_pipeline_artifacts` added to `tests/test_training.py`
- Integration test runs 10k-step SAC training; verifies all 5 sections in metadata.json,
  experiment_summary.md, results_index.jsonl, best_model.zip + best_model_vecnorm.pkl
- Test passes in ~63s; 77/77 fast tests green; zero ruff violations

## File List
- `robot_lab/training.py` — added `random`/`numpy` imports + global seed calls
- `tests/test_training.py` — added `TestEpic1Pipeline` class
