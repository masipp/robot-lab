# Story 1.3: Atomic Model + VecNorm Checkpointing

Status: review

## Story

As a researcher (Marco),
I want a `RobotLabCheckpointCallback` that always saves model `.zip` and VecNormalize `.pkl` as an
atomic pair, wired into every training run by default,
so that no checkpoint can become a corrupt artifact (a model without its matching VecNorm stats).

## Acceptance Criteria

1. **Given** `train()` is called by any caller (CLI or library),
   **When** the training callback list is assembled,
   **Then** `RobotLabCheckpointCallback` is present without the caller explicitly adding it.

2. **Given** `RobotLabCheckpointCallback._on_step()` fires at a checkpoint interval,
   **When** the model `.zip` is saved,
   **Then** the VecNorm `.pkl` save immediately follows within the same method — they are never
   written independently.

3. **Given** a training run completes normally,
   **When** I inspect the `models/best/` directory,
   **Then** both `best_model.zip` and `best_model_vecnorm.pkl` are present as a matched pair.

4. **Given** `make test` is run after implementation,
   **When** `test_callbacks.py` executes the checkpoint tests,
   **Then** the atomic-pair invariant is verified without running a full live training session.

## Tasks / Subtasks

- [x] Task 1: Implement `RobotLabCheckpointCallback` in `robot_lab/utils/callbacks.py` (AC: 1, 2)
  - [x] Time-based: saves every N seconds (default 600s / 10 min)
  - [x] `_save_atomic_pair()` — writes `.zip` then `.pkl` in the same method call
  - [x] If `training_env` has no `save` attribute (non-VecNormalized env), skip `.pkl` silently
  - [x] Filename pattern: `{name_prefix}_ckpt{N:04d}_step{timesteps}.zip` + `..._vecnorm.pkl`
  - [x] Replaces the two separate `TimeBasedCheckpointCallback` + `VecNormalizeSaveCallback`

- [x] Task 2: Implement `RobotLabEvalCallback` in `robot_lab/utils/callbacks.py` (AC: 3)
  - [x] Subclass `EvalCallback`
  - [x] After `super()._on_step()` detects a new best reward → also save `best_model_vecnorm.pkl`
  - [x] VecNorm saved to same directory as `best_model.zip` (`self.best_model_save_path`)

- [x] Task 3: Wire into `training.py` (AC: 1, 3)
  - [x] Replace `EvalCallback` → `RobotLabEvalCallback`
  - [x] Replace `TimeBasedCheckpointCallback` + `VecNormalizeSaveCallback` → `RobotLabCheckpointCallback`
  - [x] Remove `CheckpointCallback` (SB3 built-in) — no longer needed without atomic guarantee
  - [x] Update imports

- [x] Task 4: Create `tests/test_callbacks.py` (AC: 2, 3, 4)
  - [x] `test_atomic_pair_both_files_written` — trigger `_save_atomic_pair()` directly; assert
    both `.zip` and `_vecnorm.pkl` are written
  - [x] `test_atomic_pair_no_vecnorm_env_skips_pkl` — env without `save` attribute; assert no
    `.pkl` written, no exception raised
  - [x] `test_eval_callback_saves_best_vecnorm` — simulate improved reward; assert
    `best_model_vecnorm.pkl` appears alongside `best_model.zip`
  - [x] All tests `@pytest.mark.fast`, use `temp_output_dir`, use `unittest.mock`

- [x] Task 5: Ruff + tests (AC: 4)
  - [x] `uv tool run ruff check robot_lab/utils/callbacks.py tests/test_callbacks.py` — zero violations
  - [x] `pytest tests/test_callbacks.py -v` — all tests pass

## Dev Notes

### Key design constraint
`_save_atomic_pair()` must write `.zip` and `.pkl` in the **same method** — never in separate
callbacks. This is the whole point of the story: eliminating the race between two independent
callbacks that previously ran on separate time budgets.

### `RobotLabEvalCallback` best-model detection
`EvalCallback` stores `self.best_mean_reward` (float, starts at `-inf`). In `_on_step()`:
```python
old_best = self.best_mean_reward
result = super()._on_step()
if self.best_mean_reward > old_best:
    # new best — save VecNorm
```
This is safe because `EvalCallback._on_step()` only updates `best_mean_reward` when it saves.

### File naming
- Checkpoint: `{prefix}_ckpt0001_step12345.zip` + `{prefix}_ckpt0001_step12345_vecnorm.pkl`
- Best: `best_model.zip` + `best_model_vecnorm.pkl` (fixed names, EvalCallback convention)

### Backward compat
`TimeBasedCheckpointCallback` and `VecNormalizeSaveCallback` are kept in the file —
they may be used by user scripts in `experiments/` — but are no longer instantiated by `train()`.

## Dev Agent Record

### Implementation Notes
- `RobotLabCheckpointCallback` and `RobotLabEvalCallback` implemented in `robot_lab/utils/callbacks.py`
- Both old callbacks (`TimeBasedCheckpointCallback`, `VecNormalizeSaveCallback`) kept for backward compat
- `training.py` wired: `RobotLabEvalCallback` replaces `EvalCallback`, `RobotLabCheckpointCallback` replaces the two-callback pair
- `tests/test_callbacks.py`: 6 tests covering atomic-pair invariant, no-vecnorm skip, interval firing, and eval callback best-model detection
- `tests/test_yaml_tracking.py` deleted — tested the old YAML-based `ExperimentTracker` API that was intentionally removed in Story 1.2
- All 73 non-training tests pass; all 4 training tests pass (verified in prior run)
- Zero ruff violations

## File List
- `robot_lab/utils/callbacks.py` — added `RobotLabCheckpointCallback`, `RobotLabEvalCallback`
- `robot_lab/training.py` — wired new callbacks, removed old callback pair
- `tests/test_callbacks.py` — new test file (6 tests)
- `tests/test_yaml_tracking.py` — deleted (stale, tested removed API)
