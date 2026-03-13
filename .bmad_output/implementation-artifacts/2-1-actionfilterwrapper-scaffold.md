# Story 2.1: ActionFilterWrapper Scaffold

Status: review

## Story

As a researcher (Marco),
I want an `ActionFilterWrapper` base class in `robot_lab/wrappers.py` with `RobotLabWrapperMixin` providing tracker parameter logging, and an intentionally empty `_apply_filter(action)` method,
So that I have a Gym wrapper skeleton where I can implement any action filtering strategy (low-pass, EMA, splines) without modifying any infrastructure code.

## Acceptance Criteria

1. **Given** `from robot_lab.wrappers import ActionFilterWrapper` is executed,
   **When** the import completes,
   **Then** `ActionFilterWrapper` is importable, subclasses `gym.Wrapper`, and mixes in `RobotLabWrapperMixin`; no error is raised.

2. **Given** `ActionFilterWrapper._apply_filter(action)` is inspected,
   **When** the method body is read,
   **Then** the method raises `NotImplementedError` with the message `"Implement _apply_filter() to define your filtering logic."` — confirming the YouAreLazy boundary.

3. **Given** a subclass of `ActionFilterWrapper` implements `_apply_filter(action)` and wraps an environment,
   **When** `env.step(action)` is called,
   **Then** the wrapper's `_apply_filter(action)` is called with the raw action before passing the result to the inner environment.

4. **Given** the wrapper is constructed with an `ExperimentTracker` instance passed in,
   **When** the wrapper is initialized,
   **Then** `RobotLabWrapperMixin` logs the wrapper class name and any init parameters to the tracker under `metadata.json["config"]`.

5. **Given** the wrapper chaining order `Monitor → ActionFilterWrapper → VecEnv → VecNormalize` is used,
   **When** `make_env()` builds the environment stack,
   **Then** `Monitor` remains the first wrapper applied and `VecNormalize` remains the last.

## Tasks / Subtasks

- [x] Task 1: Add `RobotLabWrapperMixin` to `robot_lab/wrappers.py` (AC: 4)
  - [x] Define `RobotLabWrapperMixin` class with `_log_to_tracker(tracker, params)` method
  - [x] When `tracker` is not None, merge `{"wrapper": ClassName, **params}` into `tracker._metadata["config"]` and write metadata
  - [x] Add Google-style docstring and full type hints
  - [x] Write `tests/test_wrappers.py::test_mixin_logs_to_tracker` — verify config key written to tracker

- [x] Task 2: Add `ActionFilterWrapper` ABC to `robot_lab/wrappers.py` (AC: 1, 2, 3)
  - [x] Subclass both `gym.Wrapper` and `RobotLabWrapperMixin`
  - [x] `__init__(self, env, tracker=None, **kwargs)` — calls `super().__init__(env)` then `_log_to_tracker(tracker, kwargs)`
  - [x] `_apply_filter(action)` raises `NotImplementedError("Implement _apply_filter() to define your filtering logic.")`
  - [x] `step(action)` calls `_apply_filter(action)` then delegates to `self.env.step(filtered_action)`
  - [x] Add Google-style docstring and full type hints
  - [x] Write `tests/test_wrappers.py::test_action_filter_wrapper_importable` (AC: 1)
  - [x] Write `tests/test_wrappers.py::test_apply_filter_raises_not_implemented` (AC: 2)
  - [x] Write `tests/test_wrappers.py::test_step_calls_apply_filter` — subclass with passthrough, verify delegation (AC: 3)

- [x] Task 3: Run ruff and full test suite (AC: all)
  - [x] `ruff check robot_lab/wrappers.py tests/test_wrappers.py` — zero violations
  - [x] `pytest tests/test_wrappers.py -v` — all tests pass
  - [x] `pytest tests/ -v --ignore=tests/test_training.py` — no regressions

## Dev Notes

### Architecture Requirements (MUST Follow)

**Mixin location**: `RobotLabWrapperMixin` in `robot_lab/wrappers.py` alongside existing wrappers.

**Class hierarchy** (from architecture.md):
```
gym.Wrapper
  └── RobotLabWrapperMixin  (adds tracker param logging only)
        └── ActionFilterWrapper
              _apply_filter(action) → intentionally empty (YouAreLazy boundary)
```

**Tracker integration**: `RobotLabWrapperMixin._log_to_tracker()` merges into `tracker._metadata["config"]`
and calls `tracker._write()` to persist. If tracker is None, silently skip — no error.

**YouAreLazy boundary**: `_apply_filter` MUST raise `NotImplementedError` — this is intentional.
The implementation of actual filtering logic belongs to Marco.

**Step delegation pattern**:
```python
def step(self, action):
    filtered_action = self._apply_filter(action)
    return self.env.step(filtered_action)
```

**Wrapper chaining order** (from AC 5): `Monitor → ActionFilterWrapper → VecEnv → VecNormalize`
This means `ActionFilterWrapper` wraps the Monitor-wrapped env, before vectorisation.

**Test environment**: Use `gym.make("MountainCarContinuous-v0")` or `CartPole-v1` — no GPU required.

**Tracker param logging format** (into `metadata.json["config"]`):
```python
{"wrapper": "MyFilterWrapper", "alpha": 0.7, ...}
```

### Testing Patterns

Use `@pytest.mark.fast` marker for all tests in this story.
Use `temp_output_dir` fixture when creating ExperimentTracker in tests.

### Existing Wrappers

`wrappers.py` already contains: `ActionRepeatWrapper`, `ExponentialMovingAverageFilter`,
`LowPassFilter`, `create_action_wrapper`. These are NOT `ActionFilterWrapper` subclasses yet —
they are standalone wrappers from Phase 0. Do NOT modify them in this story.

## Dev Agent Record

### Implementation Plan
- Added `RobotLabWrapperMixin` before existing wrapper classes in `robot_lab/wrappers.py`
- `_log_to_tracker(tracker, params)`: merges `{"wrapper": ClassName, **params}` into
  `tracker._metadata["config"]` then calls `tracker._write()`. No-ops silently when tracker=None.
- `ActionFilterWrapper(gym.Wrapper, RobotLabWrapperMixin)`: `__init__` accepts `tracker=None`
  and `**kwargs`; forwards kwargs to `_log_to_tracker` as param metadata.
- `_apply_filter` raises `NotImplementedError("Implement _apply_filter() to define your filtering logic.")`
- `step` calls `_apply_filter(action)` → `self.env.step(filtered_action)`
- Created `tests/test_wrappers.py` with 12 tests across two classes covering all ACs
- NOTE: subclasses that have explicit params (e.g. `alpha`) must forward those into `**kwargs`
  when calling `super().__init__()` if they want them logged to tracker.

### Completion Notes
- 12/12 tests in `tests/test_wrappers.py` passing
- 89/89 total fast tests passing (zero regressions)
- Zero ruff violations after `--fix` applied to import ordering

## File List
- `robot_lab/wrappers.py` — added `RobotLabWrapperMixin` and `ActionFilterWrapper`
- `tests/test_wrappers.py` — created with 12 tests

## Change Log
- 2026-03-13: Story created for Epic 2, Story 2.1
- 2026-03-13: Implementation complete; all 12 tests passing; status → review
