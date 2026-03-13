# Story 2.2: Smoothness Metric Plugin

Status: review

## Story

As a researcher (Marco),
I want an `ActionSmoothnessMetricPlugin` registered as a built-in default plugin that logs `∑‖aₜ−aₜ₋₁‖²` per episode to TensorBoard under `smoothness/action_delta_norm` and into `metadata.json["metrics"]`,
So that smoothness is automatically tracked for any training run without requiring me to write metric collection code.

## Acceptance Criteria

1. **Given** `plugins/defaults.py` is loaded via lazy `_register_defaults()`,
   **When** `ActionSmoothnessMetricPlugin` is registered,
   **Then** it appears in `MetricsRegistry.list_plugins()` without any manual registration call in experiment scripts.

2. **Given** a training run is active and `MetricsRegistry.on_episode_end(context)` fires,
   **When** the plugin processes the episode,
   **Then** `smoothness/action_delta_norm` is logged as a TensorBoard scalar for that episode.

3. **Given** `tracker.end_run("COMPLETED")` is called,
   **When** the final `metadata.json` is written,
   **Then** `metadata.json["metrics"]["smoothness_action_delta_norm"]` contains the per-episode series (or at minimum the final mean value).

4. **Given** `ActionSmoothnessMetricPlugin` is imported at the module level in `defaults.py`,
   **When** `import robot_lab` is executed without running an experiment,
   **Then** no metric computation occurs and no files are written (plugin logic is lifecycle-gated, not import-triggered).

## Tasks / Subtasks

- [x] Task 1: Add aggregate dispatch methods to `PluginRegistry` in `robot_lab/experiments/plugins/__init__.py` (AC: 2)
  - [x] Add `on_step(context)` method — calls `plugin.on_step(context)` for each active plugin
  - [x] Add `on_episode_end(context)` method — calls `plugin.on_episode_end(context)` for each active plugin
  - [x] Add `on_eval(context)` method — calls `plugin.on_eval(context)` for each active plugin
  - [x] Exception in one plugin MUST NOT prevent other plugins from running (log + continue)
  - [x] Write `tests/test_plugins.py::test_registry_dispatches_on_episode_end` — register mock plugin, call on_episode_end, assert called

- [x] Task 2: Create `ActionSmoothnessMetricPlugin` in `robot_lab/experiments/plugins/defaults.py` (AC: 1, 2, 3, 4)
  - [x] Track `_prev_action` and `_episode_delta_sq_sum` per instance as state (reset on `on_step` first call or episode reset)
  - [x] `on_step(context)`: extract `context.get("actions")`, compute `‖aₜ−aₜ₋₁‖²`, accumulate into `_episode_delta_sq_sum`; skip gracefully if actions missing
  - [x] `on_episode_end(context)`: compute per-episode total; log to TensorBoard via `context.get("sb3_logger")` if present; call `tracker.update("metrics", {"smoothness_action_delta_norm": [value]})` if tracker present; reset state
  - [x] `on_eval(context)`: no-op
  - [x] Add Google-style docstring and full type hints
  - [x] Write `tests/test_plugins.py::test_smoothness_plugin_registered_by_defaults` (AC: 1)
  - [x] Write `tests/test_plugins.py::test_smoothness_plugin_on_episode_end_logs_to_tracker` (AC: 3) — use real ExperimentTracker
  - [x] Write `tests/test_plugins.py::test_smoothness_plugin_tb_logging` (AC: 2) — mock sb3_logger, verify record() called
  - [x] Write `tests/test_plugins.py::test_smoothness_plugin_no_import_side_effects` (AC: 4)

- [x] Task 3: Register `ActionSmoothnessMetricPlugin` in `_register_defaults()` (AC: 1)
  - [x] Import and instantiate in `_register_defaults()` alongside existing built-ins
  - [x] Verify idempotency: calling `_register_defaults()` twice does not double-register

- [x] Task 4: Run ruff and full test suite (AC: all)
  - [x] `ruff check robot_lab/experiments/plugins/ tests/test_plugins.py` — zero violations
  - [x] `pytest tests/test_plugins.py -v` — all tests pass
  - [x] `pytest tests/ -v --ignore=tests/test_training.py` — no regressions

## Dev Notes

### Architecture Requirements

**Plugin file location**: `ActionSmoothnessMetricPlugin` goes in `robot_lab/experiments/plugins/defaults.py`
alongside `BasicRewardLogPlugin` and `SystemMetadataPlugin`.

**PluginRegistry dispatch pattern** (add to `PluginRegistry` class in `__init__.py`):
```python
def on_episode_end(self, context: dict) -> None:
    for plugin in self.list_plugins():
        try:
            plugin.on_episode_end(context)
        except Exception as exc:
            logger.warning(f"[{self._name}] Plugin error in on_episode_end: {exc}")
```

**Context dict keys used by ActionSmoothnessMetricPlugin**:
- `context.get("actions")` — `np.ndarray` of shape `(n_envs, action_dim)` or `(action_dim,)`
- `context.get("tracker")` — `ExperimentTracker` instance (may be None)
- `context.get("sb3_logger")` — SB3 Logger (may be None); call `.record("smoothness/action_delta_norm", val)` and `.dump(step)` if present

**Metric formula**: For a single episode with T steps:
```
smoothness_action_delta_norm = ∑_{t=1}^{T} ‖aₜ - aₜ₋₁‖²
```
Use `np.sum((action - prev_action) ** 2)` for the squared L2 norm.

**State management**: Plugin state (`_prev_action`, `_episode_delta_sq_sum`) resets at episode end.
For multi-env contexts, use the mean action across envs: `action = actions.mean(axis=0)` when actions is 2D.

**Tracker update format**:
```python
tracker.update("metrics", {"smoothness_action_delta_norm": [value]})
```
This appends via `_deep_merge`. Subsequent episodes add more values to the list via the `_deep_merge`
logic in `tracker.update()`.

### Testing Patterns

- Use `@pytest.mark.fast` for all tests in this story
- Use `temp_output_dir` fixture when creating ExperimentTracker
- For sb3_logger mock: create a `unittest.mock.MagicMock()` and pass as `context["sb3_logger"]`
- Reset `_defaults_registered` flag between tests to avoid cross-test contamination:
  ```python
  import robot_lab.experiments.plugins as plugins_mod
  plugins_mod._defaults_registered = False
  plugins_mod.metrics_registry._global_plugins.clear()
  ```

### Deep-merge behavior
`tracker.update("metrics", {"smoothness_action_delta_norm": [value]})` — `_deep_merge` replaces
scalar with list if key absent, or deep-merges if key present. Check how the first vs subsequent
calls behave to ensure values accumulate correctly.

## Dev Agent Record

### Implementation Plan
- Added `on_step()`, `on_episode_end()`, `on_eval()` aggregate dispatch methods to `PluginRegistry`
  in `robot_lab/experiments/plugins/__init__.py`; each method iterates `list_plugins()` and
  catches+logs individual plugin exceptions so one bad plugin doesn't kill others.
- Added `ActionSmoothnessMetricPlugin` to `robot_lab/experiments/plugins/defaults.py`:
  - `on_step()`: extracts action from context, handles 2D (multi-env) via mean(axis=0),
    accumulates `‖aₜ−aₜ₋₁‖²` into `_episode_delta_sq_sum`.
  - `on_episode_end()`: logs to TensorBoard via `sb3_logger.record()`; writes full per-episode
    series list to `tracker.update("metrics", {...})`; resets state.
  - `_all_episode_values` keeps the complete series; each `on_episode_end` overwrites the
    full list in tracker so `_deep_merge` (which replaces lists) always has the latest series.
- Registered `ActionSmoothnessMetricPlugin()` in `_register_defaults()` after `BasicRewardLogPlugin()`.
- Updated `test_lazy_defaults_idempotent` expected count from 1 → 2 metrics plugins.

### Completion Notes
- 24/24 tests in `tests/test_plugins.py` passing
- 97/97 total fast tests passing (zero regressions)
- Zero ruff violations

## File List
- `robot_lab/experiments/plugins/__init__.py` — added `on_step()`, `on_episode_end()`, `on_eval()` to `PluginRegistry`; registered `ActionSmoothnessMetricPlugin` in `_register_defaults()`
- `robot_lab/experiments/plugins/defaults.py` — added `ActionSmoothnessMetricPlugin`
- `tests/test_plugins.py` — added 8 new tests (dispatch + smoothness plugin coverage)

## Change Log
- 2026-03-13: Story created for Epic 2, Story 2.2
- 2026-03-13: Implementation complete; all 24 tests passing; status → review
