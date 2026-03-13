# Story 1.1: Plugin/Registry Infrastructure

Status: review

## Story

As a developer (Marco),
I want a `robot_lab/experiments/plugins/` module with three separate registries (`MetricsRegistry`, `VisualizationRegistry`, `MetadataRegistry`) and their abstract base classes,
so that I can extend experiment observability by registering an interface implementation — with zero changes to `training.py`, `tracker.py`, or `visualization.py`.

## Acceptance Criteria

1. **Given** the package is installed,  
   **When** I run `from robot_lab.experiments.plugins import register_metric_plugin, register_visualization_plugin, register_metadata_plugin, MetricsPlugin, VisualizationPlugin, MetadataPlugin`,  
   **Then** all six names are importable with no errors.

2. **Given** I execute `import robot_lab` with no experiment running,  
   **When** the import completes,  
   **Then** no plugins are auto-registered and no files are created (no side effects on import).

3. **Given** any registry is accessed for the first time,  
   **When** `_register_defaults()` is triggered lazily,  
   **Then** the built-in default plugins (basic reward logging, system metadata) are registered; repeated calls are idempotent.

4. **Given** a plugin is registered via `register_metric_plugin()` in an experiment script,  
   **When** a second, independent experiment runs in the same process,  
   **Then** run-scoped plugins from the first experiment do not affect the second run.

## Tasks / Subtasks

- [ ] Task 1: Create plugin base classes in `robot_lab/experiments/plugins/base.py` (AC: 1)
  - [ ] Define `MetricsPlugin` ABC with `on_step(context)`, `on_episode_end(context)`, `on_eval(context)` abstract methods
  - [ ] Define `VisualizationPlugin` ABC with `render(results)` abstract method
  - [ ] Define `MetadataPlugin` ABC with `collect(context)` abstract method
  - [ ] Add Google-style docstrings and full type hints to all ABCs
  - [ ] Write `tests/test_plugins.py::test_base_classes_are_abstract` — verify instantiating ABCs raises `TypeError`

- [ ] Task 2: Create registry singletons in `robot_lab/experiments/plugins/__init__.py` (AC: 1, 2, 3, 4)
  - [ ] Implement `PluginRegistry` generic class with: `register(plugin)`, `list_plugins()`, `_lazy_init()`, and `_initialized` flag
  - [ ] Instantiate three module-level singletons: `metrics_registry`, `visualization_registry`, `metadata_registry`
  - [ ] Implement `register_metric_plugin()`, `register_visualization_plugin()`, `register_metadata_plugin()` convenience functions
  - [ ] Re-export `MetricsPlugin`, `VisualizationPlugin`, `MetadataPlugin` from `base.py`
  - [ ] Ensure **no** `_register_defaults()` call at module level (must be lazy — AC 2)
  - [ ] Write `test_plugins.py::test_no_import_side_effects` — verify `import robot_lab` does not call `_register_defaults()`

- [ ] Task 3: Implement lazy `_register_defaults()` and built-in plugins in `robot_lab/experiments/plugins/defaults.py` (AC: 3)
  - [ ] Implement `BasicRewardLogPlugin(MetricsPlugin)` — on_episode_end logs `episode_reward` into tracker context
  - [ ] Implement `SystemMetadataPlugin(MetadataPlugin)` — collect() returns `get_system_info()` dict from `robot_lab.utils.metadata`
  - [ ] Implement `_register_defaults()` in `plugins/__init__.py` that registers both built-in plugins; guarded by `_defaults_registered` flag to be idempotent
  - [ ] Trigger `_register_defaults()` lazily on first call to `register_metric_plugin()`, `register_visualization_plugin()`, or `register_metadata_plugin()`
  - [ ] Write `test_plugins.py::test_lazy_defaults_idempotent` — call `_register_defaults()` three times; assert `len(metrics_registry.list_plugins())` equals 1 (not 3)

- [ ] Task 4: Create `robot_lab/experiments/plugins/contrib/__init__.py` (AC: 1)
  - [ ] Empty `__init__.py` — namespace placeholder for Marco's phase-specific implementations
  - [ ] Write `test_plugins.py::test_contrib_importable` — `from robot_lab.experiments.plugins.contrib import *` raises no error

- [ ] Task 5: Implement run-scoped plugin isolation (AC: 4)
  - [ ] Add `PluginRegistry.push_run_scope()` and `pop_run_scope()` context manager API that isolates plugins registered within that scope
  - [ ] Alternatively: add `PluginRegistry.clear_run_scoped()` method to be called at experiment start/end in the tracker
  - [ ] Document the isolation pattern in the module docstring with a usage example
  - [ ] Write `test_plugins.py::test_run_scoped_isolation` — register plugin in scope A, verify it is absent in scope B

- [ ] Task 6: Update `robot_lab/experiments/__init__.py` re-exports (AC: 1)
  - [ ] Add to `__all__`: `register_metric_plugin`, `register_visualization_plugin`, `register_metadata_plugin`, `MetricsPlugin`, `VisualizationPlugin`, `MetadataPlugin`
  - [ ] Import from `robot_lab.experiments.plugins` in `__init__.py`
  - [ ] Write `test_plugins.py::test_public_api_importable` — import all six names from `robot_lab.experiments.plugins` and assert they are not None

- [ ] Task 7: Run ruff and full test suite; verify zero violations (AC: all)
  - [ ] `ruff check robot_lab/experiments/plugins/` — zero violations
  - [ ] `ruff format robot_lab/experiments/plugins/` — no diffs
  - [ ] `make test` or `pytest tests/test_plugins.py -v` — all tests pass

## Dev Notes

### Architecture Requirements (MUST Follow)

**Plugin module location**: `robot_lab/experiments/plugins/` — see [architecture.md](../../.bmad_output/planning-artifacts/architecture.md) under "Plugin/Registry Architecture".

```
robot_lab/experiments/plugins/
├── __init__.py      ← singletons + register_*() + _register_defaults()
├── base.py          ← MetricsPlugin, VisualizationPlugin, MetadataPlugin ABCs
├── defaults.py      ← BasicRewardLogPlugin, SystemMetadataPlugin
└── contrib/
    └── __init__.py  ← empty namespace
```

**Registration API (exact interface required by architecture):**
```python
from robot_lab.experiments.plugins import (
    register_metric_plugin,
    register_visualization_plugin,
    register_metadata_plugin,
    MetricsPlugin,
    VisualizationPlugin,
    MetadataPlugin,
)
```

**Zero import side effects (NFR-013 hard requirement):**
- `plugins/__init__.py` MUST NOT call `_register_defaults()` at module level
- `defaults.py` MUST NOT register plugins in module-level code
- `contrib/__init__.py` MUST be empty
- Validated by `test_no_import_side_effects` test

**Plugin lifecycle hook signatures (authoritative):**
```python
class MetricsPlugin(ABC):
    @abstractmethod
    def on_step(self, context: dict) -> None: ...
    @abstractmethod
    def on_episode_end(self, context: dict) -> None: ...
    @abstractmethod
    def on_eval(self, context: dict) -> None: ...

class VisualizationPlugin(ABC):
    @abstractmethod
    def render(self, results: dict) -> None: ...

class MetadataPlugin(ABC):
    @abstractmethod
    def collect(self, context: dict) -> dict: ...
```

**`context` dict shape** (passed at invocation — do NOT hard-code access to optional keys):
```python
context = {
    "tracker": ExperimentTracker,   # always present
    "n_steps": int,                  # current step count
    "episode_rewards": list[float],  # present on episode_end only
    "locals": dict,                  # SB3 callback locals (optional)
}
```

**Idempotency pattern for `_register_defaults()`:**
```python
_defaults_registered = False

def _register_defaults() -> None:
    global _defaults_registered
    if _defaults_registered:
        return
    metrics_registry.register(BasicRewardLogPlugin())
    metadata_registry.register(SystemMetadataPlugin())
    _defaults_registered = True
```

**Run-scoped isolation strategy**: The simplest correct approach is `PluginRegistry.copy()` to snapshot global state at run start, then restore at run end. The `push_run_scope` / `pop_run_scope` API is preferred for testability. Whatever pattern is chosen, document it with a usage example.

### Project Structure Notes

**Files to create (all new):**
- `robot_lab/experiments/plugins/__init__.py`
- `robot_lab/experiments/plugins/base.py`
- `robot_lab/experiments/plugins/defaults.py`
- `robot_lab/experiments/plugins/contrib/__init__.py`
- `tests/test_plugins.py`

**Files to modify:**
- `robot_lab/experiments/__init__.py` — add plugin re-exports to `__all__`

**Files NOT to touch:**
- `robot_lab/training.py` — plugin wiring is in later stories (Epic 1 Story 1.5)
- `robot_lab/experiments/tracker.py` — refactored in Story 1.2
- Any existing test files

**Naming conventions:**
- Classes: `PascalCase` (`MetricsPlugin`, `BasicRewardLogPlugin`)
- Functions: `snake_case` (`register_metric_plugin`, `_register_defaults`)
- Registry instances: `snake_case` module-level singletons (`metrics_registry`)
- All `metadata.json` keys (when relevant): `snake_case`

### Testing Standards

Use `temp_output_dir` fixture from `conftest.py` for any tests that write files.

Test file: `tests/test_plugins.py`  
Required test functions:
1. `test_public_api_importable` — six names importable from `robot_lab.experiments.plugins`
2. `test_no_import_side_effects` — `import robot_lab` doesn't call `_register_defaults()`
3. `test_base_classes_are_abstract` — instantiating ABCs raises `TypeError`
4. `test_lazy_defaults_idempotent` — three calls to `_register_defaults()`, count stays at 1
5. `test_run_scoped_isolation` — plugin registered in scope A absent from scope B
6. `test_contrib_importable` — `from robot_lab.experiments.plugins.contrib import *` is safe

All tests must pass **without GPU**, **without live training**, and **under `make test-fast`** (< 5 seconds total).

**Markers**: Use `@pytest.mark.fast` for all tests in this story.

### References

- [Source: .bmad_output/planning-artifacts/architecture.md#Plugin/Registry Architecture]
- [Source: .bmad_output/planning-artifacts/architecture.md#Implementation Patterns & Consistency Rules]
- [Source: .bmad_output/planning-artifacts/architecture.md#Project Structure & Boundaries]
- [Source: .bmad_output/planning-artifacts/epics.md#Story 1.1: Plugin/Registry Infrastructure]
- [Source: robot_lab/experiments/__init__.py] — existing re-export pattern to follow
- [Source: robot_lab/utils/metadata.py#get_system_info] — used by `SystemMetadataPlugin.collect()`
- AR-001: Three separate singleton registries
- AR-013: Plugin registration discipline — never register on import

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6

### Debug Log References

### Completion Notes List

### File List
