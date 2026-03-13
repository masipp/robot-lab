# Story 1.2: Consolidated Experiment Metadata Schema

Status: review

## Story

As a researcher (Marco),
I want `ExperimentTracker` to write a single `metadata.json` per run with nested `run`, `config`, `system`, `metrics`, and `custom` sections (replacing the current multi-file schema),
so that each experiment run is fully self-contained and human-readable without cross-referencing separate files.

## Acceptance Criteria

1. **Given** `tracker.start_run()` is called before the first training step,  
   **When** the file is written,  
   **Then** `metadata.json["run"]["status"]` equals `"RUNNING"`, `metadata.json["config"]` contains an immutable full-config snapshot, and `metadata.json["system"]` contains Python version, GPU info, and git commit hash.

2. **Given** `tracker.end_run("COMPLETED")` is called at run completion,  
   **When** the file is updated,  
   **Then** `metadata.json["run"]["status"]` equals `"COMPLETED"` and `metadata.json["run"]["finished_at"]` is a valid ISO timestamp.

3. **Given** an unhandled exception occurs during training,  
   **When** the `finally` block in `train()` executes,  
   **Then** `metadata.json["run"]["status"]` equals `"FAILED"`.

4. **Given** a `KeyboardInterrupt` occurs,  
   **When** the `finally` block in `train()` executes,  
   **Then** `metadata.json["run"]["status"]` equals `"INTERRUPTED"`.

5. **Given** a `MetadataPlugin` calls `tracker.update("custom", data)`,  
   **When** data is merged into the file,  
   **Then** plugin outputs appear under `metadata.json["custom"]` only — the `run`, `config`, `system`, and `metrics` keys are unmodified.

6. **Given** `make test` is run after this story is implemented,  
   **When** `test_tracker.py` executes,  
   **Then** all tracker schema tests pass.

## Tasks / Subtasks

- [x] Task 1: Rewrite `ExperimentTracker` with the new `metadata.json` schema (AC: 1, 2, 3, 4, 5)
  - [x] New constructor signature: `__init__(experiment_name, run_name, base_dir, seed, phase, config_snapshot)`
  - [x] `start_run()` — writes initial `metadata.json` with `run.status = "RUNNING"`, `run.started_at`, `config` snapshot (immutable copy), and `system` section populated from `get_system_info()`
  - [x] `update(section, data)` — deep-merges `data` into `metadata[section]`; raises `ValueError` for unknown section names
  - [x] `end_run(status)` — writes final `metadata.json` with `run.status` and `run.finished_at`; status must be one of `COMPLETED`, `INTERRUPTED`, `FAILED`
  - [x] Protect `run`, `config`, `system` sections in `update()` — only `metrics` and `custom` are writable via `update()`
  - [x] `get_run_dir()` — returns `Path` to run directory (needed by checkpoint callback)
  - [x] `get_metadata()` — returns in-memory copy of current `metadata` dict (for testing)
  - [x] Write all file I/O with `json.dumps(..., indent=2)` — human-readable output
  - [x] No YAML files written by tracker (old schema used YAML — this is a breaking change)

- [x] Task 2: New directory layout (AC: 1)
  - [x] Run directory: `{base_dir}/experiments/{experiment_name}/runs/{run_id}/`
  - [x] Use `generate_run_id(suffix=f"{algo}_{env}")` from `robot_lab.utils.run_utils` for `run_id`
  - [x] `metadata.json` written to run directory root
  - [x] Accept `output_dir: Optional[str]` parameter; when provided, resolve `base_dir` relative to it via `get_experiments_dir(output_dir)`

- [x] Task 3: Wire `ExperimentTracker` into `training.py` COMPLETED/FAILED/INTERRUPTED status paths (AC: 2, 3, 4)
  - [x] Add `tracker` construction before `model.learn()` in `train()`
  - [x] Call `tracker.start_run()` before `model.learn()`
  - [x] Wrap `model.learn()` in `try/except KeyboardInterrupt/Exception/finally`:
    - `KeyboardInterrupt` → `tracker.end_run("INTERRUPTED")`
    - `Exception` → `tracker.end_run("FAILED")`
    - Normal completion → `tracker.end_run("COMPLETED")`
  - [x] No other changes to existing training logic — do NOT refactor anything else

- [x] Task 4: Create `tests/test_tracker.py` (AC: 1–6)
  - [x] `test_start_run_writes_metadata_json` — assert five top-level keys, `run.status == "RUNNING"`, `config` has the snapshot
  - [x] `test_end_run_completed` — assert `run.status == "COMPLETED"` and `run.finished_at` is parseable ISO
  - [x] `test_end_run_failed` — assert `run.status == "FAILED"`
  - [x] `test_update_metrics_merges` — call `update("metrics", {...})` twice, assert both merged
  - [x] `test_update_custom_does_not_pollute_core` — call `update("custom", {...})`, assert `run`/`config`/`system`/`metrics` keys unchanged
  - [x] `test_update_protected_section_raises` — `update("run", {...})` raises `ValueError`
  - [x] `test_config_snapshot_is_immutable` — mutate original config after `start_run()`, assert `metadata.json["config"]` unchanged
  - [x] All tests use `temp_output_dir` fixture (never write to `data/`)

- [x] Task 5: Run ruff and full test suite (AC: 6)
  - [x] `ruff check robot_lab/experiments/tracker.py robot_lab/training.py` — zero violations
  - [x] `pytest tests/test_tracker.py -v` — all 7 tests pass

## Dev Notes

### Architecture Requirements (MUST Follow)

**New `metadata.json` schema (authoritative):**
```json
{
  "run": {
    "run_id": "20260313_abc12345_sac_mountaincar",
    "experiment": "0_foundations",
    "seed": 42,
    "phase": 0,
    "status": "RUNNING",
    "started_at": "2026-03-13T10:00:00.000000",
    "finished_at": null
  },
  "config": { /* full config snapshot — immutable copy taken at start_run() */ },
  "system": {
    "python_version": "3.12.x",
    "gpu_name": "NVIDIA GeForce GTX 1080",
    "cuda_version": "12.6",
    "git_commit": "abc123..."
  },
  "metrics": {},
  "custom": {}
}
```

**Status values (only these, no others):**
```
RUNNING → COMPLETED | INTERRUPTED | FAILED
```

**`update()` write rules:**
- `update("metrics", data)` — merges `data` into `metadata["metrics"]`
- `update("custom", data)` — merges `data` into `metadata["custom"]`
- `update("run", ...)` — raises `ValueError("[Tracker] Section 'run' is read-only after start_run().")`
- `update("config", ...)` — same
- `update("system", ...)` — same
- Unknown section → `ValueError("[Tracker] Unknown section 'xyz'.")`

**Directory structure:**
```
{output_dir}/experiments/{experiment_name}/runs/{run_id}/
    metadata.json
```
Use `get_experiments_dir(output_dir)` from `robot_lab.utils.paths` for base path resolution.

**`training.py` integration (minimal required change):**
```python
try:
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        run_name=run_id,
        base_dir=...,
        seed=seed,
        phase=0,
        config_snapshot=config,
    )
    tracker.start_run()
    model.learn(...)
    tracker.end_run("COMPLETED")
except KeyboardInterrupt:
    if tracker:
        tracker.end_run("INTERRUPTED")
    raise
except Exception:
    if tracker:
        tracker.end_run("FAILED")
    raise
```

**BREAKING CHANGE:** Old tracker methods (`log_params`, `log_env_config`, `set_computed_metrics`, `log_metrics`, `log_artifact`, `set_tag`, `set_status`, `set_description`) are **removed** — replaced entirely by `start_run()`, `update()`, `end_run()`. Any existing calls to old methods will break. Check `training.py` only — no other callers exist yet.

**System info**: use `get_system_info()` from `robot_lab.utils.metadata` — do NOT duplicate the system info collection logic.

**Config snapshot immutability**: use `copy.deepcopy(config_snapshot)` in `__init__` so later mutations to the caller's dict don't affect stored config.

### Project Structure Notes

**Files to modify:**
- `robot_lab/experiments/tracker.py` — full rewrite of `ExperimentTracker` class (keep the same module name/path)
- `robot_lab/training.py` — add tracker construction and try/except wiring (minimal change only)

**Files to create:**
- `tests/test_tracker.py`

**Files NOT to touch:**
- `robot_lab/experiments/schemas.py` — existing schemas untouched
- `robot_lab/experiments/results_db.py` — extended in Story 1.4, not here
- Any test file other than `test_tracker.py`
- `robot_lab/utils/metadata.py` — use `get_system_info()` as-is

**`get_experiments_dir` existence check**: look in `robot_lab/utils/paths.py` first. If it doesn't exist, add it there — do not hardcode path logic in tracker.

### Testing Standards

All tests use `temp_output_dir` fixture (`Path` to a temp directory, auto-cleaned).  
Test file: `tests/test_tracker.py`  
Marker: `@pytest.mark.fast` on all tests (no GPU, no live training).

### References

- [Source: .bmad_output/planning-artifacts/architecture.md#Data Architecture (Experiment Storage)]
- [Source: .bmad_output/planning-artifacts/architecture.md#ExperimentTracker Write Protocol]
- [Source: .bmad_output/planning-artifacts/architecture.md#ExperimentTracker Status Transitions]
- [Source: .bmad_output/planning-artifacts/epics.md#Story 1.2]
- [Source: robot_lab/utils/metadata.py#get_system_info] — reuse directly
- [Source: robot_lab/utils/run_utils.py#generate_run_id] — reuse directly
- [Source: robot_lab/utils/paths.py] — check for `get_experiments_dir`
- AR-002, AR-006, AR-015, AR-016

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6

### Debug Log References

### Completion Notes List

### File List
