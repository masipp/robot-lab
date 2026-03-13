# Story 1.4: Results Index + Experiment Summary Template

Status: review

## Story

As a researcher (Marco),
I want `tracker.end_run()` to append a lightweight summary line to `data/experiments/results_index.jsonl`
and write a pre-filled `experiment_summary.md` to the run directory,
so that I can query runs across experiments without loading full metadata files, and have a
portfolio-ready document waiting after every run.

## Acceptance Criteria

1. **Given** `tracker.end_run()` is called on run completion,
   **When** the JSONL index is updated,
   **Then** a single valid JSON line is appended to `data/experiments/results_index.jsonl`
   containing at minimum: `run_id`, `experiment`, `seed`, `phase`, `final_reward`, `status`, `timestamp`.

2. **Given** `tracker.end_run()` is called,
   **When** the summary template is written,
   **Then** `experiment_summary.md` is present in the run directory with
   `metadata.json["run"]` and `metadata.json["config"]` values substituted into the template placeholders.

3. **Given** 10 runs have been recorded across sessions,
   **When** I read `data/experiments/results_index.jsonl`,
   **Then** it contains exactly 10 newline-delimited JSON lines in append order,
   each independently valid JSON.

4. **Given** `import robot_lab` is executed with no runs started,
   **When** the import completes,
   **Then** `results_index.jsonl` is not created and no disk writes occur.

## Tasks / Subtasks

- [x] Task 1: Add `get_results_index_path()` to `robot_lab/utils/paths.py` (AC: 1, 4)
  - [x] Returns `{data_dir}/experiments/results_index.jsonl` path **without** creating the file
  - [x] Accepts optional `output_dir` parameter (same pattern as all other path helpers)

- [x] Task 2: Implement JSONL index append in `tracker.py` (AC: 1, 3)
  - [x] `_append_results_index()` private method — called at the end of `end_run()`
  - [x] Record dict: `run_id`, `experiment`, `seed`, `phase`, `final_reward`, `status`, `timestamp`
  - [x] `final_reward` extracted from `metadata["metrics"].get("final_mean_reward", None)`
  - [x] File opened in `"a"` mode — pure append, never overwrite
  - [x] Parent directory created if needed (but only when actually writing)
  - [x] Single line: `json.dumps(record) + "\n"`

- [x] Task 3: Implement Markdown summary template in `tracker.py` (AC: 2)
  - [x] `_write_summary_md()` private method — called at the end of `end_run()`
  - [x] Template string defined as a module-level constant `_SUMMARY_TEMPLATE`
  - [x] Substituted fields: `run_id`, `experiment`, `seed`, `phase`, `status`, `started_at`,
    `finished_at`, `algorithm` (from config or "N/A"), `final_reward`
  - [x] Remainder of template left as blank fill-in sections: `## Results`, `## Observations`,
    `## Next Steps`
  - [x] Written to `{run_dir}/experiment_summary.md`

- [x] Task 4: Wire both into `end_run()` (AC: 1, 2)
  - [x] Call `_append_results_index()` after `self._write()`
  - [x] Call `_write_summary_md()` after `_append_results_index()`
  - [x] Both wrapped in try/except so a failure here never prevents the status from being saved

- [x] Task 5: Write tests in `tests/test_tracker.py` (AC: 1–4)
  - [x] `test_end_run_appends_results_index` — call `end_run("COMPLETED")`, read JSONL, assert
    required keys and valid JSON
  - [x] `test_results_index_appends_multiple_runs` — three trackers in same `output_dir`;
    assert file has exactly 3 lines, each valid JSON
  - [x] `test_end_run_writes_summary_md` — assert `experiment_summary.md` exists and contains
    `run_id` and experiment name
  - [x] `test_no_index_on_import` — import `robot_lab`; assert `results_index.jsonl` not present in cwd
  - [x] All tests use `temp_output_dir` fixture, `@pytest.mark.fast`

- [x] Task 6: Ruff + full tests (AC: all)
  - [x] `uv tool run ruff check robot_lab/experiments/tracker.py robot_lab/utils/paths.py` — zero violations
  - [x] `pytest tests/test_tracker.py -v` — all 11 tests pass

## Dev Notes

### JSONL format
Each line must be independently parseable with `json.loads()`. No trailing comma, no array wrapper.
```
{"run_id": "20260313_120000_sac_mountaincar", "experiment": "0_foundations", ...}
{"run_id": "20260313_130000_sac_walker2d", "experiment": "0_foundations", ...}
```

### Summary template placeholder strategy
Use Python `str.format_map()` with a SafeDict that returns `"N/A"` for missing keys — never KeyError.

### results_index.jsonl location
`get_results_index_path(output_dir)` returns the path; file created on first write only.
The `output_dir` stored in `self._output_dir` must be threaded through from `__init__`.

### Backward compat
`end_run()` existing behavior (write metadata.json, log) is unchanged.
The two new side effects are additive only.

## Dev Agent Record

### Implementation
- `get_results_index_path()` added to `robot_lab/utils/paths.py` — returns path without creating file
- `_SafeDict`, `_SUMMARY_TEMPLATE`, `_append_results_index()`, `_write_summary_md()` added to `tracker.py`
- `self._output_dir` stored in `__init__` for thread-through
- Both new methods called from `end_run()` wrapped in try/except (non-fatal)
- 4 new tests added to `tests/test_tracker.py`; all 11 tracker tests pass
- 77/77 fast tests green, zero ruff violations

## File List
- `robot_lab/utils/paths.py` — added `get_results_index_path()`
- `robot_lab/experiments/tracker.py` — added `_SafeDict`, `_SUMMARY_TEMPLATE`, `_append_results_index()`, `_write_summary_md()`, wired into `end_run()`
- `tests/test_tracker.py` — 4 new Story 1.4 tests appended
