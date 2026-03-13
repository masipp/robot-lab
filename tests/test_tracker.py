"""Tests for robot_lab.experiments.tracker — Story 1.2 (Consolidated Experiment Metadata Schema)
and Story 1.4 (Results Index + Experiment Summary Template)."""

import json
from pathlib import Path

import pytest

from robot_lab.experiments.tracker import ExperimentTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tracker(tmp: Path, config: dict | None = None) -> ExperimentTracker:
    """Return a fresh ExperimentTracker pointing at *tmp* as output root."""
    return ExperimentTracker(
        experiment_name="test_exp",
        run_name="test_run",
        seed=7,
        phase=0,
        config_snapshot=config or {"algorithm": "SAC", "lr": 3e-4},
        output_dir=str(tmp),
    )


def _read_metadata(tracker: ExperimentTracker) -> dict:
    meta_path = tracker.get_run_dir() / "metadata.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_start_run_writes_metadata_json(temp_output_dir: Path) -> None:
    """start_run() must create metadata.json with all five top-level keys
    and set run.status to RUNNING, and persist the config snapshot."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()

    meta_path = tracker.get_run_dir() / "metadata.json"
    assert meta_path.exists(), "metadata.json must exist after start_run()"

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"run", "config", "system", "metrics", "custom"}
    assert data["run"]["status"] == "RUNNING"
    assert data["run"]["run_id"] == tracker.run_id
    assert data["run"]["seed"] == 7
    assert data["run"]["finished_at"] is None
    assert data["config"]["algorithm"] == "SAC"


@pytest.mark.fast
def test_end_run_completed(temp_output_dir: Path) -> None:
    """end_run('COMPLETED') must set status and populate finished_at."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()
    tracker.end_run("COMPLETED")

    data = _read_metadata(tracker)
    assert data["run"]["status"] == "COMPLETED"
    finished = data["run"]["finished_at"]
    assert finished is not None, "finished_at must be set"
    # Must be parseable as ISO datetime — datetime.fromisoformat raises on bad input
    from datetime import datetime
    datetime.fromisoformat(finished)  # raises ValueError if malformed


@pytest.mark.fast
def test_end_run_failed(temp_output_dir: Path) -> None:
    """end_run('FAILED') must set status to FAILED."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()
    tracker.end_run("FAILED")

    data = _read_metadata(tracker)
    assert data["run"]["status"] == "FAILED"


@pytest.mark.fast
def test_update_metrics_merges(temp_output_dir: Path) -> None:
    """Two successive update('metrics', ...) calls must deep-merge, not overwrite."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()

    tracker.update("metrics", {"episode_reward_mean": 120.5})
    tracker.update("metrics", {"episode_length_mean": 500})

    data = _read_metadata(tracker)
    assert data["metrics"]["episode_reward_mean"] == pytest.approx(120.5)
    assert data["metrics"]["episode_length_mean"] == 500


@pytest.mark.fast
def test_update_custom_does_not_pollute_core_sections(temp_output_dir: Path) -> None:
    """Writing to 'custom' must not affect run/config/system/metrics sections."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()
    original = _read_metadata(tracker)

    tracker.update("custom", {"note": "hello", "extra": 99})

    after = _read_metadata(tracker)
    assert after["run"] == original["run"]
    assert after["config"] == original["config"]
    assert after["system"] == original["system"]
    assert after["metrics"] == original["metrics"]
    assert after["custom"]["note"] == "hello"


@pytest.mark.fast
def test_update_protected_section_raises(temp_output_dir: Path) -> None:
    """update() for a read-only section must raise ValueError."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()

    with pytest.raises(ValueError, match="read-only"):
        tracker.update("run", {"status": "PWNED"})

    with pytest.raises(ValueError, match="read-only"):
        tracker.update("config", {"injected": True})

    with pytest.raises(ValueError, match="read-only"):
        tracker.update("system", {"gpu": "fake"})


@pytest.mark.fast
def test_config_snapshot_is_immutable(temp_output_dir: Path) -> None:
    """Mutating the original config after start_run() must not change stored config."""
    cfg = {"algorithm": "PPO", "lr": 1e-3}
    tracker = ExperimentTracker(
        experiment_name="test_exp",
        run_name="immutable_check",
        seed=0,
        phase=0,
        config_snapshot=cfg,
        output_dir=str(temp_output_dir),
    )
    tracker.start_run()

    # Mutate the original dict AFTER start_run
    cfg["algorithm"] = "TAMPERED"
    cfg["new_key"] = "injected"

    data = _read_metadata(tracker)
    assert data["config"]["algorithm"] == "PPO", (
        "Stored config must reflect snapshot at start_run()"
    )
    assert "new_key" not in data["config"]


# ---------------------------------------------------------------------------
# Story 1.4 — Results Index + Experiment Summary Template
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_end_run_appends_results_index(temp_output_dir: Path) -> None:
    """end_run() must append a valid JSON line to results_index.jsonl."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()
    tracker.end_run("COMPLETED")

    from robot_lab.utils.paths import get_results_index_path
    index_path = get_results_index_path(str(temp_output_dir))
    assert index_path.exists(), "results_index.jsonl must be created after end_run()"

    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    required_keys = ("run_id", "experiment", "seed", "phase", "final_reward", "status", "timestamp")
    for required_key in required_keys:
        assert required_key in record, f"Missing required key: {required_key}"
    assert record["status"] == "COMPLETED"
    assert record["experiment"] == "test_exp"
    assert record["seed"] == 7


@pytest.mark.fast
def test_results_index_appends_multiple_runs(temp_output_dir: Path) -> None:
    """Three separate trackers writing to the same output_dir produce 3 JSONL lines."""
    for i in range(3):
        tracker = ExperimentTracker(
            experiment_name="multi_run_exp",
            run_name=f"run_{i}",
            seed=i,
            phase=0,
            config_snapshot={"algorithm": "SAC"},
            output_dir=str(temp_output_dir),
        )
        tracker.start_run()
        tracker.end_run("COMPLETED")

    from robot_lab.utils.paths import get_results_index_path
    index_path = get_results_index_path(str(temp_output_dir))
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "Exactly 3 JSONL lines expected"

    for line in lines:
        record = json.loads(line)  # must not raise
        assert record["experiment"] == "multi_run_exp"


@pytest.mark.fast
def test_end_run_writes_summary_md(temp_output_dir: Path) -> None:
    """end_run() must write experiment_summary.md with run_id and experiment name."""
    tracker = _build_tracker(temp_output_dir)
    tracker.start_run()
    tracker.end_run("COMPLETED")

    summary_path = tracker.get_run_dir() / "experiment_summary.md"
    assert summary_path.exists(), "experiment_summary.md must be created after end_run()"

    content = summary_path.read_text(encoding="utf-8")
    assert tracker.run_id in content
    assert "test_exp" in content
    assert "COMPLETED" in content
    # Confirm the fill-in sections are present
    assert "## Results" in content
    assert "## Observations" in content
    assert "## Next Steps" in content


@pytest.mark.fast
def test_no_index_on_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing robot_lab must not create results_index.jsonl."""
    monkeypatch.chdir(tmp_path)

    import robot_lab  # noqa: F401 — side-effect check only

    index_path = tmp_path / "data" / "experiments" / "results_index.jsonl"
    assert not index_path.exists(), "results_index.jsonl must not exist after bare import"

