"""Experiment tracking for robot_lab using a consolidated metadata.json schema.

Schema (per run):
    {
        "run":    { run_id, experiment, seed, phase, status, started_at, finished_at },
        "config": { /* immutable full-config snapshot taken at start_run() */ },
        "system": { python_version, gpu_name, cuda_version, git_commit, ... },
        "metrics": {},
        "custom": {}
    }

Only "metrics" and "custom" are writable via tracker.update().
"run", "config", and "system" are populated once at start_run() and are read-only.
"""

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from robot_lab.utils.metadata import get_system_info
from robot_lab.utils.paths import get_experiments_dir, get_results_index_path
from robot_lab.utils.run_utils import generate_run_id

# Sections that are populated once at start_run() and must not be mutated via update().
_READ_ONLY_SECTIONS = frozenset({"run", "config", "system"})
_WRITABLE_SECTIONS = frozenset({"metrics", "custom"})
_ALL_SECTIONS = _READ_ONLY_SECTIONS | _WRITABLE_SECTIONS

# Valid terminal status values.
_TERMINAL_STATUSES = frozenset({"COMPLETED", "INTERRUPTED", "FAILED"})

# ---------------------------------------------------------------------------
# Experiment summary Markdown template
# ---------------------------------------------------------------------------

_SUMMARY_TEMPLATE = """\
# Experiment Summary

| Field | Value |
|---|---|
| Run ID | {run_id} |
| Experiment | {experiment} |
| Status | {status} |
| Seed | {seed} |
| Phase | {phase} |
| Algorithm | {algorithm} |
| Started | {started_at} |
| Finished | {finished_at} |
| Final Reward | {final_reward} |

## Results

<!-- Fill in: learning curves, key metrics, plots -->

## Observations

<!-- Fill in: what worked, what didn't, surprises -->

## Next Steps

<!-- Fill in: follow-up experiments, hyperparameter changes, environment modifications -->
"""


class ExperimentTracker:
    """Lightweight experiment tracker using a single consolidated metadata.json.

    Usage::

        tracker = ExperimentTracker(
            experiment_name="0_foundations",
            run_name="sac_mountaincar",
            seed=42,
            phase=0,
            config_snapshot=config,
            output_dir=output_dir,
        )
        tracker.start_run()
        try:
            model.learn(...)
            tracker.end_run("COMPLETED")
        except KeyboardInterrupt:
            tracker.end_run("INTERRUPTED")
            raise
        except Exception:
            tracker.end_run("FAILED")
            raise
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        seed: int = 0,
        phase: int = 0,
        config_snapshot: Optional[dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        # Legacy parameter kept for backward compat — unused
        base_dir: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Initialise the tracker and create the run directory.

        Args:
            experiment_name: Experiment campaign name (e.g. "0_foundations").
            run_name: Short descriptor used in the run_id suffix (e.g. "sac_mountaincar").
            seed: Random seed for this run (recorded in metadata).
            phase: Research phase number (0–4).
            config_snapshot: Full hyperparameter config dict; deep-copied immediately so
                later mutations by the caller do not affect stored config.
            output_dir: Optional custom output root. Resolved via get_experiments_dir().
            base_dir: Ignored (legacy compat).
            tag: Ignored (legacy compat).
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.seed = seed
        self.phase = phase
        self._config_snapshot: dict[str, Any] = copy.deepcopy(config_snapshot or {})
        self._output_dir: Optional[str] = output_dir

        # Generate unique run ID.
        self.run_id = generate_run_id(suffix=run_name)

        # Build run directory: {experiments_dir}/{experiment_name}/runs/{run_id}/
        experiments_base = get_experiments_dir(output_dir)
        self.run_dir: Path = experiments_base / experiment_name / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_path: Path = self.run_dir / "metadata.json"

        # In-memory metadata dict — written to disk on start_run() and end_run().
        self._metadata: dict[str, Any] = {
            "run": {},
            "config": {},
            "system": {},
            "metrics": {},
            "custom": {},
        }

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Write initial metadata.json with status RUNNING.

        Must be called once, before the first training step. Subsequent calls
        are a no-op (idempotent guard against double-initialisation).
        """
        if self._metadata["run"].get("status") == "RUNNING":
            logger.warning("⚠ [Tracker] start_run() called again — already RUNNING, skipping.")
            return

        self._metadata["run"] = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "seed": self.seed,
            "phase": self.phase,
            "status": "RUNNING",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        }
        self._metadata["config"] = copy.deepcopy(self._config_snapshot)
        self._metadata["system"] = get_system_info()

        self._write()
        logger.info(f"✔ [Tracker] Run started: {self.run_id} → {self._metadata_path}")

    def end_run(self, status: str) -> None:
        """Finalise metadata.json with terminal status and finished_at timestamp.

        Args:
            status: Terminal status — must be one of COMPLETED, INTERRUPTED, FAILED.

        Raises:
            ValueError: If status is not a recognised terminal value.
        """
        if status not in _TERMINAL_STATUSES:
            raise ValueError(
                f"[Tracker] Invalid status '{status}'. "
                f"Must be one of: {sorted(_TERMINAL_STATUSES)}."
            )

        self._metadata["run"]["status"] = status
        self._metadata["run"]["finished_at"] = datetime.now().isoformat()

        self._write()
        logger.info(f"✔ [Tracker] Run ended ({status}): {self.run_id}")

        try:
            self._append_results_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠ [Tracker] Could not append results index: {exc}")

        try:
            self._write_summary_md()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠ [Tracker] Could not write experiment_summary.md: {exc}")

    def update(self, section: str, data: dict[str, Any]) -> None:
        """Deep-merge data into a writable metadata section.

        Only 'metrics' and 'custom' are writable. Attempting to write to
        'run', 'config', or 'system' raises ValueError.

        Args:
            section: Target section name ('metrics' or 'custom').
            data: Dict to deep-merge into the section.

        Raises:
            ValueError: If section is read-only or unknown.
        """
        if section in _READ_ONLY_SECTIONS:
            raise ValueError(
                f"[Tracker] Section '{section}' is read-only after start_run(). "
                f"Writable sections: {sorted(_WRITABLE_SECTIONS)}."
            )
        if section not in _ALL_SECTIONS:
            raise ValueError(
                f"[Tracker] Unknown section '{section}'. "
                f"Known sections: {sorted(_ALL_SECTIONS)}."
            )
        _deep_merge(self._metadata[section], data)
        self._write()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_run_dir(self) -> Path:
        """Return the Path to this run's output directory.

        Returns:
            Absolute Path to the run directory.
        """
        return self.run_dir

    def get_metadata(self) -> dict[str, Any]:
        """Return a deep copy of the current in-memory metadata dict.

        Returns:
            Deep copy of the full metadata dict (all five sections).
        """
        return copy.deepcopy(self._metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_results_index(self) -> None:
        """Append a one-line JSON summary to the shared results_index.jsonl."""
        index_path = get_results_index_path(self._output_dir)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "run_id": self._metadata["run"].get("run_id", self.run_id),
            "experiment": self._metadata["run"].get("experiment", self.experiment_name),
            "seed": self._metadata["run"].get("seed", self.seed),
            "phase": self._metadata["run"].get("phase", self.phase),
            "final_reward": self._metadata["metrics"].get("final_mean_reward", None),
            "status": self._metadata["run"].get("status", "UNKNOWN"),
            "timestamp": self._metadata["run"].get("finished_at", datetime.now().isoformat()),
        }

        with index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        logger.debug(f"✔ [Tracker] Results index updated → {index_path}")

    def _write_summary_md(self) -> None:
        """Write a pre-filled experiment_summary.md to the run directory."""
        run = self._metadata["run"]
        config = self._metadata["config"]

        fields = {
            "run_id": run.get("run_id", self.run_id),
            "experiment": run.get("experiment", self.experiment_name),
            "status": run.get("status", "UNKNOWN"),
            "seed": run.get("seed", self.seed),
            "phase": run.get("phase", self.phase),
            "algorithm": config.get("algorithm", "N/A"),
            "started_at": run.get("started_at", "N/A"),
            "finished_at": run.get("finished_at", "N/A"),
            "final_reward": self._metadata["metrics"].get("final_mean_reward", "N/A"),
        }

        summary_path = self.run_dir / "experiment_summary.md"
        summary_path.write_text(
            _SUMMARY_TEMPLATE.format_map(_SafeDict(fields)),
            encoding="utf-8",
        )
        logger.debug(f"✔ [Tracker] Summary written → {summary_path}")

    def _write(self) -> None:
        """Atomically write the in-memory metadata dict to metadata.json."""
        self._metadata_path.write_text(
            json.dumps(self._metadata, indent=2, default=str),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class _SafeDict(dict):
    """dict subclass that returns 'N/A' for missing keys in str.format_map()."""

    def __missing__(self, key: str) -> str:
        return "N/A"


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge source into target in-place.

    Nested dicts are merged; all other types are overwritten.

    Args:
        target: Dict to merge into (modified in-place).
        source: Dict providing new/updated values.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
