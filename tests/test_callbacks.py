"""Tests for robot_lab.utils.callbacks — Story 1.3 (Atomic Model + VecNorm Checkpointing)."""

import time
import unittest.mock
from pathlib import Path

import pytest

from robot_lab.utils.callbacks import RobotLabCheckpointCallback, RobotLabEvalCallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(tmp: Path) -> unittest.mock.MagicMock:
    """Return a minimal SB3-like model mock whose .save() writes an empty .zip."""

    def _save(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")

    mock_model = unittest.mock.MagicMock()
    mock_model.save.side_effect = _save
    return mock_model


def _make_mock_env_with_vecnorm(tmp: Path) -> unittest.mock.MagicMock:
    """Return an env mock that has a .save() method (simulates VecNormalize)."""

    def _save(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")

    mock_env = unittest.mock.MagicMock()
    mock_env.save.side_effect = _save
    return mock_env


def _make_mock_env_plain() -> unittest.mock.MagicMock:
    """Return an env mock WITHOUT a .save() method (plain, non-VecNormalized env)."""
    mock_env = unittest.mock.MagicMock(spec=[])  # spec=[] means no attributes
    return mock_env


def _wire_callback(cb: RobotLabCheckpointCallback, model, env) -> None:
    """Attach minimal SB3 internals so methods like _save_atomic_pair() work.

    BaseCallback exposes ``training_env`` as a read-only property backed by
    ``self._locals`` / ``self.model.env``.  We mock the property directly so
    tests don't need a real VecEnv.
    """
    cb.model = model
    cb.num_timesteps = 1000
    # Patch the read-only property on the *instance's class* via a fresh subclass
    type(cb).training_env = property(lambda self: env)


# ---------------------------------------------------------------------------
# RobotLabCheckpointCallback tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_atomic_pair_both_files_written(temp_output_dir: Path) -> None:
    """_save_atomic_pair() must write both .zip and _vecnorm.pkl to the same dir."""
    cb = RobotLabCheckpointCallback(
        save_freq_seconds=600,
        save_path=str(temp_output_dir),
        name_prefix="sac_test",
    )
    model = _make_mock_model(temp_output_dir)
    env = _make_mock_env_with_vecnorm(temp_output_dir)
    _wire_callback(cb, model, env)

    cb._save_atomic_pair()

    zip_files = list(temp_output_dir.glob("*.zip"))
    pkl_files = list(temp_output_dir.glob("*_vecnorm.pkl"))
    assert len(zip_files) == 1, "Exactly one .zip must be written"
    assert len(pkl_files) == 1, "Exactly one _vecnorm.pkl must be written"
    # Stems must share the same checkpoint prefix (differ only by _vecnorm suffix)
    assert zip_files[0].stem in pkl_files[0].stem


@pytest.mark.fast
def test_atomic_pair_no_vecnorm_env_skips_pkl(temp_output_dir: Path) -> None:
    """When the env has no .save(), _save_atomic_pair() writes only .zip, no exception."""
    cb = RobotLabCheckpointCallback(
        save_freq_seconds=600,
        save_path=str(temp_output_dir),
        name_prefix="sac_test",
    )
    model = _make_mock_model(temp_output_dir)
    env = _make_mock_env_plain()
    _wire_callback(cb, model, env)

    cb._save_atomic_pair()  # must not raise

    zip_files = list(temp_output_dir.glob("*.zip"))
    pkl_files = list(temp_output_dir.glob("*.pkl"))
    assert len(zip_files) == 1
    assert len(pkl_files) == 0, "No .pkl should be written when env has no .save()"


@pytest.mark.fast
def test_checkpoint_count_increments(temp_output_dir: Path) -> None:
    """Each call to _save_atomic_pair() increments the internal counter."""
    cb = RobotLabCheckpointCallback(
        save_freq_seconds=600,
        save_path=str(temp_output_dir),
        name_prefix="sac_test",
    )
    model = _make_mock_model(temp_output_dir)
    env = _make_mock_env_with_vecnorm(temp_output_dir)
    _wire_callback(cb, model, env)

    cb._save_atomic_pair()
    cb._save_atomic_pair()

    assert cb._checkpoint_count == 2
    assert len(list(temp_output_dir.glob("*.zip"))) == 2


@pytest.mark.fast
def test_on_step_fires_when_interval_elapsed(temp_output_dir: Path) -> None:
    """_on_step() must call _save_atomic_pair when the time interval has elapsed."""
    cb = RobotLabCheckpointCallback(
        save_freq_seconds=1,  # 1 second for testability
        save_path=str(temp_output_dir),
        name_prefix="sac_test",
    )
    model = _make_mock_model(temp_output_dir)
    env = _make_mock_env_with_vecnorm(temp_output_dir)
    _wire_callback(cb, model, env)

    cb._last_save_time = time.time() - 2  # simulate 2s elapsed

    result = cb._on_step()

    assert result is True
    assert cb._checkpoint_count == 1


# ---------------------------------------------------------------------------
# RobotLabEvalCallback tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_eval_callback_saves_vecnorm_on_new_best(temp_output_dir: Path) -> None:
    """When a new best model is detected, best_model_vecnorm.pkl must be written."""
    best_dir = temp_output_dir / "best"
    best_dir.mkdir()

    cb = RobotLabEvalCallback.__new__(RobotLabEvalCallback)
    cb.best_mean_reward = -float("inf")
    cb.best_model_save_path = str(best_dir)

    # Simulate VecNormalize env
    def _save(path: str) -> None:
        Path(path).write_bytes(b"")

    mock_env = unittest.mock.MagicMock()
    mock_env.save.side_effect = _save
    # Patch read-only property on a per-instance class
    type(cb).training_env = property(lambda self: mock_env)

    # Patch super()._on_step() to simulate parent finding a new best reward
    def _fake_parent_on_step(self):
        self.best_mean_reward = 150.0
        return True

    with unittest.mock.patch.object(
        RobotLabEvalCallback.__bases__[0], "_on_step", _fake_parent_on_step
    ):
        result = cb._on_step()

    assert result is True
    vecnorm_path = best_dir / "best_model_vecnorm.pkl"
    assert vecnorm_path.exists(), "best_model_vecnorm.pkl must be written alongside best_model.zip"


@pytest.mark.fast
def test_eval_callback_no_vecnorm_when_no_improvement(temp_output_dir: Path) -> None:
    """If best_mean_reward does not improve, no additional vecnorm file is written."""
    best_dir = temp_output_dir / "best"
    best_dir.mkdir()

    cb = RobotLabEvalCallback.__new__(RobotLabEvalCallback)
    cb.best_mean_reward = 200.0  # already high
    cb.best_model_save_path = str(best_dir)

    mock_env = unittest.mock.MagicMock()
    type(cb).training_env = property(lambda self: mock_env)

    def _fake_parent_no_improvement(self):
        return True  # best_mean_reward stays the same

    with unittest.mock.patch.object(
        RobotLabEvalCallback.__bases__[0], "_on_step", _fake_parent_no_improvement
    ):
        cb._on_step()

    mock_env.save.assert_not_called()
