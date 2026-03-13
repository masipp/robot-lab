"""Tests for robot_lab/wrappers.py — RobotLabWrapperMixin and ActionFilterWrapper."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from robot_lab.experiments.tracker import ExperimentTracker
from robot_lab.wrappers import ActionFilterWrapper, RobotLabWrapperMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PassthroughFilter(ActionFilterWrapper):
    """Minimal concrete subclass for testing — filter is identity."""

    def _apply_filter(self, action: np.ndarray) -> np.ndarray:  # noqa: D102
        return action


class _ScaleFilter(ActionFilterWrapper):
    """Concrete subclass that scales action by a factor — for delegation tests.

    Passes ``scale`` explicitly into the parent ``**kwargs`` so it is captured
    by ``_log_to_tracker`` when a tracker is provided.
    """

    def __init__(self, env: gym.Env, scale: float = 2.0, tracker=None, **kwargs):
        super().__init__(env, tracker=tracker, scale=scale, **kwargs)
        self.scale = scale

    def _apply_filter(self, action: np.ndarray) -> np.ndarray:  # noqa: D102
        return action * self.scale


# ---------------------------------------------------------------------------
# Task 1 tests: RobotLabWrapperMixin
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRobotLabWrapperMixin:
    """Tests for RobotLabWrapperMixin tracker integration."""

    def test_mixin_exists(self):
        """RobotLabWrapperMixin is importable and is a class."""
        assert RobotLabWrapperMixin is not None

    def test_mixin_logs_to_tracker(self, temp_output_dir: Path):
        """_log_to_tracker merges wrapper name and params into tracker config."""
        tracker = ExperimentTracker(
            experiment_name="test_wrappers",
            run_name="mixin_test",
            seed=0,
            output_dir=str(temp_output_dir),
        )
        tracker.start_run()

        env = gym.make("MountainCarContinuous-v0")
        _PassthroughFilter(env, tracker=tracker)

        # After construction the tracker config should have "wrapper" key
        assert "wrapper" in tracker._metadata["config"], (
            "Expected 'wrapper' key in tracker._metadata['config'] after wrapper init"
        )
        assert tracker._metadata["config"]["wrapper"] == "_PassthroughFilter"
        env.close()

    def test_mixin_no_tracker_no_error(self):
        """Constructing ActionFilterWrapper without tracker raises no error."""
        env = gym.make("MountainCarContinuous-v0")
        wrapper = _PassthroughFilter(env)  # tracker=None by default
        assert wrapper is not None
        env.close()

    def test_mixin_logs_kwargs_to_tracker(self, temp_output_dir: Path):
        """Extra __init__ kwargs (e.g. scale=2.0) are logged to tracker config."""
        tracker = ExperimentTracker(
            experiment_name="test_wrappers",
            run_name="mixin_kwargs_test",
            seed=0,
            output_dir=str(temp_output_dir),
        )
        tracker.start_run()

        env = gym.make("MountainCarContinuous-v0")
        _ScaleFilter(env, scale=3.5, tracker=tracker)

        cfg = tracker._metadata["config"]
        assert cfg.get("wrapper") == "_ScaleFilter"
        assert cfg.get("scale") == 3.5
        env.close()


# ---------------------------------------------------------------------------
# Task 2 tests: ActionFilterWrapper
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestActionFilterWrapper:
    """Tests for ActionFilterWrapper ABC."""

    def test_action_filter_wrapper_importable(self):
        """ActionFilterWrapper is importable and is a class. (AC: 1)"""
        assert ActionFilterWrapper is not None

    def test_is_gym_wrapper_subclass(self):
        """ActionFilterWrapper subclasses gym.Wrapper. (AC: 1)"""
        assert issubclass(ActionFilterWrapper, gym.Wrapper)

    def test_mixes_in_robotlab_wrapper_mixin(self):
        """ActionFilterWrapper mixes in RobotLabWrapperMixin. (AC: 1)"""
        assert issubclass(ActionFilterWrapper, RobotLabWrapperMixin)

    def test_apply_filter_raises_not_implemented(self):
        """_apply_filter() raises NotImplementedError with the required message. (AC: 2)"""
        env = gym.make("MountainCarContinuous-v0")

        # Instantiate via a subclass that does NOT override _apply_filter
        class _NoFilter(ActionFilterWrapper):
            pass

        wrapper = _NoFilter(env)
        action = env.action_space.sample()

        with pytest.raises(
            NotImplementedError,
            match="Implement _apply_filter\\(\\) to define your filtering logic\\.",
        ):
            wrapper._apply_filter(action)

        env.close()

    def test_apply_filter_signature(self):
        """ActionFilterWrapper._apply_filter is defined (even if abstract body)."""
        assert hasattr(ActionFilterWrapper, "_apply_filter")
        assert callable(ActionFilterWrapper._apply_filter)

    def test_step_calls_apply_filter(self):
        """step() routes action through _apply_filter before passing to inner env. (AC: 3)"""
        env = gym.make("MountainCarContinuous-v0")
        wrapper = _PassthroughFilter(env)
        wrapper.reset(seed=0)

        calls = []
        original_filter = wrapper._apply_filter

        def _spy(action):
            calls.append(action.copy())
            return original_filter(action)

        wrapper._apply_filter = _spy

        action = env.action_space.sample()
        wrapper.step(action)

        assert len(calls) == 1, "_apply_filter must be called exactly once per step()"
        np.testing.assert_array_equal(calls[0], action)
        env.close()

    def test_filtered_action_passed_to_inner_env(self):
        """step() passes _apply_filter's return value to inner environment. (AC: 3)"""
        inner_env = gym.make("MountainCarContinuous-v0")
        wrapper = _ScaleFilter(inner_env, scale=0.0)  # zero out the action
        wrapper.reset(seed=0)

        # Record what the inner env actually receives
        received_actions = []
        original_step = inner_env.step

        def _step_spy(action):
            received_actions.append(action.copy())
            return original_step(action)

        inner_env.step = _step_spy

        action = np.array([1.0])
        wrapper.step(action)

        assert len(received_actions) == 1
        np.testing.assert_array_equal(received_actions[0], np.array([0.0]))
        inner_env.close()

    def test_cannot_instantiate_directly(self):
        """ActionFilterWrapper without _apply_filter raises on step, not import. (AC: 2)"""
        env = gym.make("MountainCarContinuous-v0")

        class _Bare(ActionFilterWrapper):
            pass

        w = _Bare(env)
        w.reset(seed=0)
        action = env.action_space.sample()
        with pytest.raises(NotImplementedError):
            w.step(action)
        env.close()
