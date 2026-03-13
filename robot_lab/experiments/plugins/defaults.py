"""Built-in default plugins registered lazily on first registry access.

These plugins provide baseline observability for all training runs without
requiring any manual registration in experiment scripts.

Plugins defined here MUST NOT register themselves at module level —
registration is triggered exclusively by ``_register_defaults()`` in
``robot_lab/experiments/plugins/__init__.py``.
"""

from typing import Any

from robot_lab.experiments.plugins.base import MetadataPlugin, MetricsPlugin


class BasicRewardLogPlugin(MetricsPlugin):
    """Log episode reward into the tracker context at episode end.

    Stores each episode's total reward under ``context["tracker"]`` via
    ``tracker.update("metrics", {...})``.  Requires the tracker to be present
    in the context dict; silently skips if absent (e.g., unit-test stubs).
    """

    def on_step(self, context: dict[str, Any]) -> None:
        """No-op: reward logging occurs only at episode end.

        Args:
            context: Runtime context dict.
        """

    def on_episode_end(self, context: dict[str, Any]) -> None:
        """Record episode reward into the tracker.

        Args:
            context: Runtime context dict; ``episode_rewards`` and ``tracker``
                keys are expected to be present.
        """
        tracker = context.get("tracker")
        episode_rewards: list[float] = context.get("episode_rewards", [])
        if tracker is not None and episode_rewards:
            tracker.update("metrics", {"episode_rewards": episode_rewards})

    def on_eval(self, context: dict[str, Any]) -> None:
        """No-op: evaluation metrics are handled by EvalCallback directly.

        Args:
            context: Runtime context dict.
        """


class SystemMetadataPlugin(MetadataPlugin):
    """Collect system metadata into metadata.json["custom"] at run end.

    Uses ``robot_lab.utils.metadata.get_system_info()`` to capture Python
    version, GPU name, CUDA version, and PyTorch version.  The returned dict
    lands under ``metadata.json["custom"]["system_metadata"]``.
    """

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return system info dict to be merged into metadata.json["custom"].

        Args:
            context: Runtime context dict (not used directly here).

        Returns:
            Dict with key ``system_metadata`` containing system information.
        """
        from robot_lab.utils.metadata import get_system_info

        return {"system_metadata": get_system_info()}


class ActionSmoothnessMetricPlugin(MetricsPlugin):
    """Track action smoothness per episode: logs ``∑‖aₜ−aₜ₋₁‖²`` to TensorBoard and tracker.

    At each training step, the plugin extracts the action from ``context["actions"]``
    and accumulates the squared L2 norm of the action delta.  At episode end, the
    accumulated sum is written to:

    - TensorBoard scalar ``smoothness/action_delta_norm`` (if ``context["sb3_logger"]``
      is present).
    - ``metadata.json["metrics"]["smoothness_action_delta_norm"]`` as a per-episode
      list (if ``context["tracker"]`` is present).

    State (``_prev_action``, ``_episode_delta_sq_sum``) resets automatically after
    each episode.

    Context keys consumed:
        ``on_step``:
            - ``actions`` (np.ndarray, optional): current action (shape ``(action_dim,)``
              or ``(n_envs, action_dim)``).  For 2D arrays the per-env mean is used.
        ``on_episode_end``:
            - ``tracker`` (ExperimentTracker, optional): writes the metric.
            - ``sb3_logger`` (SB3 Logger, optional): logs TensorBoard scalar.
            - ``n_steps`` (int, optional): global step count for TensorBoard ``dump()``.
    """

    def __init__(self) -> None:
        """Initialise plugin state."""
        self._prev_action: Any = None
        self._episode_delta_sq_sum: float = 0.0
        self._all_episode_values: list[float] = []

    def on_step(self, context: dict[str, Any]) -> None:
        """Accumulate squared action delta for the current step.

        Args:
            context: Runtime context dict; ``actions`` key is read if present.
        """
        import numpy as np

        action = context.get("actions")
        if action is None:
            return

        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 2:
            action = action.mean(axis=0)

        if self._prev_action is not None:
            delta = action - self._prev_action
            self._episode_delta_sq_sum += float(np.sum(delta ** 2))

        self._prev_action = action.copy()

    def on_episode_end(self, context: dict[str, Any]) -> None:
        """Log accumulated episode smoothness to TensorBoard and tracker.

        Args:
            context: Runtime context dict with optional ``tracker`` and ``sb3_logger``.
        """
        value = self._episode_delta_sq_sum
        self._all_episode_values.append(value)

        tracker = context.get("tracker")
        sb3_logger = context.get("sb3_logger")
        n_steps = context.get("n_steps", 0)

        if sb3_logger is not None:
            try:
                sb3_logger.record("smoothness/action_delta_norm", value)
                sb3_logger.dump(n_steps)
            except Exception:  # noqa: BLE001
                pass

        if tracker is not None:
            try:
                tracker.update(
                    "metrics",
                    {"smoothness_action_delta_norm": list(self._all_episode_values)},
                )
            except Exception:  # noqa: BLE001
                pass

        # Reset per-episode state.
        self._prev_action = None
        self._episode_delta_sq_sum = 0.0

    def on_eval(self, context: dict[str, Any]) -> None:
        """No-op: smoothness tracking is episode-scoped, not eval-scoped.

        Args:
            context: Runtime context dict (not used).
        """
