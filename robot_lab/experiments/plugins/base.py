"""Abstract base classes for robot_lab experiment plugins.

This module defines the three plugin interfaces that external code can implement
to extend observability without modifying core infrastructure.

Plugin Types:
    MetricsPlugin:        Lifecycle hooks fired during training steps and episodes.
    VisualizationPlugin:  Renders results after run completion.
    MetadataPlugin:       Collects contextual metadata merged into metadata.json["custom"].
"""

from abc import ABC, abstractmethod
from typing import Any


class MetricsPlugin(ABC):
    """Abstract base class for experiment metric collection plugins.

    Implementations are registered with ``MetricsRegistry`` and invoked at
    step, episode-end, and evaluation boundaries during training.

    The ``context`` dict passed to each hook contains:
        - ``tracker`` (ExperimentTracker): active tracker for the current run.
        - ``n_steps`` (int): total timesteps elapsed.
        - ``episode_rewards`` (list[float]): present on ``on_episode_end`` only.
        - ``locals`` (dict): SB3 callback local variables (optional).

    Do not access keys that may be absent without a ``context.get()`` guard.
    """

    @abstractmethod
    def on_step(self, context: dict[str, Any]) -> None:
        """Called after every training step.

        Args:
            context: Runtime context dict described in class docstring.
        """

    @abstractmethod
    def on_episode_end(self, context: dict[str, Any]) -> None:
        """Called when an episode completes.

        Args:
            context: Runtime context dict; ``episode_rewards`` key is present here.
        """

    @abstractmethod
    def on_eval(self, context: dict[str, Any]) -> None:
        """Called after each evaluation run.

        Args:
            context: Runtime context dict described in class docstring.
        """


class VisualizationPlugin(ABC):
    """Abstract base class for post-run visualization plugins.

    Implementations are registered with ``VisualizationRegistry`` and invoked
    after run completion to render results (e.g., learning curves, video exports).
    """

    @abstractmethod
    def render(self, results: dict[str, Any]) -> None:
        """Render results after a run completes.

        Args:
            results: Flattened results dict derived from ``metadata.json``.
        """


class MetadataPlugin(ABC):
    """Abstract base class for custom metadata collection plugins.

    Implementations are registered with ``MetadataRegistry`` and invoked at
    run end. The dict returned by ``collect()`` is merged into
    ``metadata.json["custom"]`` without overwriting core sections (``run``,
    ``config``, ``system``, ``metrics``).
    """

    @abstractmethod
    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        """Collect metadata to be merged into metadata.json["custom"].

        Args:
            context: Runtime context dict described in ``MetricsPlugin``.

        Returns:
            Dict of ``snake_case`` key/value pairs to merge under ``custom``.
        """
