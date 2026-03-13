"""Plugin/registry infrastructure for robot_lab experiment observability.

Three module-level singleton registries manage lifecycle hooks without coupling
to training, tracker, or visualization code:

- ``metrics_registry`` — MetricsPlugin hooks (on_step, on_episode_end, on_eval)
- ``visualization_registry`` — VisualizationPlugin hooks (render)
- ``metadata_registry`` — MetadataPlugin hooks (collect)

Usage — global plugin (active for all runs in the process):

    from robot_lab.experiments.plugins import register_metric_plugin
    from my_plugins import MyMetrics
    register_metric_plugin(MyMetrics())

Usage — run-scoped plugin (isolated to a single experiment run):

    from robot_lab.experiments.plugins import metrics_registry
    with metrics_registry.run_scope():
        register_metric_plugin(RunScopedPlugin())
        train(...)  # plugin is active only inside this block

Registration rules:
- Plugins MUST NOT register themselves at import/module level.
- Global plugins: call register_*() explicitly in experiment scripts.
- Built-in defaults: registered lazily on first registry access via _register_defaults().
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Generic, TypeVar

from robot_lab.experiments.plugins.base import (
    MetadataPlugin,
    MetricsPlugin,
    VisualizationPlugin,
)

# ---------------------------------------------------------------------------
# Generic registry
# ---------------------------------------------------------------------------

_P = TypeVar("_P")


class PluginRegistry(Generic[_P]):
    """Thread-local-safe generic registry for a single plugin type.

    The registry maintains two separate lists:
    - ``_global_plugins``: persistent across all runs in the process.
    - ``_run_plugins``: active only within the innermost ``run_scope()`` block.

    ``list_plugins()`` returns the union of both lists.
    """

    def __init__(self, name: str) -> None:
        """Initialise the registry.

        Args:
            name: Human-readable registry name used in log messages.
        """
        self._name = name
        self._global_plugins: list[_P] = []
        self._run_plugins: list[_P] = []
        self._in_run_scope: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: _P, *, run_scoped: bool = False) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance to register.
            run_scoped: If True, register in the current run scope only.
                Raises RuntimeError if called outside an active run_scope().
        """
        if run_scoped:
            if not self._in_run_scope:
                raise RuntimeError(
                    f"[{self._name}] run_scoped=True requires an active run_scope() context."
                )
            self._run_plugins.append(plugin)
        else:
            self._global_plugins.append(plugin)

    def list_plugins(self) -> list[_P]:
        """Return all active plugins (global + current run-scoped).

        Returns:
            List of active plugin instances.
        """
        return self._global_plugins + self._run_plugins

    # ------------------------------------------------------------------
    # Run-scope context manager
    # ------------------------------------------------------------------

    @contextmanager
    def run_scope(self) -> Generator[None, None, None]:
        """Context manager that isolates run-scoped plugins.

        Plugins registered with ``run_scoped=True`` (or via
        ``register_*_plugin(..., run_scoped=True)``) inside this block are
        automatically cleared when the block exits — they never bleed into
        subsequent runs.

        Example::

            with metrics_registry.run_scope():
                register_metric_plugin(MyPlugin(), run_scoped=True)
                train(...)  # MyPlugin active here only
            # MyPlugin gone after this line
        """
        self._in_run_scope = True
        try:
            yield
        finally:
            self._run_plugins.clear()
            self._in_run_scope = False

    def clear_run_scoped(self) -> None:
        """Discard all run-scoped plugins without a context manager.

        Intended for ``ExperimentTracker.end_run()`` cleanup in non-contextmanager
        usage patterns.
        """
        self._run_plugins.clear()

    # ------------------------------------------------------------------
    # Aggregate dispatch (MetricsPlugin only — typed via Generic[_P])
    # ------------------------------------------------------------------

    def on_step(self, context: dict) -> None:
        """Call ``on_step(context)`` on every active MetricsPlugin.

        Errors in individual plugins are caught and logged so one failing
        plugin does not prevent the others from running.

        Args:
            context: Runtime context dict passed to each plugin.
        """
        for plugin in self.list_plugins():
            try:
                plugin.on_step(context)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                from loguru import logger as _logger
                _logger.warning(f"[{self._name}] Plugin error in on_step: {exc}")

    def on_episode_end(self, context: dict) -> None:
        """Call ``on_episode_end(context)`` on every active MetricsPlugin.

        Errors in individual plugins are caught and logged so one failing
        plugin does not prevent the others from running.

        Args:
            context: Runtime context dict passed to each plugin.
        """
        for plugin in self.list_plugins():
            try:
                plugin.on_episode_end(context)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                from loguru import logger as _logger
                _logger.warning(f"[{self._name}] Plugin error in on_episode_end: {exc}")

    def on_eval(self, context: dict) -> None:
        """Call ``on_eval(context)`` on every active MetricsPlugin.

        Errors in individual plugins are caught and logged so one failing
        plugin does not prevent the others from running.

        Args:
            context: Runtime context dict passed to each plugin.
        """
        for plugin in self.list_plugins():
            try:
                plugin.on_eval(context)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                from loguru import logger as _logger
                _logger.warning(f"[{self._name}] Plugin error in on_eval: {exc}")


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

metrics_registry: PluginRegistry[MetricsPlugin] = PluginRegistry("MetricsRegistry")
visualization_registry: PluginRegistry[VisualizationPlugin] = PluginRegistry(
    "VisualizationRegistry"
)
metadata_registry: PluginRegistry[MetadataPlugin] = PluginRegistry("MetadataRegistry")

# ---------------------------------------------------------------------------
# Lazy default registration
# ---------------------------------------------------------------------------

_defaults_registered: bool = False


def _register_defaults() -> None:
    """Register built-in default plugins. Idempotent — safe to call multiple times.

    Built-in defaults:
    - ``BasicRewardLogPlugin`` — logs episode reward into tracker context.
    - ``SystemMetadataPlugin`` — collects system info into metadata.json["custom"].

    Called lazily on first invocation of register_metric_plugin(),
    register_visualization_plugin(), or register_metadata_plugin().
    """
    global _defaults_registered
    if _defaults_registered:
        return
    from robot_lab.experiments.plugins.defaults import (
        ActionSmoothnessMetricPlugin,
        BasicRewardLogPlugin,
        SystemMetadataPlugin,
    )

    metrics_registry.register(BasicRewardLogPlugin())
    metrics_registry.register(ActionSmoothnessMetricPlugin())
    metadata_registry.register(SystemMetadataPlugin())
    _defaults_registered = True


# ---------------------------------------------------------------------------
# Public registration helpers
# ---------------------------------------------------------------------------


def register_metric_plugin(plugin: MetricsPlugin, *, run_scoped: bool = False) -> None:
    """Register a MetricsPlugin with the global metrics registry.

    Triggers lazy default registration on first call.

    Args:
        plugin: MetricsPlugin instance to register.
        run_scoped: If True, register in the active run_scope only.
    """
    _register_defaults()
    metrics_registry.register(plugin, run_scoped=run_scoped)


def register_visualization_plugin(
    plugin: VisualizationPlugin, *, run_scoped: bool = False
) -> None:
    """Register a VisualizationPlugin with the global visualization registry.

    Triggers lazy default registration on first call.

    Args:
        plugin: VisualizationPlugin instance to register.
        run_scoped: If True, register in the active run_scope only.
    """
    _register_defaults()
    visualization_registry.register(plugin, run_scoped=run_scoped)


def register_metadata_plugin(
    plugin: MetadataPlugin, *, run_scoped: bool = False
) -> None:
    """Register a MetadataPlugin with the global metadata registry.

    Triggers lazy default registration on first call.

    Args:
        plugin: MetadataPlugin instance to register.
        run_scoped: If True, register in the active run_scope only.
    """
    _register_defaults()
    metadata_registry.register(plugin, run_scoped=run_scoped)


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

__all__ = [
    # Base classes
    "MetricsPlugin",
    "VisualizationPlugin",
    "MetadataPlugin",
    # Registry singletons
    "metrics_registry",
    "visualization_registry",
    "metadata_registry",
    "PluginRegistry",
    # Registration helpers
    "register_metric_plugin",
    "register_visualization_plugin",
    "register_metadata_plugin",
    # Internal (for testing)
    "_register_defaults",
]
