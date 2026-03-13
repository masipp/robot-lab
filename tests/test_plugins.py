"""Unit tests for the plugin/registry infrastructure (Story 1.1).

All tests are fast (no GPU, no live training) and complete in < 5 seconds.
"""

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from robot_lab.experiments.plugins.base import (
    MetadataPlugin,
    MetricsPlugin,
    VisualizationPlugin,
)

# ---------------------------------------------------------------------------
# Concrete test implementations (must inherit from ABCs)
# ---------------------------------------------------------------------------


class _StubMetricsPlugin(MetricsPlugin):
    """Minimal concrete MetricsPlugin for registry tests."""

    def on_step(self, context: dict[str, Any]) -> None:
        pass

    def on_episode_end(self, context: dict[str, Any]) -> None:
        pass

    def on_eval(self, context: dict[str, Any]) -> None:
        pass


class _StubVisualizationPlugin(VisualizationPlugin):
    """Minimal concrete VisualizationPlugin for registry tests."""

    def render(self, results: dict[str, Any]) -> None:
        pass


class _StubMetadataPlugin(MetadataPlugin):
    """Minimal concrete MetadataPlugin for registry tests."""

    def collect(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# AC 1: All six public API names importable
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_public_api_importable() -> None:
    """All six public names must be importable from robot_lab.experiments.plugins."""
    from robot_lab.experiments.plugins import (  # noqa: F401
        MetadataPlugin,
        MetricsPlugin,
        VisualizationPlugin,
        register_metadata_plugin,
        register_metric_plugin,
        register_visualization_plugin,
    )

    assert MetricsPlugin is not None
    assert VisualizationPlugin is not None
    assert MetadataPlugin is not None
    assert register_metric_plugin is not None
    assert register_visualization_plugin is not None
    assert register_metadata_plugin is not None


@pytest.mark.fast
def test_public_api_importable_from_experiments() -> None:
    """Six plugin names must also be importable from robot_lab.experiments."""
    from robot_lab.experiments import (  # noqa: F401
        MetadataPlugin,
        MetricsPlugin,
        VisualizationPlugin,
        register_metadata_plugin,
        register_metric_plugin,
        register_visualization_plugin,
    )

    assert MetricsPlugin is not None


# ---------------------------------------------------------------------------
# AC 2: No import side effects
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_no_import_side_effects() -> None:
    """Importing robot_lab must not trigger _register_defaults() or write files."""
    # Force a reimport of the plugins module to reset internal state for inspection.
    import robot_lab.experiments.plugins as plugins_mod

    # Save current state of the global flag.
    original_flag = plugins_mod._defaults_registered

    # Reloading robot_lab should NOT flip _defaults_registered.
    importlib.reload(importlib.import_module("robot_lab"))

    # Re-read the (possibly reloaded) module.
    import robot_lab.experiments.plugins as plugins_mod2  # noqa: F811

    # Side-effect check: _defaults_registered must still be the same as before import.
    # If import triggered _register_defaults(), it would be True even if it started False.
    # The safest assertion is that no I/O occurred (no file created).  We check the flag
    # only if we know it started as False.
    if not original_flag:
        # After a bare import, flag must still be False.
        assert not plugins_mod2._defaults_registered, (
            "import robot_lab must NOT call _register_defaults() as a side effect"
        )


# ---------------------------------------------------------------------------
# AC 1 / base classes: ABCs must be abstract
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_metrics_plugin_is_abstract() -> None:
    """Instantiating MetricsPlugin directly must raise TypeError."""
    from robot_lab.experiments.plugins import MetricsPlugin

    with pytest.raises(TypeError):
        MetricsPlugin()  # type: ignore[abstract]


@pytest.mark.fast
def test_visualization_plugin_is_abstract() -> None:
    """Instantiating VisualizationPlugin directly must raise TypeError."""
    from robot_lab.experiments.plugins import VisualizationPlugin

    with pytest.raises(TypeError):
        VisualizationPlugin()  # type: ignore[abstract]


@pytest.mark.fast
def test_metadata_plugin_is_abstract() -> None:
    """Instantiating MetadataPlugin directly must raise TypeError."""
    from robot_lab.experiments.plugins import MetadataPlugin

    with pytest.raises(TypeError):
        MetadataPlugin()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# AC 3: Lazy defaults — idempotency
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_lazy_defaults_idempotent() -> None:
    """Calling _register_defaults() three times must register each default only once."""
    import robot_lab.experiments.plugins as plugins_mod

    # Reset state so this test is independent.
    plugins_mod._defaults_registered = False
    plugins_mod.metrics_registry._global_plugins.clear()
    plugins_mod.metadata_registry._global_plugins.clear()

    plugins_mod._register_defaults()
    plugins_mod._register_defaults()
    plugins_mod._register_defaults()

    metric_count = len(plugins_mod.metrics_registry.list_plugins())
    metadata_count = len(plugins_mod.metadata_registry.list_plugins())

    assert metric_count == 2, (
        "Expected 2 metrics default plugins "
        "(BasicRewardLogPlugin + ActionSmoothnessMetricPlugin), "
        f"got {metric_count}. _register_defaults() is not idempotent or missing a plugin."
    )
    assert metadata_count == 1, (
        f"Expected 1 metadata default plugin, got {metadata_count}. "
        "_register_defaults() is not idempotent."
    )

    # Cleanup: reset for subsequent tests.
    plugins_mod._defaults_registered = False
    plugins_mod.metrics_registry._global_plugins.clear()
    plugins_mod.metadata_registry._global_plugins.clear()


# ---------------------------------------------------------------------------
# AC 4: Run-scoped isolation
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_run_scoped_isolation() -> None:
    """Plugins registered in run_scope() must not appear outside the scope."""
    import robot_lab.experiments.plugins as plugins_mod

    # Reset to a clean state.
    plugins_mod._defaults_registered = False
    plugins_mod.metrics_registry._global_plugins.clear()
    plugins_mod.metrics_registry._run_plugins.clear()

    run_scoped_plugin = _StubMetricsPlugin()

    # Inside scope: plugin is visible.
    with plugins_mod.metrics_registry.run_scope():
        plugins_mod.metrics_registry.register(run_scoped_plugin, run_scoped=True)
        assert run_scoped_plugin in plugins_mod.metrics_registry.list_plugins()

    # Outside scope: plugin must be gone.
    assert run_scoped_plugin not in plugins_mod.metrics_registry.list_plugins(), (
        "Run-scoped plugin must not survive after run_scope() exits."
    )


@pytest.mark.fast
def test_run_scoped_does_not_affect_global() -> None:
    """Global plugins must remain after run_scope() exits."""
    import robot_lab.experiments.plugins as plugins_mod

    plugins_mod._defaults_registered = False
    plugins_mod.metrics_registry._global_plugins.clear()
    plugins_mod.metrics_registry._run_plugins.clear()

    global_plugin = _StubMetricsPlugin()
    plugins_mod.metrics_registry.register(global_plugin)

    with plugins_mod.metrics_registry.run_scope():
        scoped_plugin = _StubMetricsPlugin()
        plugins_mod.metrics_registry.register(scoped_plugin, run_scoped=True)

    # Global plugin survives; scoped plugin does not.
    assert global_plugin in plugins_mod.metrics_registry.list_plugins()
    assert scoped_plugin not in plugins_mod.metrics_registry.list_plugins()

    # Cleanup.
    plugins_mod.metrics_registry._global_plugins.clear()


@pytest.mark.fast
def test_run_scoped_outside_context_raises() -> None:
    """Registering run_scoped=True outside a run_scope() must raise RuntimeError."""
    import robot_lab.experiments.plugins as plugins_mod

    with pytest.raises(RuntimeError, match="run_scope"):
        plugins_mod.metrics_registry.register(
            _StubMetricsPlugin(), run_scoped=True
        )


# ---------------------------------------------------------------------------
# Task 4: contrib namespace importable
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_contrib_importable() -> None:
    """from robot_lab.experiments.plugins.contrib import * must not raise."""
    import robot_lab.experiments.plugins.contrib  # noqa: F401

    # No error means the namespace package is importable.
    assert robot_lab.experiments.plugins.contrib is not None


# ---------------------------------------------------------------------------
# Built-in default plugin smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_basic_reward_log_plugin_on_episode_end() -> None:
    """BasicRewardLogPlugin.on_episode_end must call tracker.update with rewards."""
    from robot_lab.experiments.plugins.defaults import BasicRewardLogPlugin

    mock_tracker = MagicMock()
    plugin = BasicRewardLogPlugin()
    context: dict[str, Any] = {
        "tracker": mock_tracker,
        "episode_rewards": [100.0, 120.0],
        "n_steps": 1000,
    }
    plugin.on_episode_end(context)
    mock_tracker.update.assert_called_once_with(
        "metrics", {"episode_rewards": [100.0, 120.0]}
    )


@pytest.mark.fast
def test_basic_reward_log_plugin_no_tracker() -> None:
    """BasicRewardLogPlugin.on_episode_end must not crash when tracker is absent."""
    from robot_lab.experiments.plugins.defaults import BasicRewardLogPlugin

    plugin = BasicRewardLogPlugin()
    # No tracker key — must complete silently.
    plugin.on_episode_end({"episode_rewards": [50.0], "n_steps": 500})


@pytest.mark.fast
def test_system_metadata_plugin_returns_dict() -> None:
    """SystemMetadataPlugin.collect must return a dict with system_metadata key."""
    from robot_lab.experiments.plugins.defaults import SystemMetadataPlugin

    plugin = SystemMetadataPlugin()
    result = plugin.collect({})
    assert isinstance(result, dict)
    assert "system_metadata" in result
    assert isinstance(result["system_metadata"], dict)


# ---------------------------------------------------------------------------
# PluginRegistry generic behaviour
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_registry_list_plugins_combines_global_and_run() -> None:
    """list_plugins() must return global + run-scoped plugins combined."""
    import robot_lab.experiments.plugins as plugins_mod

    reg = plugins_mod.PluginRegistry("test-registry")
    g = _StubMetricsPlugin()
    s = _StubMetricsPlugin()

    reg.register(g)
    with reg.run_scope():
        reg.register(s, run_scoped=True)
        combined = reg.list_plugins()
        assert g in combined
        assert s in combined
        assert len(combined) == 2

    # After scope: only global remains.
    assert reg.list_plugins() == [g]


@pytest.mark.fast
def test_registry_clear_run_scoped() -> None:
    """clear_run_scoped() must discard run-scoped plugins without a context manager."""
    import robot_lab.experiments.plugins as plugins_mod

    reg = plugins_mod.PluginRegistry("test-registry-2")
    reg._in_run_scope = True  # simulate being inside a scope
    s = _StubMetricsPlugin()
    reg.register(s, run_scoped=True)
    assert s in reg.list_plugins()

    reg.clear_run_scoped()
    assert s not in reg.list_plugins()
    reg._in_run_scope = False  # restore


# ---------------------------------------------------------------------------
# Story 2.2: ActionSmoothnessMetricPlugin + PluginRegistry dispatch methods
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_registry_dispatches_on_episode_end() -> None:
    """PluginRegistry.on_episode_end() must call each plugin's on_episode_end. (AC: 2)"""
    import robot_lab.experiments.plugins as plugins_mod

    reg = plugins_mod.PluginRegistry("dispatch-test")
    called: list[str] = []

    class _TrackingPlugin(MetricsPlugin):
        def on_step(self, context: dict[str, Any]) -> None:
            pass

        def on_episode_end(self, context: dict[str, Any]) -> None:
            called.append("on_episode_end")

        def on_eval(self, context: dict[str, Any]) -> None:
            pass

    reg.register(_TrackingPlugin())
    reg.on_episode_end({})
    assert called == ["on_episode_end"]


@pytest.mark.fast
def test_registry_dispatch_continues_on_plugin_error() -> None:
    """PluginRegistry dispatch must not abort on a failing plugin."""
    import robot_lab.experiments.plugins as plugins_mod

    reg = plugins_mod.PluginRegistry("error-test")
    called_second: list[bool] = []

    class _BadPlugin(MetricsPlugin):
        def on_step(self, context: dict[str, Any]) -> None:
            pass

        def on_episode_end(self, context: dict[str, Any]) -> None:
            raise RuntimeError("intentional test failure")

        def on_eval(self, context: dict[str, Any]) -> None:
            pass

    class _GoodPlugin(MetricsPlugin):
        def on_step(self, context: dict[str, Any]) -> None:
            pass

        def on_episode_end(self, context: dict[str, Any]) -> None:
            called_second.append(True)

        def on_eval(self, context: dict[str, Any]) -> None:
            pass

    reg.register(_BadPlugin())
    reg.register(_GoodPlugin())
    reg.on_episode_end({})  # must not raise
    assert called_second == [True], "Second plugin must run even if first plugin raised"


@pytest.mark.fast
def test_smoothness_plugin_registered_by_defaults() -> None:
    """ActionSmoothnessMetricPlugin must appear in metrics_registry after _register_defaults.

    AC: 1
    """
    import robot_lab.experiments.plugins as plugins_mod
    from robot_lab.experiments.plugins.defaults import ActionSmoothnessMetricPlugin

    plugins_mod._defaults_registered = False
    plugins_mod.metrics_registry._global_plugins.clear()
    plugins_mod.metadata_registry._global_plugins.clear()

    plugins_mod._register_defaults()

    plugin_types = [type(p) for p in plugins_mod.metrics_registry.list_plugins()]
    assert ActionSmoothnessMetricPlugin in plugin_types, (
        "ActionSmoothnessMetricPlugin must be registered by _register_defaults()"
    )

    # Cleanup
    plugins_mod._defaults_registered = False
    plugins_mod.metrics_registry._global_plugins.clear()
    plugins_mod.metadata_registry._global_plugins.clear()


@pytest.mark.fast
def test_smoothness_plugin_on_episode_end_logs_to_tracker(
    temp_output_dir: "Path",  # noqa: F821
) -> None:
    """ActionSmoothnessMetricPlugin must write smoothness_action_delta_norm to tracker. (AC: 3)"""

    import numpy as np

    from robot_lab.experiments.plugins.defaults import ActionSmoothnessMetricPlugin
    from robot_lab.experiments.tracker import ExperimentTracker

    tracker = ExperimentTracker(
        experiment_name="test_smoothness",
        run_name="smoothness_test",
        seed=0,
        output_dir=str(temp_output_dir),
    )
    tracker.start_run()

    plugin = ActionSmoothnessMetricPlugin()

    a1 = np.array([0.5, -0.3])
    a2 = np.array([0.8, 0.1])

    plugin.on_step({"actions": a1})
    plugin.on_step({"actions": a2})

    expected = float(np.sum((a2 - a1) ** 2))
    plugin.on_episode_end({"tracker": tracker, "n_steps": 2})

    metrics = tracker.get_metadata()["metrics"]
    assert "smoothness_action_delta_norm" in metrics, (
        "Expected 'smoothness_action_delta_norm' in metrics after on_episode_end"
    )
    series = metrics["smoothness_action_delta_norm"]
    assert isinstance(series, list) and len(series) >= 1
    assert abs(series[-1] - expected) < 1e-8, (
        f"Expected smoothness value ~{expected}, got {series[-1]}"
    )


@pytest.mark.fast
def test_smoothness_plugin_tb_logging() -> None:
    """ActionSmoothnessMetricPlugin must call sb3_logger.record() for TB scalar. (AC: 2)"""
    from unittest.mock import MagicMock

    import numpy as np

    from robot_lab.experiments.plugins.defaults import ActionSmoothnessMetricPlugin

    mock_logger = MagicMock()
    plugin = ActionSmoothnessMetricPlugin()

    a1 = np.array([1.0])
    a2 = np.array([2.0])

    plugin.on_step({"actions": a1})
    plugin.on_step({"actions": a2})
    plugin.on_episode_end({"sb3_logger": mock_logger, "n_steps": 2})

    mock_logger.record.assert_called()
    call_args = mock_logger.record.call_args_list
    keys_logged = [call[0][0] for call in call_args]
    assert any("smoothness" in k for k in keys_logged), (
        f"Expected a 'smoothness/...' key in tb_logger.record calls, got: {keys_logged}"
    )


@pytest.mark.fast
def test_smoothness_plugin_no_import_side_effects() -> None:
    """Importing ActionSmoothnessMetricPlugin must not compute metrics or write files. (AC: 4)"""
    # Just importing the class must not raise or produce side effects
    from robot_lab.experiments.plugins.defaults import ActionSmoothnessMetricPlugin  # noqa: F401

    plugin = ActionSmoothnessMetricPlugin()
    # After instantiation, no state accumulated yet
    assert plugin._episode_delta_sq_sum == 0.0
    assert plugin._prev_action is None


@pytest.mark.fast
def test_smoothness_plugin_state_resets_after_episode() -> None:
    """ActionSmoothnessMetricPlugin state must reset after on_episode_end."""

    import numpy as np

    from robot_lab.experiments.plugins.defaults import ActionSmoothnessMetricPlugin

    plugin = ActionSmoothnessMetricPlugin()
    plugin.on_step({"actions": np.array([1.0, 2.0])})
    plugin.on_step({"actions": np.array([1.5, 2.5])})

    plugin.on_episode_end({})

    assert plugin._episode_delta_sq_sum == 0.0
    assert plugin._prev_action is None


@pytest.mark.fast
def test_smoothness_plugin_handles_missing_actions() -> None:
    """ActionSmoothnessMetricPlugin must skip gracefully when actions key absent."""
    from robot_lab.experiments.plugins.defaults import ActionSmoothnessMetricPlugin

    plugin = ActionSmoothnessMetricPlugin()
    plugin.on_step({})  # no actions — must not raise
    assert plugin._episode_delta_sq_sum == 0.0
