"""Namespace package for Marco's phase-specific plugin implementations.

Place experiment-specific plugin modules here, e.g.:

    robot_lab/experiments/plugins/contrib/curriculum_metrics.py
    robot_lab/experiments/plugins/contrib/smoothness_metrics.py

Import them in experiment scripts and register explicitly:

    from robot_lab.experiments.plugins.contrib.smoothness_metrics import (
        ActionSmoothnessMetricPlugin,
    )
    from robot_lab.experiments.plugins import register_metric_plugin
    register_metric_plugin(ActionSmoothnessMetricPlugin())
"""
