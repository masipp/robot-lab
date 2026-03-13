"""Experiment automation infrastructure for robot_lab."""

from robot_lab.experiments.schemas import (
    ExperimentSpec,
    RunResult,
    RunMetadata,
    RunMetrics,
    ExperimentMetadata,
    TrainingConfig,
)
from robot_lab.experiments.tracker import ExperimentTracker
from robot_lab.experiments.results_db import ResultsDatabase
from robot_lab.experiments.spec_templates import get_template
from robot_lab.experiments.ai_planner import AIExperimentPlanner
from robot_lab.experiments.plugins import (
    MetricsPlugin,
    VisualizationPlugin,
    MetadataPlugin,
    register_metric_plugin,
    register_visualization_plugin,
    register_metadata_plugin,
)

__all__ = [
    "ExperimentSpec",
    "RunResult",
    "RunMetadata",
    "RunMetrics",
    "ExperimentMetadata",
    "TrainingConfig",
    "ExperimentTracker",
    "ResultsDatabase",
    "get_template",
    "AIExperimentPlanner",
    # Plugin public API
    "MetricsPlugin",
    "VisualizationPlugin",
    "MetadataPlugin",
    "register_metric_plugin",
    "register_visualization_plugin",
    "register_metadata_plugin",
]
