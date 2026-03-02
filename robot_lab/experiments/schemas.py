"""Pydantic schemas for experiment specifications and validation."""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator, field_validator, model_validator


class ExperimentMetadata(BaseModel):
    """Metadata for an experiment campaign."""
    name: str = Field(..., description="Unique experiment name")
    description: str = Field(..., description="Human-readable description")
    created_by: str = Field(default="user", description="Who created this experiment")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of creation"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class EnvironmentConfig(BaseModel):
    """Configuration for a single environment."""
    name: str = Field(..., description="Gymnasium environment name")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides for environment parameters"
    )
    control_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Control-specific parameters (kp, kd, actuator_type, etc.)"
    )
    physics_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Physics simulation parameters (timestep, gravity, friction, etc.)"
    )


class HyperparameterOverrides(BaseModel):
    """Hyperparameter overrides for an algorithm."""
    source: str = Field(..., description="Path to base config file")
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific parameter overrides"
    )


class HyperparameterSweep(BaseModel):
    """Definition of a hyperparameter sweep."""
    parameter: str = Field(..., description="Name of parameter to sweep")
    values: Optional[List[Any]] = Field(None, description="Discrete values to try")
    range: Optional[List[float]] = Field(None, description="Continuous range [min, max]")
    type: Literal["discrete", "continuous"] = Field(..., description="Sweep type")
    sampling: Optional[Literal["uniform", "log_uniform", "grid"]] = Field(
        "uniform",
        description="Sampling strategy for continuous"
    )
    num_samples: Optional[int] = Field(None, description="Number of samples for continuous")
    
    @field_validator('values')
    @classmethod
    def values_for_discrete(cls, v, info):
        """Validate that discrete sweeps have values."""
        if info.data.get('type') == 'discrete' and not v:
            raise ValueError("Discrete sweeps must specify 'values'")
        return v
    
    @model_validator(mode='after')
    def validate_continuous_fields(self):
        """Validate that continuous sweeps have range and num_samples."""
        if self.type == 'continuous':
            if not self.range:
                raise ValueError("Continuous sweeps must specify 'range'")
            if not self.num_samples:
                raise ValueError("Continuous sweeps must specify 'num_samples'")
        return self


class TrainingConfig(BaseModel):
    """Configuration for training runs."""
    num_seeds: int = Field(default=1, ge=1, description="Number of random seeds to use")
    seeds: Optional[List[int]] = Field(None, description="Specific seeds to use")
    total_timesteps: int = Field(default=100000, ge=1000, description="Training timesteps")
    num_envs: int = Field(default=8, ge=1, description="Number of parallel environments")
    eval_frequency: int = Field(default=10000, ge=1, description="Evaluation frequency")
    eval_episodes: int = Field(default=10, ge=1, description="Episodes per evaluation")
    save_freq: Optional[int] = Field(None, description="Checkpoint save frequency")
    checkpoint_best: bool = Field(default=True, description="Save best model")
    
    @validator('seeds')
    def validate_seeds(cls, v, values):
        """Ensure seeds list matches num_seeds."""
        if v is not None and len(v) != values.get('num_seeds', 1):
            raise ValueError(f"Length of 'seeds' must match 'num_seeds'")
        return v


class EvaluationCriteria(BaseModel):
    """Criteria for evaluating experiment results."""
    primary_metric: str = Field(default="mean_reward", description="Primary metric to optimize")
    secondary_metrics: List[str] = Field(
        default_factory=list,
        description="Additional metrics to track"
    )
    aggregation: Literal["mean_last_10", "mean_last_100", "max", "mean_all"] = Field(
        default="mean_last_100",
        description="How to aggregate episode results"
    )
    comparison_method: Literal["statistical_test", "simple_comparison"] = Field(
        default="simple_comparison",
        description="Method for comparing configurations"
    )
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)


class ResourceLimits(BaseModel):
    """Resource limits for experiment execution."""
    max_concurrent_runs: int = Field(default=4, ge=1, description="Max parallel runs")
    max_total_runs: Optional[int] = Field(None, description="Max total runs")
    max_runtime_hours: Optional[float] = Field(None, description="Max runtime in hours")
    gpu_required: bool = Field(default=False, description="Whether GPU is required")


class OutputConfig(BaseModel):
    """Configuration for experiment outputs."""
    base_dir: str = Field(default="experiments", description="Base experiments directory")
    save_models: bool = Field(default=True, description="Save trained models")
    save_logs: bool = Field(default=True, description="Save TensorBoard logs")
    generate_report: bool = Field(default=True, description="Generate markdown report")
    tensorboard: bool = Field(default=True, description="Enable TensorBoard logging")
    mlflow_tracking: bool = Field(default=False, description="Enable MLflow tracking")
    wandb_tracking: bool = Field(default=False, description="Enable W&B tracking")


class ExperimentSpec(BaseModel):
    """Complete experiment specification."""
    experiment_metadata: ExperimentMetadata
    environments: List[EnvironmentConfig] = Field(..., min_items=1)
    algorithms: List[Literal["SAC", "PPO"]] = Field(..., min_items=1)
    base_hyperparameters: Dict[str, HyperparameterOverrides]
    hyperparameter_sweeps: List[HyperparameterSweep] = Field(default_factory=list)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    output_config: OutputConfig = Field(default_factory=OutputConfig)
    
    @validator('base_hyperparameters')
    def validate_algo_hyperparams(cls, v, values):
        """Ensure hyperparameters are defined for all algorithms."""
        algorithms = values.get('algorithms', [])
        for algo in algorithms:
            if algo not in v:
                raise ValueError(f"Missing hyperparameters for algorithm: {algo}")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "experiment_metadata": {
                    "name": "walker2d_sac_sweep",
                    "description": "SAC hyperparameter sweep on Walker2d",
                    "created_by": "AI_Researcher",
                    "tags": ["sac", "walker2d", "hyperparam_search"]
                },
                "environments": [
                    {"name": "Walker2d-v5", "config_overrides": {}}
                ],
                "algorithms": ["SAC"],
                "base_hyperparameters": {
                    "SAC": {
                        "source": "configs/walker2d_sac.json",
                        "overrides": {}
                    }
                },
                "hyperparameter_sweeps": [
                    {
                        "parameter": "learning_rate",
                        "values": [0.0001, 0.0003, 0.001],
                        "type": "discrete"
                    }
                ],
                "training_config": {
                    "num_seeds": 3,
                    "total_timesteps": 500000
                }
            }
        }


class RunMetadata(BaseModel):
    """Metadata for a single training run."""
    run_id: str
    experiment_id: str
    environment: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    seed: int
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class RunMetrics(BaseModel):
    """Metrics collected during a training run."""
    final_mean_reward: Optional[float] = None
    final_std_reward: Optional[float] = None
    best_mean_reward: Optional[float] = None
    best_timestep: Optional[int] = None
    training_timesteps: int
    wall_clock_time: Optional[float] = None
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Complete result of a training run."""
    metadata: RunMetadata
    metrics: RunMetrics
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths to generated files (model, logs, etc.)"
    )
