"""Example experiment specification templates."""

HYPERPARAM_SWEEP_TEMPLATE = {
    "experiment_metadata": {
        "name": "walker2d_sac_hyperparam_sweep",
        "description": "Systematic hyperparameter search for SAC on Walker2d",
        "created_by": "user",
        "tags": ["hyperparameter_search", "sac", "walker2d"]
    },
    "environments": [
        {
            "name": "Walker2d-v5",
            "config_overrides": {}
        }
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
            "values": [0.0001, 0.0003, 0.001, 0.003],
            "type": "discrete"
        },
        {
            "parameter": "batch_size",
            "values": [64, 128, 256, 512],
            "type": "discrete"
        }
    ],
    "training_config": {
        "num_seeds": 5,
        "seeds": [42, 123, 456, 789, 1011],
        "total_timesteps": 500000,
        "num_envs": 8,
        "eval_frequency": 10000,
        "eval_episodes": 10,
        "save_freq": 50000,
        "checkpoint_best": True
    },
    "evaluation_criteria": {
        "primary_metric": "mean_reward",
        "secondary_metrics": ["episode_length"],
        "aggregation": "mean_last_100",
        "comparison_method": "simple_comparison",
        "confidence_level": 0.95
    },
    "resource_limits": {
        "max_concurrent_runs": 4,
        "max_total_runs": 100,
        "max_runtime_hours": 48,
        "gpu_required": False
    },
    "output_config": {
        "base_dir": "experiments/walker2d_sac_sweep",
        "save_models": True,
        "save_logs": True,
        "generate_report": True,
        "tensorboard": True,
        "mlflow_tracking": False,
        "wandb_tracking": False
    }
}


ALGORITHM_COMPARISON_TEMPLATE = {
    "experiment_metadata": {
        "name": "locomotion_algo_comparison",
        "description": "Compare SAC and PPO on locomotion tasks",
        "created_by": "user",
        "tags": ["algorithm_comparison", "locomotion"]
    },
    "environments": [
        {"name": "Walker2d-v5", "config_overrides": {}},
        {"name": "HalfCheetah-v5", "config_overrides": {}}
    ],
    "algorithms": ["SAC", "PPO"],
    "base_hyperparameters": {
        "SAC": {
            "source": "configs/default_sac.json",
            "overrides": {}
        },
        "PPO": {
            "source": "configs/default_ppo.json",
            "overrides": {}
        }
    },
    "hyperparameter_sweeps": [],
    "training_config": {
        "num_seeds": 5,
        "total_timesteps": 500000,
        "num_envs": 8,
        "eval_frequency": 10000,
        "eval_episodes": 10,
        "checkpoint_best": True
    },
    "evaluation_criteria": {
        "primary_metric": "mean_reward",
        "secondary_metrics": ["episode_length", "training_time"],
        "aggregation": "mean_last_100",
        "comparison_method": "statistical_test",
        "confidence_level": 0.95
    },
    "resource_limits": {
        "max_concurrent_runs": 4,
        "gpu_required": False
    },
    "output_config": {
        "base_dir": "experiments/locomotion_comparison",
        "save_models": True,
        "save_logs": True,
        "generate_report": True,
        "tensorboard": True
    }
}


QUICK_TEST_TEMPLATE = {
    "experiment_metadata": {
        "name": "quick_test",
        "description": "Quick test run with minimal training",
        "created_by": "user",
        "tags": ["test"]
    },
    "environments": [
        {"name": "MountainCarContinuous-v0", "config_overrides": {}}
    ],
    "algorithms": ["SAC"],
    "base_hyperparameters": {
        "SAC": {
            "source": "configs/mountaincarcontinuous_sac.json",
            "overrides": {}
        }
    },
    "hyperparameter_sweeps": [],
    "training_config": {
        "num_seeds": 1,
        "seeds": [42],
        "total_timesteps": 20000,
        "num_envs": 4,
        "eval_frequency": 5000,
        "eval_episodes": 5,
        "checkpoint_best": False
    },
    "evaluation_criteria": {
        "primary_metric": "mean_reward",
        "aggregation": "mean_last_10"
    },
    "resource_limits": {
        "max_concurrent_runs": 1,
        "gpu_required": False
    },
    "output_config": {
        "base_dir": "experiments/quick_test",
        "save_models": True,
        "save_logs": False,
        "generate_report": False,
        "tensorboard": False
    }
}


def get_template(template_name: str) -> dict:
    """Get an experiment template by name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template dictionary
        
    Raises:
        ValueError: If template name is unknown
    """
    templates = {
        "hyperparam_sweep": HYPERPARAM_SWEEP_TEMPLATE,
        "algorithm_comparison": ALGORITHM_COMPARISON_TEMPLATE,
        "quick_test": QUICK_TEST_TEMPLATE
    }
    
    if template_name not in templates:
        available = ", ".join(templates.keys())
        raise ValueError(
            f"Unknown template: {template_name}. Available: {available}"
        )
    
    return templates[template_name].copy()
