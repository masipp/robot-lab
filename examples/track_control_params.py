"""
Example script demonstrating control parameter tracking for experiments.

This script shows how to:
1. Extract environment configuration (control & physics parameters)
2. Log it with ExperimentTracker
3. Verify the saved metadata

Usage:
    python examples/track_control_params.py
"""

from pathlib import Path
from robot_lab.experiments import ExperimentTracker
from robot_lab.utils.mujoco_config import (
    extract_environment_config,
    summarize_control_config,
)
import gymnasium as gym


def main():
    """Demonstrate control parameter tracking."""
    
    # 1. Create environment
    print("🔧 Creating A1Quadruped environment...")
    env = gym.make("A1Quadruped-v0")
    
    # 2. Extract full configuration
    print("\n📊 Extracting environment configuration...")
    env_config = extract_environment_config(env)
    
    # 3. Print summary
    print("\n" + "="*60)
    if "control_parameters" in env_config:
        print(summarize_control_config(env_config["control_parameters"]))
    print("="*60)
    
    # 4. Display key parameters
    print("\n🎯 Key Environment Parameters:")
    print(f"  Environment ID: {env_config.get('env_id')}")
    print(f"  Observation dim: {env_config['observation_space']['shape']}")
    print(f"  Action dim: {env_config['action_space']['shape']}")
    print(f"  Max episode steps: {env_config.get('max_episode_steps')}")
    
    if "physics_parameters" in env_config:
        phys = env_config["physics_parameters"]
        print(f"\n⚙️  Physics Parameters:")
        print(f"  Timestep: {phys.get('timestep')} sec")
        print(f"  Integrator: {phys.get('integrator')}")
        print(f"  Gravity: {phys.get('gravity')}")
    
    # 5. Create experiment tracker
    print("\n📝 Creating experiment tracker...")
    tracker = ExperimentTracker(
        experiment_name="example_control_tracking",
        run_name="demo_run",
        base_dir="data/experiments"
    )
    
    # 6. Log environment configuration
    print("💾 Logging environment configuration...")
    tracker.log_env_config(env_config)
    
    # 7. Also log some example hyperparameters
    tracker.log_params({
        "algorithm": "SAC",
        "learning_rate": 0.0003,
        "buffer_size": 300000,
        "total_timesteps": 100000,
    })
    
    # 8. Set experiment tags
    tracker.set_tag("experiment_id", "001")
    tracker.set_tag("hypothesis", "baseline_control")
    tracker.set_status("completed")
    
    # 9. Show where files were saved
    run_dir = tracker.get_run_dir()
    print(f"\n✅ Experiment metadata saved to: {run_dir}")
    print("\nSaved files:")
    for file in run_dir.glob("*.json"):
        print(f"  - {file.name}")
    
    # 10. Verify we can load it back
    print("\n🔍 Verifying saved metadata...")
    import json
    
    env_config_file = run_dir / "environment_config.json"
    if env_config_file.exists():
        with open(env_config_file) as f:
            loaded_config = json.load(f)
        
        num_actuators = loaded_config.get("control_parameters", {}).get("num_actuators", 0)
        timestep = loaded_config.get("physics_parameters", {}).get("timestep", "N/A")
        
        print(f"  ✓ Loaded config: {num_actuators} actuators, dt={timestep}")
    
    print("\n🎉 Done! You can now inspect the saved metadata files.")
    print(f"\nTo view control parameters:")
    print(f"  cat {run_dir}/environment_config.json | jq '.control_parameters'")
    print(f"\nTo view physics parameters:")
    print(f"  cat {run_dir}/environment_config.json | jq '.physics_parameters'")
    
    env.close()


if __name__ == "__main__":
    main()
