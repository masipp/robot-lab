"""Utilities for extracting MuJoCo configuration parameters.

This module provides functions to extract control and physics parameters
from MuJoCo models for experiment tracking and reproducibility.
"""

from typing import Dict, Any, Optional, List
import numpy as np


def extract_control_parameters(model: Any) -> Dict[str, Any]:
    """Extract control parameters from a MuJoCo model.
    
    Args:
        model: MuJoCo MjModel instance
        
    Returns:
        Dictionary of control parameters including:
        - actuator types and gains
        - control ranges
        - joint properties (armature, damping)
    """
    try:
        import mujoco
    except ImportError:
        return {"error": "MuJoCo not available"}
    
    control_params = {
        "num_actuators": model.nu,
        "actuators": [],
        "joints": [],
    }
    
    # Extract actuator information
    for i in range(model.nu):
        actuator_info = {
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "type": _get_actuator_type(model, i),
            "gear": model.actuator_gear[i].tolist(),
            "ctrl_range": model.actuator_ctrlrange[i].tolist(),
        }
        
        # Extract gains for position/velocity actuators
        if model.actuator_gainprm[i, 0] != 0:
            actuator_info["kp"] = float(model.actuator_gainprm[i, 0])
        if model.actuator_biasprm[i, 1] != 0:
            actuator_info["kd"] = float(model.actuator_biasprm[i, 1])
        
        control_params["actuators"].append(actuator_info)
    
    # Extract joint properties
    for i in range(model.njnt):
        joint_info = {
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
            "type": _get_joint_type(model, i),
            "armature": float(model.jnt_armature[i]) if i < len(model.jnt_armature) else None,
            "damping": float(model.dof_damping[i]) if i < len(model.dof_damping) else None,
            "range": model.jnt_range[i].tolist() if model.jnt_limited[i] else None,
        }
        control_params["joints"].append(joint_info)
    
    return control_params


def extract_physics_parameters(model: Any) -> Dict[str, Any]:
    """Extract physics simulation parameters from a MuJoCo model.
    
    Args:
        model: MuJoCo MjModel instance
        
    Returns:
        Dictionary of physics parameters
    """
    try:
        import mujoco
    except ImportError:
        return {"error": "MuJoCo not available"}
    
    physics_params = {
        "timestep": float(model.opt.timestep),
        "gravity": model.opt.gravity.tolist(),
        "integrator": _get_integrator_name(model.opt.integrator),
        "solver": _get_solver_name(model.opt.solver),
        "iterations": int(model.opt.iterations),
        "tolerance": float(model.opt.tolerance),
        "magnetic": model.opt.magnetic.tolist(),
        "wind": model.opt.wind.tolist(),
        "density": float(model.opt.density),
        "viscosity": float(model.opt.viscosity),
    }
    
    return physics_params


def extract_environment_config(env: Any) -> Dict[str, Any]:
    """Extract configuration from a Gymnasium environment.
    
    Args:
        env: Gymnasium environment instance
        
    Returns:
        Dictionary of environment configuration
    """
    config = {
        "env_id": getattr(env, "spec", None).id if hasattr(env, "spec") else None,
        "observation_space": {
            "shape": env.observation_space.shape,
            "dtype": str(env.observation_space.dtype),
        },
        "action_space": {
            "shape": env.action_space.shape,
            "dtype": str(env.action_space.dtype),
        },
    }
    
    # Try to get action bounds
    if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
        config["action_space"]["low"] = env.action_space.low.tolist()
        config["action_space"]["high"] = env.action_space.high.tolist()
    
    # Try to extract max_episode_steps
    if hasattr(env, "spec") and env.spec is not None:
        config["max_episode_steps"] = env.spec.max_episode_steps
    elif hasattr(env, "_max_episode_steps"):
        config["max_episode_steps"] = env._max_episode_steps
    elif hasattr(env, "max_steps"):
        config["max_episode_steps"] = env.max_steps
    
    # Try to get MuJoCo-specific parameters
    if hasattr(env, "model"):
        config["control_parameters"] = extract_control_parameters(env.model)
        config["physics_parameters"] = extract_physics_parameters(env.model)
        
        # Physics steps per action (if available)
        if hasattr(env, "frame_skip"):
            config["physics_steps_per_action"] = env.frame_skip
    
    return config


def summarize_control_config(control_params: Dict[str, Any]) -> str:
    """Create a human-readable summary of control configuration.
    
    Args:
        control_params: Control parameters dictionary
        
    Returns:
        Formatted string summary
    """
    lines = ["Control Configuration Summary:"]
    lines.append(f"  Actuators: {control_params.get('num_actuators', 0)}")
    
    # Summarize actuator types
    actuators = control_params.get("actuators", [])
    if actuators:
        actuator_types = {}
        for act in actuators:
            act_type = act.get("type", "unknown")
            actuator_types[act_type] = actuator_types.get(act_type, 0) + 1
        
        for act_type, count in actuator_types.items():
            lines.append(f"    - {act_type}: {count}")
        
        # Show gains if available
        example_act = actuators[0]
        if "kp" in example_act:
            kp_values = [a.get("kp") for a in actuators if "kp" in a]
            if kp_values:
                lines.append(f"  Position gains (kp): min={min(kp_values)}, max={max(kp_values)}")
        if "kd" in example_act:
            kd_values = [a.get("kd") for a in actuators if "kd" in a]
            if kd_values:
                lines.append(f"  Derivative gains (kd): min={min(kd_values)}, max={max(kd_values)}")
    
    # Summarize joints
    joints = control_params.get("joints", [])
    if joints:
        lines.append(f"  Joints: {len(joints)}")
        dampings = [j.get("damping") for j in joints if j.get("damping") is not None]
        if dampings:
            lines.append(f"    Damping: min={min(dampings):.3f}, max={max(dampings):.3f}")
        armatures = [j.get("armature") for j in joints if j.get("armature") is not None]
        if armatures:
            lines.append(f"    Armature: min={min(armatures):.3f}, max={max(armatures):.3f}")
    
    return "\n".join(lines)


# Helper functions for type conversion

def _get_actuator_type(model: Any, actuator_id: int) -> str:
    """Get the type of actuator as a string."""
    try:
        import mujoco
        actuator_type = model.actuator_dyntype[actuator_id]
        
        # Map MuJoCo actuator types to readable names
        type_map = {
            mujoco.mjtDyn.mjDYN_NONE: "position",
            mujoco.mjtDyn.mjDYN_INTEGRATOR: "velocity", 
            mujoco.mjtDyn.mjDYN_FILTER: "filter",
            mujoco.mjtDyn.mjDYN_MUSCLE: "muscle",
            mujoco.mjtDyn.mjDYN_USER: "user",
        }
        
        # Also check gain type for position servos
        if model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED:
            return "position_servo"
        
        return type_map.get(actuator_type, f"unknown({actuator_type})")
    except:
        return "unknown"


def _get_joint_type(model: Any, joint_id: int) -> str:
    """Get the type of joint as a string."""
    try:
        import mujoco
        joint_type = model.jnt_type[joint_id]
        
        type_map = {
            mujoco.mjtJoint.mjJNT_FREE: "free",
            mujoco.mjtJoint.mjJNT_BALL: "ball",
            mujoco.mjtJoint.mjJNT_SLIDE: "slide",
            mujoco.mjtJoint.mjJNT_HINGE: "hinge",
        }
        
        return type_map.get(joint_type, f"unknown({joint_type})")
    except:
        return "unknown"


def _get_integrator_name(integrator_id: int) -> str:
    """Get integrator name from MuJoCo ID."""
    try:
        import mujoco
        integrator_map = {
            mujoco.mjtIntegrator.mjINT_EULER: "euler",
            mujoco.mjtIntegrator.mjINT_RK4: "rk4",
            mujoco.mjtIntegrator.mjINT_IMPLICIT: "implicit",
        }
        return integrator_map.get(integrator_id, f"unknown({integrator_id})")
    except:
        return "unknown"


def _get_solver_name(solver_id: int) -> str:
    """Get solver name from MuJoCo ID."""
    try:
        import mujoco
        solver_map = {
            mujoco.mjtSolver.mjSOL_PGS: "pgs",
            mujoco.mjtSolver.mjSOL_CG: "cg",
            mujoco.mjtSolver.mjSOL_NEWTON: "newton",
        }
        return solver_map.get(solver_id, f"unknown({solver_id})")
    except:
        return "unknown"
