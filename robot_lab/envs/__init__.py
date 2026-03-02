"""Custom environment registration and registry for robot_lab.

This module provides centralized environment management including:
- Environment registration with Gymnasium
- Environment metadata and discovery
- Centralized environment creation via registry

Available custom environments:
- GripperEnv-v0: MuJoCo-based gripper manipulation
- A1Quadruped-v0: Unitree A1 quadruped locomotion
"""

from robot_lab.envs.registry import (
    get_env_registry,
    make_env,
    get_env_info,
    EnvCategory,
    EnvDifficulty,
    EnvMetadata,
    EnvironmentRegistry,
)

# Import custom environment classes for direct use
from robot_lab.envs.manipulation import GripperEnv
from robot_lab.envs.locomotion import A1QuadrupedEnv, MuJoCoLocomotionWrapper


def register_custom_envs():
    """Register all custom environments with gymnasium.
    
    This function is called automatically on package import.
    Uses the centralized registry for registration.
    """
    registry = get_env_registry()
    registry.register_custom_envs()


__all__ = [
    # Registration
    "register_custom_envs",
    
    # Registry system
    "get_env_registry",
    "make_env",
    "get_env_info",
    "EnvironmentRegistry",
    "EnvCategory",
    "EnvDifficulty",
    "EnvMetadata",
    
    # Custom environment classes (for direct instantiation)
    "GripperEnv",
    "A1QuadrupedEnv",
    "MuJoCoLocomotionWrapper",
]
