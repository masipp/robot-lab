"""robot_lab: Reinforcement learning playground for robotic environments."""

__version__ = "0.1.0"

from robot_lab.envs import register_custom_envs

# Register custom environments on import
register_custom_envs()
