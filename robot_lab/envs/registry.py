"""Centralized environment registry and factory for robot_lab.

This module provides:
- Environment metadata and categorization
- Centralized environment creation
- Environment validation and discovery

Environment metadata is loaded from JSON configuration files:
- builtin_envs.json: Standard Gymnasium environments
- custom_envs.json: Custom robot_lab environments

To add a new environment, simply add an entry to the appropriate JSON file.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import gymnasium as gym
from gymnasium.envs.registration import register
from loguru import logger
from importlib.resources import files


class EnvCategory(str, Enum):
    """Environment categories for organization."""
    CLASSIC_CONTROL = "classic_control"
    LOCOMOTION = "locomotion"
    MANIPULATION = "manipulation"
    MUJOCO = "mujoco"
    CUSTOM = "custom"


class EnvDifficulty(str, Enum):
    """Environment difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class EnvMetadata:
    """Metadata for an environment."""
    env_id: str
    category: EnvCategory
    difficulty: EnvDifficulty
    description: str
    observation_space_desc: str
    action_space_desc: str
    is_custom: bool = False
    requires_mujoco: bool = False
    requires_robot_descriptions: bool = False
    default_algorithm: str = "SAC"
    recommended_timesteps: int = 100000
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class EnvironmentRegistry:
    """Central registry for all available environments.
    
    Loads environment metadata from JSON configuration files.
    """
    
    def __init__(self):
        self._registry: Dict[str, EnvMetadata] = {}
        self._custom_envs_registered = False
        self._initialize_builtin_envs()
    
    def _load_env_metadata_from_json(self, json_filename: str) -> List[Dict[str, Any]]:
        """Load environment metadata from a JSON file in the configs directory.
        
        Args:
            json_filename: Name of the JSON file (e.g., 'builtin_envs.json')
        
        Returns:
            List of environment metadata dictionaries
        """
        try:
            # Use importlib.resources to access bundled config files
            config_files = files('robot_lab').joinpath('configs')
            config_text = config_files.joinpath(json_filename).read_text()
            data = json.loads(config_text)
            return data.get('environments', [])
        except Exception as e:
            logger.warning(f"Failed to load {json_filename}: {e}")
            return []
    
    def _initialize_builtin_envs(self):
        """Initialize metadata for built-in Gymnasium environments from JSON."""
        env_data_list = self._load_env_metadata_from_json('builtin_envs.json')
        
        for env_data in env_data_list:
            try:
                metadata = EnvMetadata(
                    env_id=env_data['env_id'],
                    category=EnvCategory(env_data['category']),
                    difficulty=EnvDifficulty(env_data['difficulty']),
                    description=env_data['description'],
                    observation_space_desc=env_data['observation_space_desc'],
                    action_space_desc=env_data['action_space_desc'],
                    is_custom=env_data.get('is_custom', False),
                    requires_mujoco=env_data.get('requires_mujoco', False),
                    requires_robot_descriptions=env_data.get('requires_robot_descriptions', False),
                    default_algorithm=env_data.get('default_algorithm', 'SAC'),
                    recommended_timesteps=env_data.get('recommended_timesteps', 100000),
                    tags=env_data.get('tags', [])
                )
                self.register_metadata(metadata)
                logger.debug(f"Loaded metadata for {metadata.env_id}")
            except Exception as e:
                logger.warning(f"Failed to load metadata for {env_data.get('env_id', 'unknown')}: {e}")
    
    def register_custom_envs(self):
        """Register all custom robot_lab environments with Gymnasium.
        
        Loads environment definitions from custom_envs.json and registers
        them with Gymnasium.
        """
        if self._custom_envs_registered:
            return
        
        # Load custom environment definitions from JSON
        custom_env_data_list = self._load_env_metadata_from_json('custom_envs.json')
        
        for env_data in custom_env_data_list:
            try:
                # Register with Gymnasium
                register(
                    id=env_data['env_id'],
                    entry_point=env_data['entry_point'],
                    max_episode_steps=env_data.get('max_episode_steps', 1000),
                )
                
                # Register metadata
                metadata = EnvMetadata(
                    env_id=env_data['env_id'],
                    category=EnvCategory(env_data['category']),
                    difficulty=EnvDifficulty(env_data['difficulty']),
                    description=env_data['description'],
                    observation_space_desc=env_data['observation_space_desc'],
                    action_space_desc=env_data['action_space_desc'],
                    is_custom=env_data.get('is_custom', True),
                    requires_mujoco=env_data.get('requires_mujoco', False),
                    requires_robot_descriptions=env_data.get('requires_robot_descriptions', False),
                    default_algorithm=env_data.get('default_algorithm', 'SAC'),
                    recommended_timesteps=env_data.get('recommended_timesteps', 100000),
                    tags=env_data.get('tags', [])
                )
                self.register_metadata(metadata)
                logger.debug(f"✓ Registered {env_data['env_id']}")
            except gym.error.Error:
                # Already registered - just update metadata
                logger.debug(f"  {env_data['env_id']} already registered")
                try:
                    metadata = EnvMetadata(
                        env_id=env_data['env_id'],
                        category=EnvCategory(env_data['category']),
                        difficulty=EnvDifficulty(env_data['difficulty']),
                        description=env_data['description'],
                        observation_space_desc=env_data['observation_space_desc'],
                        action_space_desc=env_data['action_space_desc'],
                        is_custom=env_data.get('is_custom', True),
                        requires_mujoco=env_data.get('requires_mujoco', False),
                        requires_robot_descriptions=env_data.get('requires_robot_descriptions', False),
                        default_algorithm=env_data.get('default_algorithm', 'SAC'),
                        recommended_timesteps=env_data.get('recommended_timesteps', 100000),
                        tags=env_data.get('tags', [])
                    )
                    self.register_metadata(metadata)
                except Exception as e:
                    logger.warning(f"Failed to register metadata for {env_data.get('env_id', 'unknown')}: {e}")
            except Exception as e:
                logger.warning(f"Failed to register {env_data.get('env_id', 'unknown')}: {e}")
        
        self._custom_envs_registered = True
    
    def register_metadata(self, metadata: EnvMetadata):
        """Register metadata for an environment.
        
        Args:
            metadata: Environment metadata to register
        """
        self._registry[metadata.env_id] = metadata
    
    def get_metadata(self, env_id: str) -> Optional[EnvMetadata]:
        """Get metadata for an environment.
        
        Args:
            env_id: Environment ID (e.g., 'Walker2d-v5')
        
        Returns:
            EnvMetadata if found, None otherwise
        """
        # Ensure custom envs are registered
        if env_id not in self._registry:
            self.register_custom_envs()
        
        return self._registry.get(env_id)
    
    def list_envs(
        self, 
        category: Optional[EnvCategory] = None,
        difficulty: Optional[EnvDifficulty] = None,
        tags: Optional[List[str]] = None,
        include_custom: bool = True
    ) -> List[EnvMetadata]:
        """List available environments with optional filtering.
        
        Args:
            category: Filter by category
            difficulty: Filter by difficulty
            tags: Filter by tags (any match)
            include_custom: Whether to include custom environments
        
        Returns:
            List of matching environment metadata
        """
        # Ensure custom envs are registered
        self.register_custom_envs()
        
        results = []
        for metadata in self._registry.values():
            # Filter by custom
            if not include_custom and metadata.is_custom:
                continue
            
            # Filter by category
            if category is not None and metadata.category != category:
                continue
            
            # Filter by difficulty
            if difficulty is not None and metadata.difficulty != difficulty:
                continue
            
            # Filter by tags
            if tags is not None:
                if not any(tag in metadata.tags for tag in tags):
                    continue
            
            results.append(metadata)
        
        return sorted(results, key=lambda m: (m.category.value, m.difficulty.value, m.env_id))
    
    def is_registered(self, env_id: str) -> bool:
        """Check if an environment is registered.
        
        Args:
            env_id: Environment ID to check
        
        Returns:
            True if registered, False otherwise
        """
        # Ensure custom envs are registered
        self.register_custom_envs()
        return env_id in self._registry
    
    def validate_env(self, env_id: str) -> bool:
        """Validate that an environment can be created.
        
        Args:
            env_id: Environment ID to validate
        
        Returns:
            True if environment can be created, False otherwise
        """
        try:
            env = gym.make(env_id)
            env.close()
            return True
        except Exception as e:
            logger.warning(f"Failed to validate environment {env_id}: {e}")
            return False


# Global registry instance
_global_registry = EnvironmentRegistry()


def get_env_registry() -> EnvironmentRegistry:
    """Get the global environment registry.
    
    Returns:
        Global EnvironmentRegistry instance
    """
    return _global_registry


def make_env(env_id: str, rank: int = 0, seed: int = 0, **kwargs) -> Callable:
    """Create an environment factory function.
    
    This is the centralized environment creation function that should be used
    throughout robot_lab for creating training and evaluation environments.
    
    Args:
        env_id: Environment ID (e.g., 'Walker2d-v5')
        rank: Unique ID for this environment instance (for vectorized envs)
        seed: Random seed base
        **kwargs: Additional arguments to pass to gym.make()
    
    Returns:
        Callable that creates and returns a gym.Env
    
    Example:
        >>> env_fn = make_env('Walker2d-v5', rank=0, seed=42)
        >>> env = env_fn()  # Creates the environment
    """
    # Ensure custom envs are registered
    registry = get_env_registry()
    registry.register_custom_envs()
    
    def _init():
        # Create environment
        env = gym.make(env_id, **kwargs)
        
        # Set seed
        env.reset(seed=seed + rank)
        
        # Wrap with Monitor to track episode statistics
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
        
        return env
    
    return _init


def get_env_info(env_id: str) -> str:
    """Get formatted information about an environment.
    
    Args:
        env_id: Environment ID
    
    Returns:
        Formatted string with environment information
    """
    registry = get_env_registry()
    metadata = registry.get_metadata(env_id)
    
    if metadata is None:
        return f"No metadata found for {env_id}"
    
    info = [
        f"Environment: {metadata.env_id}",
        f"Category: {metadata.category.value}",
        f"Difficulty: {metadata.difficulty.value}",
        f"Description: {metadata.description}",
        f"Observation Space: {metadata.observation_space_desc}",
        f"Action Space: {metadata.action_space_desc}",
        f"Default Algorithm: {metadata.default_algorithm}",
        f"Recommended Timesteps: {metadata.recommended_timesteps:,}",
    ]
    
    if metadata.is_custom:
        info.append("Type: Custom robot_lab environment")
    
    if metadata.requires_mujoco:
        info.append("Requires: MuJoCo")
    
    if metadata.requires_robot_descriptions:
        info.append("Requires: robot_descriptions package")
    
    if metadata.tags:
        info.append(f"Tags: {', '.join(metadata.tags)}")
    
    return "\n".join(info)


__all__ = [
    "EnvironmentRegistry",
    "EnvMetadata", 
    "EnvCategory",
    "EnvDifficulty",
    "get_env_registry",
    "make_env",
    "get_env_info",
]
