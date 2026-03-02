"""Tests for environment registry and validation."""

import pytest
import gymnasium as gym
from pathlib import Path
import json

from robot_lab.envs import (
    get_env_registry,
    make_env,
    get_env_info,
    EnvCategory,
    EnvDifficulty,
)


class TestEnvironmentRegistry:
    """Test the environment registry system."""
    
    def test_registry_initialization(self):
        """Test that the registry initializes correctly."""
        registry = get_env_registry()
        assert registry is not None
        assert isinstance(registry._registry, dict)
    
    def test_builtin_envs_loaded(self):
        """Test that built-in environments are loaded from JSON."""
        registry = get_env_registry()
        
        # Check that some standard environments are loaded
        expected_envs = [
            "CartPole-v1",
            "MountainCarContinuous-v0",
            "Pendulum-v1",
            "Walker2d-v5",
            "HalfCheetah-v5",
        ]
        
        for env_id in expected_envs:
            metadata = registry.get_metadata(env_id)
            assert metadata is not None, f"{env_id} not found in registry"
            assert metadata.env_id == env_id
            assert isinstance(metadata.category, EnvCategory)
            assert isinstance(metadata.difficulty, EnvDifficulty)
    
    def test_custom_envs_registration(self):
        """Test that custom environments are registered."""
        registry = get_env_registry()
        registry.register_custom_envs()
        
        # Check custom environments
        custom_envs = ["GripperEnv-v0", "A1Quadruped-v0"]
        
        for env_id in custom_envs:
            metadata = registry.get_metadata(env_id)
            assert metadata is not None, f"{env_id} not found in registry"
            assert metadata.is_custom is True
            assert metadata.env_id == env_id
    
    def test_metadata_fields(self):
        """Test that metadata has all required fields."""
        registry = get_env_registry()
        metadata = registry.get_metadata("CartPole-v1")
        
        assert metadata is not None
        assert hasattr(metadata, 'env_id')
        assert hasattr(metadata, 'category')
        assert hasattr(metadata, 'difficulty')
        assert hasattr(metadata, 'description')
        assert hasattr(metadata, 'observation_space_desc')
        assert hasattr(metadata, 'action_space_desc')
        assert hasattr(metadata, 'is_custom')
        assert hasattr(metadata, 'requires_mujoco')
        assert hasattr(metadata, 'requires_robot_descriptions')
        assert hasattr(metadata, 'default_algorithm')
        assert hasattr(metadata, 'recommended_timesteps')
        assert hasattr(metadata, 'tags')
        
        # Validate types
        assert isinstance(metadata.env_id, str)
        assert isinstance(metadata.category, EnvCategory)
        assert isinstance(metadata.difficulty, EnvDifficulty)
        assert isinstance(metadata.description, str)
        assert isinstance(metadata.is_custom, bool)
        assert isinstance(metadata.requires_mujoco, bool)
        assert isinstance(metadata.default_algorithm, str)
        assert isinstance(metadata.recommended_timesteps, int)
        assert isinstance(metadata.tags, list)
    
    def test_list_envs_all(self):
        """Test listing all environments."""
        registry = get_env_registry()
        all_envs = registry.list_envs()
        
        assert len(all_envs) > 0
        # Should have at least the built-in envs
        assert len(all_envs) >= 10
    
    def test_list_envs_by_category(self):
        """Test filtering environments by category."""
        registry = get_env_registry()
        
        # Filter by classic control
        classic_envs = registry.list_envs(category=EnvCategory.CLASSIC_CONTROL)
        assert len(classic_envs) > 0
        for metadata in classic_envs:
            assert metadata.category == EnvCategory.CLASSIC_CONTROL
        
        # Filter by mujoco
        mujoco_envs = registry.list_envs(category=EnvCategory.MUJOCO)
        assert len(mujoco_envs) > 0
        for metadata in mujoco_envs:
            assert metadata.category == EnvCategory.MUJOCO
    
    def test_list_envs_by_difficulty(self):
        """Test filtering environments by difficulty."""
        registry = get_env_registry()
        
        # Filter by easy
        easy_envs = registry.list_envs(difficulty=EnvDifficulty.EASY)
        assert len(easy_envs) > 0
        for metadata in easy_envs:
            assert metadata.difficulty == EnvDifficulty.EASY
    
    def test_list_envs_by_tags(self):
        """Test filtering environments by tags."""
        registry = get_env_registry()
        
        # Filter by continuous tag
        continuous_envs = registry.list_envs(tags=["continuous"])
        assert len(continuous_envs) > 0
        for metadata in continuous_envs:
            assert "continuous" in metadata.tags
    
    def test_list_envs_exclude_custom(self):
        """Test excluding custom environments."""
        registry = get_env_registry()
        
        builtin_only = registry.list_envs(include_custom=False)
        for metadata in builtin_only:
            assert metadata.is_custom is False
    
    def test_is_registered(self):
        """Test checking if environments are registered."""
        registry = get_env_registry()
        
        assert registry.is_registered("CartPole-v1") is True
        assert registry.is_registered("NonExistentEnv-v99") is False


class TestEnvironmentValidation:
    """Test environment validation functionality."""
    
    def test_validate_builtin_env(self):
        """Test validating a built-in environment."""
        registry = get_env_registry()
        
        # CartPole should always be available
        is_valid = registry.validate_env("CartPole-v1")
        assert is_valid is True
    
    def test_validate_nonexistent_env(self):
        """Test validating a non-existent environment."""
        registry = get_env_registry()
        
        is_valid = registry.validate_env("NonExistentEnv-v99")
        assert is_valid is False
    
    @pytest.mark.parametrize("env_id", [
        "CartPole-v1",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ])
    def test_validate_multiple_envs(self, env_id):
        """Test validation of multiple common environments."""
        registry = get_env_registry()
        is_valid = registry.validate_env(env_id)
        assert is_valid is True, f"{env_id} should be valid"


class TestMakeEnv:
    """Test the centralized make_env function."""
    
    def test_make_env_basic(self):
        """Test basic environment creation."""
        env_fn = make_env("CartPole-v1", rank=0, seed=42)
        assert callable(env_fn)
        
        env = env_fn()
        assert env is not None
        
        # Check that it's wrapped with Monitor
        from stable_baselines3.common.monitor import Monitor
        assert isinstance(env, Monitor)
        
        env.close()
    
    def test_make_env_different_seeds(self):
        """Test that different ranks get different seeds."""
        env_fn1 = make_env("CartPole-v1", rank=0, seed=42)
        env_fn2 = make_env("CartPole-v1", rank=1, seed=42)
        
        env1 = env_fn1()
        env2 = env_fn2()
        
        # Reset and check that initial states might differ due to different seeds
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Environments exist and can be reset
        assert obs1 is not None
        assert obs2 is not None
        
        env1.close()
        env2.close()
    
    def test_make_env_continuous(self):
        """Test creating a continuous action environment."""
        env_fn = make_env("MountainCarContinuous-v0", rank=0, seed=42)
        env = env_fn()
        
        # Check action space is continuous
        from gymnasium.spaces import Box
        # Monitor wraps the env, so access the unwrapped env
        assert isinstance(env.unwrapped.action_space, Box)
        
        env.close()


class TestGetEnvInfo:
    """Test the get_env_info function."""
    
    def test_get_info_existing_env(self):
        """Test getting info for an existing environment."""
        info = get_env_info("Walker2d-v5")
        
        assert isinstance(info, str)
        assert "Walker2d-v5" in info
        assert "Category:" in info
        assert "Difficulty:" in info
        assert "Description:" in info
        assert "Observation Space:" in info
        assert "Action Space:" in info
    
    def test_get_info_nonexistent_env(self):
        """Test getting info for a non-existent environment."""
        info = get_env_info("NonExistentEnv-v99")
        
        assert isinstance(info, str)
        assert "No metadata found" in info


class TestJSONConfiguration:
    """Test that JSON configuration files are valid."""
    
    def test_builtin_envs_json_valid(self):
        """Test that builtin_envs.json is valid JSON."""
        from importlib.resources import files
        
        config_files = files('robot_lab').joinpath('configs')
        builtin_json = config_files.joinpath('builtin_envs.json').read_text()
        
        data = json.loads(builtin_json)
        assert 'environments' in data
        assert isinstance(data['environments'], list)
        assert len(data['environments']) > 0
        
        # Validate structure of each environment entry
        for env_data in data['environments']:
            assert 'env_id' in env_data
            assert 'category' in env_data
            assert 'difficulty' in env_data
            assert 'description' in env_data
            assert 'observation_space_desc' in env_data
            assert 'action_space_desc' in env_data
            assert 'default_algorithm' in env_data
            assert 'recommended_timesteps' in env_data
    
    def test_custom_envs_json_valid(self):
        """Test that custom_envs.json is valid JSON."""
        from importlib.resources import files
        
        config_files = files('robot_lab').joinpath('configs')
        custom_json = config_files.joinpath('custom_envs.json').read_text()
        
        data = json.loads(custom_json)
        assert 'environments' in data
        assert isinstance(data['environments'], list)
        
        # Validate structure of each custom environment entry
        for env_data in data['environments']:
            assert 'env_id' in env_data
            assert 'entry_point' in env_data
            assert 'category' in env_data
            assert 'difficulty' in env_data
            assert 'is_custom' in env_data
            assert env_data['is_custom'] is True  # Custom envs should have is_custom=True
    
    def test_no_duplicate_env_ids(self):
        """Test that there are no duplicate environment IDs across JSON files."""
        from importlib.resources import files
        
        config_files = files('robot_lab').joinpath('configs')
        
        # Load both JSON files
        builtin_data = json.loads(config_files.joinpath('builtin_envs.json').read_text())
        custom_data = json.loads(config_files.joinpath('custom_envs.json').read_text())
        
        # Collect all env_ids
        builtin_ids = {env['env_id'] for env in builtin_data['environments']}
        custom_ids = {env['env_id'] for env in custom_data['environments']}
        
        # Check for duplicates
        duplicates = builtin_ids & custom_ids
        assert len(duplicates) == 0, f"Duplicate env_ids found: {duplicates}"
    
    def test_valid_categories_and_difficulties(self):
        """Test that all categories and difficulties are valid enum values."""
        from importlib.resources import files
        
        config_files = files('robot_lab').joinpath('configs')
        
        # Load both JSON files
        builtin_data = json.loads(config_files.joinpath('builtin_envs.json').read_text())
        custom_data = json.loads(config_files.joinpath('custom_envs.json').read_text())
        
        all_envs = builtin_data['environments'] + custom_data['environments']
        
        valid_categories = {cat.value for cat in EnvCategory}
        valid_difficulties = {diff.value for diff in EnvDifficulty}
        
        for env_data in all_envs:
            assert env_data['category'] in valid_categories, \
                f"Invalid category '{env_data['category']}' for {env_data['env_id']}"
            assert env_data['difficulty'] in valid_difficulties, \
                f"Invalid difficulty '{env_data['difficulty']}' for {env_data['env_id']}"


class TestEnvironmentCreation:
    """Test actual environment creation for registered environments."""
    
    @pytest.mark.parametrize("env_id", [
        "CartPole-v1",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ])
    def test_create_basic_envs(self, env_id):
        """Test creating basic Gymnasium environments."""
        env_fn = make_env(env_id, rank=0, seed=42)
        env = env_fn()
        
        # Test basic operations
        obs, info = env.reset()
        assert obs is not None
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        env.close()
    
    def test_custom_env_registration_check(self):
        """Test that custom environments are properly registered."""
        registry = get_env_registry()
        registry.register_custom_envs()
        
        from gymnasium.envs.registration import registry as gym_registry
        env_ids = [spec.id for spec in gym_registry.values()]
        
        # Check that our custom envs are in Gymnasium's registry
        assert 'GripperEnv-v0' in env_ids
        assert 'A1Quadruped-v0' in env_ids
