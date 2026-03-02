# Adding a New Environment to robot_lab

This guide shows you how to add a new environment to robot_lab using the JSON-based registry system.

## Quick Start

To add a new custom environment, you only need to:

1. Create your environment class (Python)
2. Add an entry to `robot_lab/configs/custom_envs.json` (JSON)
3. Export your environment class from its module

That's it! No need to modify registration code.

## Step-by-Step Guide

### 1. Create Your Environment Class

Create a new file in the appropriate directory:
- Manipulation tasks: `robot_lab/envs/manipulation/`
- Locomotion tasks: `robot_lab/envs/locomotion/`

Example (`robot_lab/envs/manipulation/reach.py`):

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ReachEnv(gym.Env):
    """Simple 2D reaching task."""
    
    def __init__(self):
        super().__init__()
        
        # 4D observation: agent (x,y), target (x,y)
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # 2D continuous action: velocity (vx, vy)
        self.action_space = spaces.Box(
            low=-0.1, 
            high=0.1, 
            shape=(2,), 
            dtype=np.float32
        )
        
        self.agent_pos = np.zeros(2)
        self.target_pos = np.zeros(2)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize positions
        self.agent_pos = self.np_random.uniform(-0.8, 0.8, size=2)
        self.target_pos = self.np_random.uniform(-0.8, 0.8, size=2)
        return self._get_obs(), {}
    
    def step(self, action):
        # Update agent position
        self.agent_pos += action
        self.agent_pos = np.clip(self.agent_pos, -1.0, 1.0)
        
        # Calculate reward (negative distance to target)
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -distance
        
        # Check if reached target
        terminated = distance < 0.05
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.target_pos]).astype(np.float32)
```

### 2. Export Your Environment

Add your environment to the module's `__init__.py`:

```python
# robot_lab/envs/manipulation/__init__.py
from robot_lab.envs.manipulation.gripper import GripperEnv
from robot_lab.envs.manipulation.reach import ReachEnv  # Add this line

__all__ = ["GripperEnv", "ReachEnv"]  # Add to exports
```

### 3. Add JSON Configuration

Add an entry to `robot_lab/configs/custom_envs.json`:

```json
{
  "environments": [
    {
      "env_id": "ReachEnv-v0",
      "entry_point": "robot_lab.envs.manipulation.reach:ReachEnv",
      "max_episode_steps": 200,
      "category": "manipulation",
      "difficulty": "easy",
      "description": "Simple 2D reaching task",
      "observation_space_desc": "Agent position (x,y) and target position (x,y) in [-1,1]",
      "action_space_desc": "Continuous: Velocity commands (vx, vy)",
      "is_custom": true,
      "requires_mujoco": false,
      "requires_robot_descriptions": false,
      "default_algorithm": "SAC",
      "recommended_timesteps": 50000,
      "tags": ["continuous", "manipulation", "custom", "simple", "2d"]
    }
  ]
}
```

### 4. Test Your Environment

```bash
# List all environments to verify it's registered
robot-lab list-envs --custom-only

# Get detailed info about your environment
robot-lab env-info --env ReachEnv-v0

# Test training
robot-lab train --env ReachEnv-v0 --algo SAC --seed 42
```

## JSON Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `env_id` | string | ✓ | Unique environment identifier (format: "EnvName-v0") |
| `entry_point` | string | ✓ | Python import path: `module.submodule:ClassName` |
| `max_episode_steps` | integer | | Maximum steps per episode (default: 1000) |
| `category` | string | ✓ | One of: `classic_control`, `mujoco`, `locomotion`, `manipulation`, `custom` |
| `difficulty` | string | ✓ | One of: `easy`, `medium`, `hard`, `expert` |
| `description` | string | ✓ | Brief description of the task |
| `observation_space_desc` | string | ✓ | Human-readable observation space description |
| `action_space_desc` | string | ✓ | Human-readable action space description |
| `is_custom` | boolean | ✓ | Always `true` for custom environments |
| `requires_mujoco` | boolean | | Set to `true` if environment needs MuJoCo |
| `requires_robot_descriptions` | boolean | | Set to `true` if environment needs robot_descriptions |
| `default_algorithm` | string | ✓ | Recommended algorithm: `SAC`, `PPO` |
| `recommended_timesteps` | integer | ✓ | Typical training duration |
| `tags` | array[string] | | Searchable tags for filtering and discovery |

## Useful Tags

Include relevant tags to help users discover your environment:

- **Action type**: `continuous`, `discrete`
- **Task type**: `locomotion`, `manipulation`, `navigation`, `reaching`, `grasping`
- **Complexity**: `simple`, `complex`, `sparse_reward`, `dense_reward`
- **Observation type**: `state`, `image_obs`, `depth`, `point_cloud`
- **Dimensionality**: `2d`, `3d`
- **Simulation**: `mujoco`, `pybullet`, `custom`

## Adding Built-in Gymnasium Environments

To add metadata for a standard Gymnasium environment (one that's already registered with Gymnasium):

Add to `robot_lab/configs/builtin_envs.json`:

```json
{
  "env_id": "BipedalWalker-v3",
  "category": "classic_control",
  "difficulty": "hard",
  "description": "Bipedal robot walking over rough terrain",
  "observation_space_desc": "24D state vector (hull angle, velocity, joint angles/speeds, leg contacts, lidar)",
  "action_space_desc": "Continuous: Hip and knee torques for both legs (4D)",
  "is_custom": false,
  "requires_mujoco": false,
  "requires_robot_descriptions": false,
  "default_algorithm": "SAC",
  "recommended_timesteps": 500000,
  "tags": ["continuous", "locomotion", "box2d", "lidar"]
}
```

**Note**: No `entry_point` needed for built-in environments!

## Examples

### Example 1: Simple Custom Environment

**File**: `robot_lab/envs/custom/simple.py`

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BalanceEnv(gym.Env):
    """Balance a point mass at origin."""
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-5, 5, size=2).astype(np.float32)
        return self.state, {}
    
    def step(self, action):
        self.state[1] += action[0]  # Apply force
        self.state[0] += self.state[1] * 0.1  # Update position
        
        reward = -abs(self.state[0])  # Reward for staying at origin
        terminated = abs(self.state[0]) > 10
        
        return self.state, reward, terminated, False, {}
```

**JSON**:
```json
{
  "env_id": "BalanceEnv-v0",
  "entry_point": "robot_lab.envs.custom.simple:BalanceEnv",
  "max_episode_steps": 100,
  "category": "classic_control",
  "difficulty": "easy",
  "description": "Balance a point mass at the origin",
  "observation_space_desc": "Position and velocity (2D)",
  "action_space_desc": "Continuous: Applied force [-1, 1]",
  "is_custom": true,
  "requires_mujoco": false,
  "default_algorithm": "SAC",
  "recommended_timesteps": 20000,
  "tags": ["continuous", "simple", "1d", "custom"]
}
```

## Validation

Run tests to ensure your environment is properly registered:

```bash
# Run registry tests
pytest tests/test_env_registry.py -v

# Run specific validation test
pytest tests/test_env_registry.py::TestEnvironmentCreation -v
```

## Troubleshooting

### Environment Not Found

```bash
# Check if environment appears in list
robot-lab list-envs | grep YourEnv

# Check for JSON syntax errors
python -m json.tool robot_lab/configs/custom_envs.json
```

### Import Errors

Verify the entry point path:
```python
# Should work without error
from robot_lab.envs.manipulation.reach import ReachEnv
```

### Registration Errors

Check that:
1. `env_id` follows format: `EnvName-v0` (must end with version)
2. `entry_point` uses correct syntax: `module.path:ClassName`
3. Class inherits from `gym.Env`
4. Class is exported in module's `__init__.py`

## Best Practices

1. **Versioning**: Start with `-v0`, increment for breaking changes
2. **Naming**: Use descriptive names (e.g., `ReachEnv-v0` not `Env1-v0`)
3. **Categories**: Use existing categories when possible
4. **Tags**: Include both general (`continuous`) and specific (`2d`, `reaching`) tags
5. **Timesteps**: Be realistic with `recommended_timesteps`
6. **Dependencies**: Clearly mark environmental requirements
7. **Testing**: Test environment creation before adding to repository

## See Also

- [Environment Registry Documentation](environment_registry.md) - Full registry system details
- [Gymnasium Documentation](https://gymnasium.farama.org/) - Environment interface reference
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/) - RL algorithm details
