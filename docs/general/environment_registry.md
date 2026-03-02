# Environment Registry System

The robot_lab environment registry provides centralized management of all available environments. Environment metadata is configured via JSON files, making it easy to add new environments without modifying Python code.

## Architecture

The registry system consists of:

- **JSON Configuration Files**: Define environment metadata
  - `robot_lab/configs/builtin_envs.json`: Standard Gymnasium environments
  - `robot_lab/configs/custom_envs.json`: Custom robot_lab environments
  
- **Registry Module**: `robot_lab/envs/registry.py`
  - Loads metadata from JSON files on initialization
  - Registers custom environments with Gymnasium
  - Provides filtering and discovery capabilities
  
- **Central Environment Factory**: `make_env()` function
  - Single point for creating environments throughout the codebase
  - Handles seeding and Monitor wrapping automatically

## Directory Structure

```
robot_lab/envs/
├── __init__.py              # Public API exports
├── registry.py              # Central registry implementation
├── manipulation/            # Manipulation task environments
│   ├── __init__.py
│   └── gripper.py
└── locomotion/              # Locomotion task environments
    ├── __init__.py
    └── quadruped.py
```

## Adding a New Environment

### 1. Create the Environment Class

Create your environment following Gymnasium's `gym.Env` interface:

```python
# robot_lab/envs/manipulation/my_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyCustomEnv(gym.Env):
    """Your custom environment."""
    
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = {}
        return observation, info
    
    def step(self, action):
        # Implement environment dynamics
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.zeros(4, dtype=np.float32)
```

### 2. Export the Environment

Add your environment to the module's `__init__.py`:

```python
# robot_lab/envs/manipulation/__init__.py
from robot_lab.envs.manipulation.gripper import GripperEnv
from robot_lab.envs.manipulation.my_env import MyCustomEnv  # Add this

__all__ = ["GripperEnv", "MyCustomEnv"]  # Add to exports
```

### 3. Add Metadata to JSON

Add an entry to `robot_lab/configs/custom_envs.json`:

```json
{
  "environments": [
    {
      "env_id": "MyCustomEnv-v0",
      "entry_point": "robot_lab.envs.manipulation.my_env:MyCustomEnv",
      "max_episode_steps": 1000,
      "category": "manipulation",
      "difficulty": "medium",
      "description": "Brief description of what your environment does",
      "observation_space_desc": "4D state vector (x, y, vx, vy)",
      "action_space_desc": "Continuous: 2D control input [-1, 1]",
      "is_custom": true,
      "requires_mujoco": false,
      "requires_robot_descriptions": false,
      "default_algorithm": "SAC",
      "recommended_timesteps": 200000,
      "tags": ["continuous", "manipulation", "custom"]
    }
  ]
}
```

### 4. Test the Environment

The environment will be automatically registered when robot_lab is imported:

```bash
# List all environments
robot-lab list-envs

# Get detailed info
robot-lab env-info --env MyCustomEnv-v0

# Train on your environment
robot-lab train --env MyCustomEnv-v0 --algo SAC
```

## JSON Schema Reference

### Environment Metadata Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `env_id` | string | ✓ | Unique environment identifier (e.g., "MyEnv-v0") |
| `entry_point` | string | ✓* | Import path to environment class (custom envs only) |
| `max_episode_steps` | integer | | Maximum steps per episode (default: 1000) |
| `category` | string | ✓ | Category: `classic_control`, `mujoco`, `locomotion`, `manipulation`, `custom` |
| `difficulty` | string | ✓ | Difficulty: `easy`, `medium`, `hard`, `expert` |
| `description` | string | ✓ | Human-readable description |
| `observation_space_desc` | string | ✓ | Description of observation space |
| `action_space_desc` | string | ✓ | Description of action space |
| `is_custom` | boolean | ✓ | Whether this is a custom environment |
| `requires_mujoco` | boolean | | Requires MuJoCo physics engine |
| `requires_robot_descriptions` | boolean | | Requires robot_descriptions package |
| `default_algorithm` | string | ✓ | Recommended algorithm: `SAC`, `PPO`, etc. |
| `recommended_timesteps` | integer | ✓ | Typical training timesteps needed |
| `tags` | array[string] | | Searchable tags for filtering |

**Note**: `entry_point` is only required for custom environments in `custom_envs.json`

### Example: Built-in Environment

```json
{
  "env_id": "Walker2d-v5",
  "category": "mujoco",
  "difficulty": "hard",
  "description": "2D biped walker locomotion",
  "observation_space_desc": "Joint angles, velocities, torso orientation (17D)",
  "action_space_desc": "Continuous: Joint torques (6D)",
  "is_custom": false,
  "requires_mujoco": true,
  "requires_robot_descriptions": false,
  "default_algorithm": "SAC",
  "recommended_timesteps": 1000000,
  "tags": ["continuous", "locomotion", "mujoco", "biped"]
}
```

### Example: Custom Environment

```json
{
  "env_id": "GripperEnv-v0",
  "entry_point": "robot_lab.envs.manipulation.gripper:GripperEnv",
  "max_episode_steps": 1000,
  "category": "manipulation",
  "difficulty": "medium",
  "description": "MuJoCo-based gripper manipulation",
  "observation_space_desc": "Joint positions and velocities",
  "action_space_desc": "Continuous: Joint torques or target positions",
  "is_custom": true,
  "requires_mujoco": true,
  "requires_robot_descriptions": false,
  "default_algorithm": "SAC",
  "recommended_timesteps": 300000,
  "tags": ["continuous", "manipulation", "custom", "mujoco"]
}
```

## Using the Registry Programmatically

### Get Environment Metadata

```python
from robot_lab.envs import get_env_registry

registry = get_env_registry()
metadata = registry.get_metadata("Walker2d-v5")

print(f"Category: {metadata.category}")
print(f"Recommended timesteps: {metadata.recommended_timesteps}")
```

### List Environments with Filters

```python
from robot_lab.envs import get_env_registry, EnvCategory, EnvDifficulty

registry = get_env_registry()

# Get all locomotion environments
locomotion_envs = registry.list_envs(category=EnvCategory.LOCOMOTION)

# Get easy environments
easy_envs = registry.list_envs(difficulty=EnvDifficulty.EASY)

# Get continuous control environments
continuous_envs = registry.list_envs(tags=["continuous"])

# Get only custom environments
custom_envs = registry.list_envs(include_custom=True)
custom_only = [e for e in custom_envs if e.is_custom]
```

### Create Environments

```python
from robot_lab.envs import make_env

# Create a single environment
env_fn = make_env("Walker2d-v5", rank=0, seed=42)
env = env_fn()

# Create vectorized environments
from stable_baselines3.common.vec_env import SubprocVecEnv

env_fns = [make_env("Walker2d-v5", rank=i, seed=42) for i in range(4)]
vec_env = SubprocVecEnv(env_fns)
```

### Validate Environment

```python
from robot_lab.envs import get_env_registry

registry = get_env_registry()

# Check if environment exists
if registry.is_registered("MyEnv-v0"):
    print("Environment is registered")

# Validate that environment can be created
if registry.validate_env("MyEnv-v0"):
    print("Environment can be instantiated")
```

## CLI Commands

### List All Environments

```bash
robot-lab list-envs
```

### Filter by Category

```bash
robot-lab list-envs --category mujoco
robot-lab list-envs --category locomotion
```

### Filter by Difficulty

```bash
robot-lab list-envs --difficulty easy
robot-lab list-envs --difficulty expert
```

### Show Custom Environments Only

```bash
robot-lab list-envs --custom-only
```

### Get Detailed Environment Info

```bash
robot-lab env-info --env Walker2d-v5
robot-lab env-info --env GripperEnv-v0
```

## Testing

The registry system includes comprehensive validation tests in `tests/test_env_registry.py`:

```bash
# Run all registry tests
pytest tests/test_env_registry.py -v

# Run specific test classes
pytest tests/test_env_registry.py::TestEnvironmentRegistry -v
pytest tests/test_env_registry.py::TestJSONConfiguration -v
```

Test coverage includes:
- Registry initialization and loading
- Metadata validation
- Environment filtering
- JSON schema validation
- Environment creation and validation
- No duplicate environment IDs
- Valid category and difficulty values

## Best Practices

1. **Naming Convention**: Use `EnvironmentName-v0` format for env_id
2. **Entry Points**: Use full module path: `package.module:ClassName`
3. **Categories**: Use existing categories when possible
4. **Tags**: Include relevant tags for discoverability:
   - Action type: `continuous`, `discrete`
   - Task type: `locomotion`, `manipulation`, `navigation`
   - Properties: `sparse_reward`, `dense_reward`, `image_obs`
5. **Timesteps**: Set realistic `recommended_timesteps` based on task complexity
6. **Dependencies**: Clearly mark `requires_mujoco` and `requires_robot_descriptions`

## Migration from Old System

If you have environments registered in the old Python-based system:

1. Find the registration code in `robot_lab/envs/__init__.py` (old version)
2. Extract the metadata
3. Add to appropriate JSON file (`custom_envs.json`)
4. Remove the Python registration code

Old system (deprecated):
```python
register(
    id='MyEnv-v0',
    entry_point='robot_lab.envs.my_env:MyEnv',
)
```

New system:
```json
{
  "env_id": "MyEnv-v0",
  "entry_point": "robot_lab.envs.my_env:MyEnv",
  ...
}
```

## Troubleshooting

### Environment Not Found

```bash
robot-lab list-envs  # Check if environment is listed
robot-lab env-info --env MyEnv-v0  # Get detailed info
```

### Import Errors

Check that:
1. Entry point path is correct
2. Environment class is defined in the specified module
3. Module is properly added to package structure

### Missing Dependencies

The registry will warn if dependencies are missing:
```
⚠ MyEnv-v0 requires MuJoCo but it's not installed
```

Install required dependencies:
```bash
pip install mujoco
pip install robot_descriptions
```
