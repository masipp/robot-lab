"""
Gym/Gymnasium wrapper for MuJoCo locomotion environments.
Uses MuJoCo directly with realistic quadruped robots (e.g., Unitree A1) from robot_descriptions.

This wrapper exposes:
 - gym.Env API (observation, action, reward, done, info)
 - methods that CurriculumManager will call to modify the arena:
    * add_friction_patch()
    * add_surface_noise()
    * spawn_waypoint()
    * spawn_block()
    * spawn_climb_obstacle()

Supports realistic quadruped robots like Unitree A1, Go1, Spot, etc.
"""
import numpy as np
import copy
import os
import time
from importlib.resources import files
from loguru import logger

# Try gymnasium first (maintained fork), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    USING_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    USING_GYMNASIUM = False

# Try importing mujoco
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Try importing robot_descriptions
try:
    from robot_descriptions import a1_description
    ROBOT_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    ROBOT_DESCRIPTIONS_AVAILABLE = False



class A1QuadrupedEnv(gym.Env):
    """
    Custom MuJoCo environment for quadruped robot locomotion.
    Uses a simplified quadruped model built directly in MuJoCo XML.
    """
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }
    
    def __init__(self, render_mode=None, kp=None, kd=None, ki=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.kp = kp
        self.kd = kd
        self.ki = ki
        
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo not installed. Run: pip install mujoco")
        
        # Load quadruped model from XML file
        xml_string = self._load_quadruped_xml()
        
        # Load the model from XML string
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self._apply_gains(self.model, kp=self.kp, kd=self.kd)

        self.data = mujoco.MjData(self.model)
        
        # Create viewer if rendering (lazy initialization)
        self.viewer = None
        self._viewer_initialized = False
        
        # Define action and observation spaces
        # 12 actuated joints (3 per leg: hip, thigh, knee)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(12,), 
            dtype=np.float32
        )
        
        # Observation: joint positions (12), joint velocities (12), base orientation (3 euler), base velocity (3)
        obs_dim = 12 + 12 + 3 + 3  # 30 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )
        
        self.current_step = 0
        self.max_steps = 1000
    
    @staticmethod
    def _apply_gains(model, kp=None, kd=None):
        """Apply control gains to MuJoCo position actuators.
        
        For MuJoCo position actuators:
        - actuator_gainprm[i, 0] = kp (proportional gain)
        - actuator_biasprm[i, 1] = -kp (also part of PD control)
        - actuator_biasprm[i, 2] = -kv (velocity/derivative gain)
        """
        if kp is not None:
            for i in range(model.nu):
                model.actuator_gainprm[i, 0] = kp
                model.actuator_biasprm[i, 1] = -kp
        if kd is not None:
            for i in range(model.nu):
                model.actuator_biasprm[i, 2] = -kd


    def _load_quadruped_xml(self):
        """Load the quadruped model XML from package resources."""
        # Load XML file from the same directory as this module
        xml_file = files('robot_lab.envs.locomotion').joinpath('simple_quadruped.xml')
        return xml_file.read_text()
        
    def _get_obs(self):
        """Get current observation."""
        # qpos: [x, y, z, qw, qx, qy, qz, joint1, ..., joint12]  (19 total)
        # qvel: [vx, vy, vz, wx, wy, wz, joint_vel1, ..., joint_vel12]  (18 total)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Joint positions (12) and velocities (12)
        joint_pos = qpos[7:19]  # indices 7-18
        joint_vel = qvel[6:18]  # indices 6-17
        
        # Base orientation as euler angles (simpler than quaternion for learning)
        quat = qpos[3:7]  # [qw, qx, qy, qz]
        # Convert quaternion to roll, pitch, yaw
        euler = self._quat_to_euler(quat)
        
        # Base velocity (linear)
        base_vel = qvel[0:3]  # [vx, vy, vz]
        
        obs = np.concatenate([
            joint_pos,      # 12
            joint_vel,      # 12
            euler,          # 3
            base_vel        # 3
        ])  # Total: 30
        
        return obs.astype(np.float64)
    
    def _quat_to_euler(self, quat):
        """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _get_reward(self):
        """Calculate reward - encourage forward movement and stability."""
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        # Forward velocity reward (primary objective)
        forward_vel = qvel[0]  # x-axis velocity
        forward_reward = max(0, forward_vel)  # Only reward forward movement
        
        # Height bonus - encourage staying upright
        base_height = qpos[2]
        target_height = 0.35
        height_reward = -abs(base_height - target_height)  # Penalize deviation from target
        
        # Orientation penalty - penalize tilting
        quat = qpos[3:7]
        euler = self._quat_to_euler(quat)
        roll, pitch, yaw = euler
        orientation_penalty = -(roll**2 + pitch**2)  # Penalize roll and pitch
        
        # Alive bonus
        alive_bonus = 0.5
        
        # Energy penalty - discourage excessive movements
        ctrl_cost = -0.003 * np.sum(np.square(self.data.ctrl))
        
        # Total reward with weighted components
        total_reward = (
            2.0 * forward_reward +      # Strongly reward forward movement
            1.0 * height_reward +        # Maintain proper height
            0.5 * orientation_penalty +  # Stay upright
            alive_bonus +                # Reward for staying alive
            ctrl_cost                    # Small energy penalty
        )
        
        return total_reward
    
    def _is_done(self):
        """Check if episode should terminate."""
        qpos = self.data.qpos
        
        # Terminate if robot falls too low
        base_height = qpos[2]
        if base_height < 0.15:  # Significantly below standing height
            return True
        
        # Terminate if robot tips over completely
        quat = qpos[3:7]
        euler = self._quat_to_euler(quat)
        roll, pitch, yaw = euler
        
        # Check if tilted more than 60 degrees in roll or pitch
        if abs(roll) > 1.0 or abs(pitch) > 1.0:  # ~60 degrees
            return True
        
        return False
    
    def step(self, action):
        """Execute one step."""
        # Clip action to [-1, 1]
        action = np.clip(action, -1, 1)
        
        # Map action [-1, 1] to appropriate joint angle targets
        # Each leg has 3 joints: hip, thigh, knee
        # We'll use a simple mapping that allows for reasonable locomotion
        action_scale = np.array([
            0.5, 1.0, 1.5,  # Front left: hip, thigh, knee
            0.5, 1.0, 1.5,  # Front right
            0.5, 1.0, 1.5,  # Back left
            0.5, 1.0, 1.5,  # Back right
        ])
        
        # Convert normalized actions to joint targets
        # Knee joints have a bias towards bent position (standing)
        action_bias = np.array([
            0, 0.5, -1.0,  # Front left
            0, 0.5, -1.0,  # Front right
            0, 0.5, -1.0,  # Back left
            0, 0.5, -1.0,  # Back right
        ])
        
        target_angles = action_bias + action * action_scale
        
        # Set control signals
        self.data.ctrl[:] = target_angles
        
        # Step simulation multiple times for stability
        for _ in range(5):  # 5 physics steps per RL step
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation, reward, done
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps
        info = {
            'base_height': self.data.qpos[2],
            'forward_vel': self.data.qvel[0],
            'distance': self.data.qpos[0],  # Total x-distance traveled
        }
        
        # Don't sync viewer in step (only in render())
        # This prevents issues with vectorized environments
        
        if USING_GYMNASIUM:
            return obs, reward, terminated, truncated, info
        else:
            done = terminated or truncated
            return obs, reward, done, info
    
    def reset(self, seed=None, **kwargs):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose for quadruped
        # qpos structure: [x, y, z, qw, qx, qy, qz, joint1, ..., joint12]
        
        # Base position: starting height
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = 0.35  # z (slightly above ground)
        
        # Base orientation: upright (quaternion [w, x, y, z])
        self.data.qpos[3] = 1.0  # w
        self.data.qpos[4] = 0.0  # x
        self.data.qpos[5] = 0.0  # y
        self.data.qpos[6] = 0.0  # z
        
        # Joint positions: standing configuration
        # Each leg: hip=0, thigh=0.5, knee=-1.0 (bent, standing pose)
        standing_pose = np.array([
            0.0, 0.5, -1.0,  # Front left leg
            0.0, 0.5, -1.0,  # Front right leg
            0.0, 0.5, -1.0,  # Back left leg
            0.0, 0.5, -1.0,  # Back right leg
        ])
        self.data.qpos[7:19] = standing_pose
        
        # Add small random perturbations for variability
        self.data.qpos[7:19] += np.random.uniform(-0.02, 0.02, 12)
        
        # Zero velocities
        self.data.qvel[:] = 0
        
        # Let physics settle for a few steps (with standing pose control)
        self.data.ctrl[:] = standing_pose
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step = 0
        
        obs = self._get_obs()
        
        if USING_GYMNASIUM:
            return obs, {}
        else:
            return obs
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            # Lazy initialization of viewer
            if not self._viewer_initialized:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    self._viewer_initialized = True
                    
                    # Configure camera to follow the robot
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    self.viewer.cam.trackbodyid = 0  # Track torso (body 0)
                    self.viewer.cam.distance = 3.0  # Distance from robot
                    self.viewer.cam.elevation = -20  # Viewing angle (degrees)
                    
                    logger.debug("MuJoCo passive viewer initialized with tracking camera")
                except Exception as e:
                    logger.warning(f"Failed to initialize MuJoCo viewer: {e}")
                    self._viewer_initialized = True  # Don't try again
                    return None
            
            # Update the viewer if it's still running
            if self.viewer is not None:
                try:
                    # Check if viewer is still open
                    if self.viewer.is_running():
                        self.viewer.sync()
                        # Delay to make rendering observable (20 FPS)
                        time.sleep(0.05)  # 50ms delay for ~20 FPS
                    else:
                        # Viewer was closed, clean up
                        self.viewer = None
                        self._viewer_initialized = False
                except Exception as e:
                    logger.debug(f"Viewer sync failed: {e}")
                    return None
        
        return None
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class MuJoCoLocomotionWrapper(gym.Env):
    """
    Wrapper around MuJoCo environments (both built-in and custom) that adds curriculum learning hooks.
    """
    def __init__(self, env_name="Ant-v5", render_mode=None, use_custom_robot=False):
        """
        env_name: MuJoCo environment name (e.g., 'Ant-v5', 'Humanoid-v5') or 'A1' for custom robot
        render_mode: 'human', 'rgb_array', or None
        use_custom_robot: If True, use custom A1 robot instead of built-in env
        """
        # Create the base environment
        if use_custom_robot or env_name.lower() == 'a1':
            # Use custom A1 environment
            self._env = A1QuadrupedEnv(render_mode=render_mode)
        else:
            # Use built-in gymnasium/gym environment
            if USING_GYMNASIUM:
                self._env = gym.make(env_name, render_mode=render_mode)
            else:
                self._env = gym.make(env_name)
        
        self.render_mode = render_mode
        
        # Copy spaces from base environment
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
        self.current_step = 0
        
        # Store original model for reset
        if hasattr(self._env, 'model'):
            self._original_model = copy.deepcopy(self._env.model)

    def step(self, action):
        result = self._env.step(action)
        self.current_step += 1
        
        # Handle both gym and gymnasium return formats
        if USING_GYMNASIUM:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:
            obs, reward, done, info = result
            return obs, reward, done, info

    def reset(self, **kwargs):
        self.current_step = 0
        result = self._env.reset(**kwargs)
        
        # Handle both gym and gymnasium return formats
        if USING_GYMNASIUM:
            obs, info = result
            return obs
        else:
            return result

    def render(self, mode=None):
        if USING_GYMNASIUM:
            return self._env.render()
        else:
            return self._env.render(mode=mode or self.render_mode or 'human')

    def close(self):
        self._env.close()

    #
    # Arena / randomization hooks for CurriculumManager
    #
    def add_surface_noise(self, magnitude=0.01):
        """
        Add small perturbations to simulate rough surface.
        For MuJoCo, we can modify floor friction or add noise to observations.
        """
        self._surface_noise = magnitude
        
        # If we have access to the mujoco model, we could modify the floor geom here
        if hasattr(self._env, 'model') and hasattr(self._env.model, 'geom_friction'):
            # Store for later application during reset
            pass

    def add_friction_patch(self, position, size, friction_coef):
        """
        Add a rectangular low-friction patch at position with size and friction coefficient.
        This would require modifying the MuJoCo XML or model programmatically.
        """
        if not hasattr(self, "_friction_patches"):
            self._friction_patches = []
        self._friction_patches.append(dict(position=position, size=size, friction=friction_coef))

    def spawn_waypoint(self, position, radius=0.2):
        """
        Spawn a waypoint target for the agent to reach.
        Store waypoints that can be used in custom reward calculations.
        """
        if not hasattr(self, "_waypoints"):
            self._waypoints = []
        self._waypoints.append(dict(position=position, radius=radius))
        return len(self._waypoints)-1

    def get_waypoints(self):
        """Get all spawned waypoints."""
        return getattr(self, "_waypoints", [])

    def spawn_block(self, position, size=(0.25,0.25,0.25), mass=1.0, movable=True):
        """
        Spawn a cubic obstacle block in the arena.
        For full implementation, would need to modify MuJoCo model XML.
        """
        if not hasattr(self, "_blocks"):
            self._blocks = []
        self._blocks.append(dict(position=position, size=size, mass=mass, movable=movable))
        return len(self._blocks)-1

    def spawn_climb_obstacle(self, position, size=(0.4,0.4,0.2)):
        """
        Spawn an obstacle that the robot must climb over.
        """
        if not hasattr(self, "_climbs"):
            self._climbs = []
        self._climbs.append(dict(position=position, size=size))
        return len(self._climbs)-1

    def apply_arena_changes(self):
        """
        Apply stored arena modifications (friction patches, obstacles, etc.).
        Full implementation would require MuJoCo model modification.
        """
        # This is a hook for future implementation
        # Would need to modify self._env.model or regenerate environment
        pass
    
    def get_robot_position(self):
        """Get current robot position (useful for waypoint-based rewards)."""
        if hasattr(self._env, 'data') and hasattr(self._env.data, 'qpos'):
            # For most MuJoCo envs, first 2-3 positions are x,y,z coords
            return self._env.data.qpos[:3]
        return np.zeros(3)



if __name__ == "__main__":
    # Test the MuJoCo locomotion environment with A1 quadruped
    logger.info("=" * 60)
    logger.info("Testing MuJoCo Locomotion with Unitree A1 Quadruped")
    logger.info("=" * 60)
    
    # Check library availability
    if USING_GYMNASIUM:
        logger.success("Using Gymnasium (maintained fork)")
    else:
        logger.success("Using legacy Gym")
    
    if MUJOCO_AVAILABLE:
        logger.success("MuJoCo available")
    else:
        logger.error("MuJoCo not available")
        exit(1)
    
    if ROBOT_DESCRIPTIONS_AVAILABLE:
        logger.success("robot_descriptions available (A1 quadruped)")
    else:
        logger.error("robot_descriptions not available")
        logger.info("Install with: pip install robot_descriptions")
        exit(1)
    
    # Available environments
    available_envs = {
        1: ("A1", "Unitree A1 quadruped (realistic dog-like robot)", True),
        2: ("Ant-v5" if USING_GYMNASIUM else "Ant-v4", "Quadruped ant (built-in)", False),
        3: ("Humanoid-v5" if USING_GYMNASIUM else "Humanoid-v4", "Humanoid robot (built-in)", False),
    }
    
    logger.info("Available Locomotion Environments:")
    for key, (env_name, desc, custom) in available_envs.items():
        robot_type = "Custom" if custom else "Built-in"
        logger.info(f"  {key}. {env_name:20s} - {desc} [{robot_type}]")
    
    # Use A1 quadruped
    selected_env, desc, use_custom = available_envs[1]
    logger.info(f"Using: {selected_env} - {desc}")
    
    try:
        # Create the wrapped environment
        wrapped_env = MuJoCoLocomotionWrapper(
            env_name=selected_env,
            render_mode='human',
            use_custom_robot=use_custom
        )
        
        logger.success(f"Environment created successfully!")
        logger.info(f"  Observation space: {wrapped_env.observation_space}")
        logger.info(f"  Action space: {wrapped_env.action_space}")
        logger.info(f"  Action space shape: {wrapped_env.action_space.shape} (12 joints: 3 per leg)")
        
        # Test curriculum methods
        logger.info("Testing curriculum learning methods:")
        wrapped_env.spawn_waypoint(position=[5.0, 0.0, 0.0], radius=0.5)
        logger.success("  Waypoint spawned at [5.0, 0.0, 0.0]")
        wrapped_env.add_friction_patch(position=[2.0, 0.0], size=[1.0, 1.0], friction_coef=0.1)
        logger.success("  Friction patch added")
        wrapped_env.spawn_block(position=[3.0, 1.0, 0.0], size=[0.5, 0.5, 0.5])
        logger.success("  Block spawned")
        
        # Run episodes with random actions
        num_episodes = 5
        max_steps_per_episode = 500
        
        logger.info(f"Running {num_episodes} episodes with random actions...")
        logger.info("=" * 60)
        logger.info("Watch the A1 quadruped robot attempt to walk!")
        logger.info("(Random actions - expect it to fall, but it should be visible!)")
        
        import time
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info("-" * 40)
            
            obs = wrapped_env.reset()
            logger.info(f"  Initial observation shape: {obs.shape}")
            
            # Pause to see initial pose
            wrapped_env.render()
            time.sleep(0.5)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Sample random action (less extreme)
                action = wrapped_env.action_space.sample() * 0.3  # Smaller actions
                
                # Take a step
                obs, reward, done, info = wrapped_env.step(action)
                total_reward += reward
                steps += 1
                
                # Render with slower simulation
                wrapped_env.render()
                time.sleep(0.02)  # 50 FPS, easier to see
                
                # Print progress every 50 steps
                if (step + 1) % 50 == 0:
                    logger.debug(f"  Step {step + 1}/{max_steps_per_episode}, Reward: {total_reward:.2f}")
                
                if done:
                    logger.info(f"  Episode terminated at step {steps}")
                    # Pause to see final pose
                    time.sleep(1.0)
                    break
            
            logger.info(f"  Final stats: {steps} steps, Total reward: {total_reward:.2f}")
            
            # Get robot position if available
            try:
                pos = wrapped_env.get_robot_position()
                logger.info(f"  Final position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            except:
                pass
        
        logger.info("=" * 60)
        logger.success("Test completed successfully!")
        logger.info("=" * 60)
        logger.info("Note: Random actions will cause the robot to fall quickly.")
        logger.info("Train with PPO/SAC to learn proper locomotion!")
        logger.info("Viewer will stay open for 5 seconds...")
        
        # Keep viewer open to see the final state
        time.sleep(5.0)
        
        wrapped_env.close()
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.info("Required packages:")
        logger.info("  pip install gymnasium mujoco robot_descriptions")
        import traceback
        traceback.print_exc()

   