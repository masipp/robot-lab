import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class GripperEnv(gym.Env):
    def __init__(self, control_mode="torque"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("gripper.xml")
        self.data = mujoco.MjData(self.model)
        self.control_mode = control_mode  # "torque" or "trajectory"

        self.n_joints = self.model.nu  # number of actuators
        obs_dim = self.model.nq + self.model.nv  # qpos + qvel

        # Observation = joint angles + velocities
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if control_mode == "torque":
            # direct torques
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        else:
            # target joint positions (trajectories)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        self.Kp = 100.0
        self.Kd = 2.0

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # Convert action to torques
        if self.control_mode == "torque":
            ctrl = action
        else:
            # PD tracking: torque = Kp*(q_des - q) - Kd*qdot
            q_des = action  # normalized to [-1,1]
            q_des = q_des * 0.5  # scale
            ctrl = self.Kp * (q_des - self.data.qpos) - self.Kd * self.data.qvel

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        # Dummy reward: encourage gripper closure (you’d replace with a real grasp metric)
        reward = -np.sum(np.square(obs))  
        done = False
        return obs, reward, done, False, {}

    def render(self):
        pass  # optional: call mujoco.viewer.launch_passive for visualization
