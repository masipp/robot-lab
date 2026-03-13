"""
Gymnasium environment wrappers for action filtering and temporal control.

These wrappers modify how actions are processed or how frequently the policy
is queried, enabling experiments on trajectory smoothness and control frequency.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Mixin — tracker parameter logging
# ---------------------------------------------------------------------------


class RobotLabWrapperMixin:
    """Mixin that logs wrapper class name and init parameters to ExperimentTracker.

    Apply this mixin alongside ``gym.Wrapper`` subclasses that participate in the
    robot-lab experiment tracking system.  The mixin adds a single utility method
    ``_log_to_tracker()`` — no RL logic lives here.

    Usage::

        class MyWrapper(gym.Wrapper, RobotLabWrapperMixin):
            def __init__(self, env, tracker=None, alpha=0.5):
                super().__init__(env)
                self._log_to_tracker(tracker, {"alpha": alpha})
    """

    def _log_to_tracker(
        self,
        tracker: Any,  # Optional[ExperimentTracker] — kept as Any to avoid circular import
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Merge wrapper identity and params into ``tracker._metadata["config"]``.

        Args:
            tracker: An ``ExperimentTracker`` instance, or ``None`` to skip logging.
            params: Additional keyword parameters to store alongside the wrapper name.
                    Must be JSON-serialisable.
        """
        if tracker is None:
            return

        entry: Dict[str, Any] = {"wrapper": type(self).__name__}
        if params:
            entry.update(params)

        tracker._metadata["config"].update(entry)
        tracker._write()


# ---------------------------------------------------------------------------
# ActionFilterWrapper — YouAreLazy scaffold
# ---------------------------------------------------------------------------


class ActionFilterWrapper(gym.Wrapper, RobotLabWrapperMixin):
    """Scaffold base class for action-filtering Gym wrappers.

    Subclass this and implement ``_apply_filter(action)`` to define your
    filtering strategy (low-pass, EMA, splines, etc.).  The infrastructure —
    tracker logging, step delegation — is provided here so you can focus on
    the filter logic itself.

    The ``_apply_filter`` method intentionally raises ``NotImplementedError``
    (YouAreLazy boundary).  Implementing the filter is a *learning-critical*
    task for the robot-lab research roadmap.

    Args:
        env: The Gymnasium environment to wrap.
        tracker: Optional ``ExperimentTracker`` instance.  When provided, the
                 wrapper class name and any extra ``kwargs`` are written to
                 ``metadata.json["config"]`` at construction time.
        **kwargs: Additional keyword arguments passed to ``_log_to_tracker``
                  as parameter metadata.

    Example::

        class MyEMAFilter(ActionFilterWrapper):
            def __init__(self, env, alpha=0.7, **kwargs):
                super().__init__(env, **kwargs)
                self.alpha = alpha

            def _apply_filter(self, action):
                # implement EMA logic here
                ...

        env = gym.make("Walker2d-v5")
        env = MyEMAFilter(env, alpha=0.7, tracker=tracker)
    """

    def __init__(
        self,
        env: gym.Env,
        tracker: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env)
        self._log_to_tracker(tracker, kwargs if kwargs else None)

    def _apply_filter(self, action: np.ndarray) -> np.ndarray:
        """Apply the action filter.  Subclasses MUST override this method.

        Args:
            action: Raw action array from the policy.

        Returns:
            Filtered action array to be passed to the inner environment.

        Raises:
            NotImplementedError: Always — this is the YouAreLazy boundary.
        """
        raise NotImplementedError("Implement _apply_filter() to define your filtering logic.")

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Filter action then delegate to the wrapped environment.

        Args:
            action: Raw action array from the policy.

        Returns:
            obs, reward, terminated, truncated, info from the inner environment.
        """
        filtered_action = self._apply_filter(action)
        return self.env.step(filtered_action)


class ActionRepeatWrapper(gym.Wrapper):
    """
    Frameskip/Action repeat wrapper.
    
    Policy is queried every N environment steps. The same action is repeated
    for N steps, and observations are only returned after N steps.
    
    This is the standard frameskip approach used in Atari and many robotics tasks.
    Reduces effective control frequency by a factor of N.
    
    Args:
        env: The environment to wrap
        n: Number of times to repeat each action (frameskip factor)
    
    Example:
        >>> env = gym.make("A1Quadruped-v0")
        >>> env = ActionRepeatWrapper(env, n=4)  # 12.5Hz control at 50Hz sim
    """

    def __init__(self, env: gym.Env, n: int = 1):
        super().__init__(env)
        if n < 1:
            raise ValueError(f"action_repeat n must be >= 1, got {n}")
        self.n = n

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action N times and accumulate reward.
        
        Args:
            action: Action from policy
        
        Returns:
            obs: Observation after N steps
            total_reward: Sum of rewards over N steps
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Info dict from last step
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.n):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class ExponentialMovingAverageFilter(gym.Wrapper):
    """
    Exponential moving average action filter.
    
    Smooths policy output using EMA: action_t = α * policy_t + (1-α) * action_{t-1}
    
    - α = 1.0: No filtering (pass-through)
    - α = 0.7: Light smoothing
    - α = 0.5: Medium smoothing  
    - α = 0.3: Heavy smoothing
    
    Lower α introduces more lag but reduces jerk. Policy is still queried every step,
    but output actions are smoothed.
    
    Args:
        env: The environment to wrap
        alpha: EMA coefficient (0 < α <= 1). Higher = less smoothing, more responsive.
    
    Example:
        >>> env = gym.make("A1Quadruped-v0")
        >>> env = ExponentialMovingAverageFilter(env, alpha=0.5)
    """

    def __init__(self, env: gym.Env, alpha: float = 1.0):
        super().__init__(env)
        if not 0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        
        self.alpha = alpha
        self.previous_action: Optional[np.ndarray] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and clear action history."""
        self.previous_action = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply EMA filter to action before passing to environment.
        
        Args:
            action: Raw action from policy
        
        Returns:
            obs, reward, terminated, truncated, info from environment
        """
        if self.previous_action is None:
            # First step: no previous action to blend with
            filtered_action = action
        else:
            # EMA: blend current action with previous
            filtered_action = self.alpha * action + (1 - self.alpha) * self.previous_action

        self.previous_action = filtered_action.copy()

        return self.env.step(filtered_action)


class LowPassFilter(gym.Wrapper):
    """
    Low-pass filter for action smoothing.
    
    Implements a simple first-order low-pass filter (RC filter):
    action_t = action_t-1 + dt/(τ + dt) * (policy_t - action_t-1)
    
    Where τ is the time constant. Smaller τ = faster response, less smoothing.
    
    Args:
        env: The environment to wrap
        tau: Filter time constant in seconds (e.g., 0.1 for 10Hz cutoff)
        dt: Environment timestep in seconds (default: 0.02 for 50Hz)
    
    Example:
        >>> env = gym.make("A1Quadruped-v0")
        >>> env = LowPassFilter(env, tau=0.1, dt=0.02)
    """

    def __init__(self, env: gym.Env, tau: float = 0.1, dt: float = 0.02):
        super().__init__(env)
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        
        self.tau = tau
        self.dt = dt
        self.alpha = dt / (tau + dt)  # Equivalent EMA coefficient
        self.previous_action: Optional[np.ndarray] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and clear action history."""
        self.previous_action = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply low-pass filter to action.
        
        Args:
            action: Raw action from policy
        
        Returns:
            obs, reward, terminated, truncated, info from environment
        """
        if self.previous_action is None:
            filtered_action = action
        else:
            # Low-pass filter: exponential approach to target
            filtered_action = self.previous_action + self.alpha * (
                action - self.previous_action
            )

        self.previous_action = filtered_action.copy()

        return self.env.step(filtered_action)


def create_action_wrapper(
    env: gym.Env, config: Dict[str, Any]
) -> gym.Env:
    """
    Factory function to create action wrappers from config dictionary.
    
    Args:
        env: Environment to wrap
        config: Configuration dict with keys:
            - 'action_repeat': {'n': int} for frameskip
            - 'action_filter': {'type': str, 'alpha': float} for EMA
            - 'action_filter': {'type': str, 'tau': float, 'dt': float} for low-pass
    
    Returns:
        Wrapped environment
    
    Example:
        >>> config = {'action_repeat': {'n': 4}}
        >>> env = create_action_wrapper(env, config)
        >>> config = {'action_filter': {'type': 'exponential_moving_average', 'alpha': 0.5}}
        >>> env = create_action_wrapper(env, config)
    """
    # Apply action repeat (frameskip) if configured
    if 'action_repeat' in config:
        repeat_config = config['action_repeat']
        n = repeat_config.get('n', 1)
        if n > 1:
            env = ActionRepeatWrapper(env, n=n)
    
    # Apply action filter if configured
    if 'action_filter' in config:
        filter_config = config['action_filter']
        filter_type = filter_config.get('type', '').lower()
        
        if filter_type == 'exponential_moving_average':
            alpha = filter_config.get('alpha', 1.0)
            env = ExponentialMovingAverageFilter(env, alpha=alpha)
        
        elif filter_type == 'low_pass':
            tau = filter_config.get('tau', 0.1)
            dt = filter_config.get('dt', 0.02)
            env = LowPassFilter(env, tau=tau, dt=dt)
        
        elif filter_type:
            raise ValueError(
                f"Unknown action filter type: {filter_type}. "
                f"Supported: 'exponential_moving_average', 'low_pass'"
            )
    
    return env
