"""Utilities for computing smooth locomotion metrics from trained policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


MetricGroups = Dict[str, List[str]]


AVAILABLE_METRICS = {
    "jerk_mean",
    "jerk_max",
    "accel_variance",
    "action_delta_mean",
    "action_delta_max",
    "action_frequency_power",
    "forward_distance",
    "episode_length",
    "energy_cost",
    "fall_rate",
    "orientation_variance",
    "base_height_variance",
}


def flatten_metric_groups(metric_groups: Optional[MetricGroups]) -> List[str]:
    """Flatten grouped metric config into an ordered unique list.

    Args:
        metric_groups: Metrics grouped by category from YAML config.

    Returns:
        Ordered list of unique metric names.
    """
    if not metric_groups:
        return []

    flattened: List[str] = []
    for group_metrics in metric_groups.values():
        if not isinstance(group_metrics, list):
            continue
        for metric_name in group_metrics:
            if metric_name not in flattened:
                flattened.append(metric_name)
    return flattened


def _extract_roll_pitch(qpos: np.ndarray) -> np.ndarray:
    """Convert quaternion in qpos[3:7] to roll/pitch Euler angles."""
    qw, qx, qy, qz = qpos[3:7]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    return np.array([roll, pitch], dtype=np.float64)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert numeric-like value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_requested_metrics(
    requested_metrics: Iterable[str],
    actions: np.ndarray,
    forward_distances: List[float],
    episode_lengths: List[int],
    episode_energy_costs: List[float],
    fall_flags: List[int],
    roll_pitch_series: np.ndarray,
    base_height_series: np.ndarray,
    dt: float,
) -> Dict[str, Optional[float]]:
    """Compute requested metrics from rollout traces."""
    computed: Dict[str, Optional[float]] = {}

    action_deltas = (
        np.diff(actions, axis=0) if len(actions) > 1 else np.empty((0, actions.shape[1]), dtype=np.float64)
    )
    action_accel = (
        np.diff(action_deltas, axis=0) / max(dt, 1e-8)
        if len(action_deltas) > 1
        else np.empty((0, actions.shape[1]), dtype=np.float64)
    )
    action_jerk = (
        np.diff(action_accel, axis=0) / max(dt, 1e-8)
        if len(action_accel) > 1
        else np.empty((0, actions.shape[1]), dtype=np.float64)
    )

    for metric_name in requested_metrics:
        if metric_name == "jerk_mean":
            value = np.linalg.norm(action_jerk, axis=1).mean() if len(action_jerk) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "jerk_max":
            value = np.linalg.norm(action_jerk, axis=1).max() if len(action_jerk) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "accel_variance":
            value = np.var(action_accel) if len(action_accel) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "action_delta_mean":
            value = np.linalg.norm(action_deltas, axis=1).mean() if len(action_deltas) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "action_delta_max":
            value = np.linalg.norm(action_deltas, axis=1).max() if len(action_deltas) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "action_frequency_power":
            if len(actions) > 2:
                freqs = np.fft.rfftfreq(len(actions), d=max(dt, 1e-8))
                fft_vals = np.fft.rfft(actions, axis=0)
                power = np.abs(fft_vals) ** 2
                high_freq_mask = freqs >= 10.0
                value = power[high_freq_mask].sum() / max(power.sum(), 1e-8)
            else:
                value = np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "forward_distance":
            value = np.mean(forward_distances) if forward_distances else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "episode_length":
            value = np.mean(episode_lengths) if episode_lengths else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "energy_cost":
            value = np.mean(episode_energy_costs) if episode_energy_costs else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "fall_rate":
            value = np.mean(fall_flags) if fall_flags else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "orientation_variance":
            value = np.var(roll_pitch_series) if len(roll_pitch_series) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        elif metric_name == "base_height_variance":
            value = np.var(base_height_series) if len(base_height_series) > 0 else np.nan
            computed[metric_name] = _safe_float(value, default=np.nan)
        else:
            computed[metric_name] = None

    return computed


def evaluate_smoothness_metrics(
    env_name: str,
    algorithm: str,
    model_path: str,
    vecnorm_path: str,
    requested_metrics: List[str],
    num_episodes: int = 10,
    seed: int = 42,
    env_kwargs: Optional[Dict[str, Any]] = None,
    env_wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
) -> Dict[str, Any]:
    """Evaluate smoothness/performance metrics for a trained policy.

    Args:
        env_name: Gymnasium environment ID.
        algorithm: RL algorithm name (SAC or PPO).
        model_path: Path to saved policy.
        vecnorm_path: Path to saved VecNormalize stats.
        requested_metrics: Flattened list of metrics to compute.
        num_episodes: Number of deterministic episodes to evaluate.
        seed: Base random seed for reproducibility.
        env_kwargs: Optional kwargs passed to gym.make.
        env_wrapper_fn: Optional wrapper factory used during training.

    Returns:
        Dictionary containing requested metrics and evaluation metadata.
    """
    env_kwargs = env_kwargs or {}

    base_env = gym.make(env_name, **env_kwargs)
    if env_wrapper_fn is not None:
        base_env = env_wrapper_fn(base_env)

    vec_env = DummyVecEnv([lambda: base_env])
    vecnorm_file = Path(vecnorm_path)
    if vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    if algorithm.upper() == "SAC":
        model = SAC.load(model_path)
    elif algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    else:
        vec_env.close()
        raise ValueError(f"Unknown algorithm for metrics evaluation: {algorithm}")

    unwrapped_env = base_env.unwrapped
    dt = 0.02
    if hasattr(unwrapped_env, "model") and hasattr(unwrapped_env.model, "opt"):
        dt = float(unwrapped_env.model.opt.timestep) * 5.0

    all_actions: List[np.ndarray] = []
    all_roll_pitch: List[np.ndarray] = []
    all_base_heights: List[float] = []

    forward_distances: List[float] = []
    episode_lengths: List[int] = []
    episode_energy_costs: List[float] = []
    fall_flags: List[int] = []

    for episode_idx in range(num_episodes):
        if hasattr(base_env, "reset"):
            try:
                base_env.reset(seed=seed + episode_idx)
            except TypeError:
                base_env.reset()
        obs = vec_env.reset()
        done = False

        episode_steps = 0
        energy_sum = 0.0
        start_x = _safe_float(getattr(unwrapped_env.data, "qpos", [0.0])[0])

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_vector = np.asarray(action[0], dtype=np.float64)

            obs, _reward, done_flags, _infos = vec_env.step(action)
            done = bool(done_flags[0])

            all_actions.append(action_vector)
            energy_sum += float(np.sum(np.square(action_vector)))
            episode_steps += 1

            qpos = np.asarray(unwrapped_env.data.qpos, dtype=np.float64)
            all_roll_pitch.append(_extract_roll_pitch(qpos))
            all_base_heights.append(float(qpos[2]))

        final_qpos = np.asarray(unwrapped_env.data.qpos, dtype=np.float64)
        end_x = float(final_qpos[0])
        final_height = float(final_qpos[2])
        final_roll_pitch = _extract_roll_pitch(final_qpos)

        forward_distances.append(end_x - start_x)
        episode_lengths.append(episode_steps)
        episode_energy_costs.append(energy_sum / max(episode_steps, 1))

        is_fall = bool(
            final_height < 0.15 or abs(final_roll_pitch[0]) > 1.0 or abs(final_roll_pitch[1]) > 1.0
        )
        fall_flags.append(1 if is_fall else 0)

    vec_env.close()

    if all_actions:
        action_array = np.vstack(all_actions)
    else:
        action_dim = int(base_env.action_space.shape[0])
        action_array = np.empty((0, action_dim), dtype=np.float64)

    roll_pitch_array = (
        np.vstack(all_roll_pitch) if all_roll_pitch else np.empty((0, 2), dtype=np.float64)
    )
    base_height_array = np.asarray(all_base_heights, dtype=np.float64)

    requested_set = [metric for metric in requested_metrics]
    computed_metrics = _compute_requested_metrics(
        requested_metrics=requested_set,
        actions=action_array,
        forward_distances=forward_distances,
        episode_lengths=episode_lengths,
        episode_energy_costs=episode_energy_costs,
        fall_flags=fall_flags,
        roll_pitch_series=roll_pitch_array,
        base_height_series=base_height_array,
        dt=dt,
    )

    unsupported_metrics = [name for name in requested_set if name not in AVAILABLE_METRICS]
    if unsupported_metrics:
        logger.warning(f"Unsupported metrics requested and set to null: {unsupported_metrics}")

    return {
        "episodes": num_episodes,
        "dt": dt,
        "requested_metrics": requested_set,
        "unsupported_metrics": unsupported_metrics,
        "computed_metrics": computed_metrics,
    }
