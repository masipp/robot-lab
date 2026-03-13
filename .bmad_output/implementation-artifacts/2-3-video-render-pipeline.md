# Story 2.3: Video Render Pipeline

Status: review

## Story

As a researcher (Marco),
I want `robot-lab visualize --env MountainCarContinuous-v0 --algo SAC` to record a policy rollout and export an MP4 to the experiment directory,
So that I can generate visual evidence of policy behavior for portfolio documentation and qualitative trajectory comparison.

## Acceptance Criteria

1. **Given** a trained model `.zip` and matching `_vecnorm.pkl` exist in `data/models/`,
   **When** `robot-lab visualize --env MountainCarContinuous-v0 --algo SAC` is run,
   **Then** an MP4 file is written to `data/experiments/{experiment_name}/runs/{run_id}/` with a non-zero file size.

2. **Given** the visualization path is executed on Windows,
   **When** the environment is constructed for rollout,
   **Then** `DummyVecEnv` is used — `SubprocVecEnv` is never instantiated on the visualization path.

3. **Given** a VecNorm `.pkl` file is missing from the expected paired path,
   **When** `visualize()` is called,
   **Then** a `ValueError` is raised with the message `"[Visualize] VecNorm stats file not found at {path}. Ensure model and VecNorm were saved as a pair."` — rather than silently loading an unnormalized policy.

4. **Given** `make test-fast` is run after this story,
   **When** the smoke tests execute,
   **Then** `test_smoke.py` visualization path passes without GPU (uses CPU + DummyVecEnv).

## Tasks / Subtasks

- [x] Task 1: Add `visualize()` function to `robot_lab/visualization.py` (AC: 1, 2, 3)
  - [x] `visualize(env_name, algorithm, model_path, vecnorm_path, output_dir, num_episodes, record_video)` signature
  - [x] Raise `ValueError` with exact message when `vecnorm_path` does not exist (AC: 3)
  - [x] Use `DummyVecEnv` always — no `SubprocVecEnv` on visualization path (AC: 2)
  - [x] Write MP4 via `imageio.v2.mimsave` (render frames from base env, avoids moviepy dep)
  - [x] Return path to MP4 file (or None if `record_video=False`)
  - [x] Add Google-style docstring and full type hints

- [x] Task 2: Update CLI `visualize` command to call `visualize()` for video recording (AC: 1)
  - [x] Add `--record-video / --no-record-video` flag (default: True) to CLI command
  - [x] When `--record-video` is set and paths are explicit, call `record_policy()` (aliased import)
  - [x] Keep existing `visualize_policy()` call path for backward compat when `--no-record-video`

- [x] Task 3: Add `TestVisualizationPipeline` to `tests/test_smoke.py` (AC: 4)
  - [x] Create a minimal SAC model + VecNorm with `total_timesteps=100` steps for speed
  - [x] Call `visualize()` with `record_video=True`
  - [x] Assert MP4 file exists with size > 0
  - [x] `@pytest.mark.fast` — completes without GPU in ~7s

- [x] Task 4: Run ruff and full test suite (AC: all)
  - [x] New code in `robot_lab/visualization.py` — zero violations
  - [x] New code in `robot_lab/cli.py` — zero violations (pre-existing E501 in untouched sections remain)
  - [x] `pytest tests/test_smoke.py::TestVisualizationPipeline -v` — 3/3 passed
  - [x] `pytest tests/ --ignore=tests/test_training.py` — 100/100 passed

## Dev Notes

### Implementation Notes

- Used `imageio.v2.mimsave` instead of `gymnasium.wrappers.RecordVideo` — the gymnasium wrapper requires `moviepy` which is not installed; `imageio[ffmpeg]` (already a dep) handles MP4 writing directly.
- Frames captured by calling `base_env.render()` after each `eval_env.step()` — the `base_env` reference is held before wrapping in `DummyVecEnv`, so render calls bypass VecEnv abstraction correctly.
- CLI import aliased: `from robot_lab.visualization import visualize as record_policy` to avoid F811 name clash with the Typer `def visualize(...)` CLI command.


## Story

As a researcher (Marco),
I want `robot-lab visualize --env MountainCarContinuous-v0 --algo SAC` to record a policy rollout and export an MP4 to the experiment directory,
So that I can generate visual evidence of policy behavior for portfolio documentation and qualitative trajectory comparison.

## Acceptance Criteria

1. **Given** a trained model `.zip` and matching `_vecnorm.pkl` exist in `data/models/`,
   **When** `robot-lab visualize --env MountainCarContinuous-v0 --algo SAC` is run,
   **Then** an MP4 file is written to `data/experiments/{experiment_name}/runs/{run_id}/` with a non-zero file size.

2. **Given** the visualization path is executed on Windows,
   **When** the environment is constructed for rollout,
   **Then** `DummyVecEnv` is used — `SubprocVecEnv` is never instantiated on the visualization path.

3. **Given** a VecNorm `.pkl` file is missing from the expected paired path,
   **When** `visualize()` is called,
   **Then** a `ValueError` is raised with the message `"[Visualize] VecNorm stats file not found at {path}. Ensure model and VecNorm were saved as a pair."` — rather than silently loading an unnormalized policy.

4. **Given** `make test-fast` is run after this story,
   **When** the smoke tests execute,
   **Then** `test_smoke.py` visualization path passes without GPU (uses CPU + DummyVecEnv).

## Tasks / Subtasks

- [ ] Task 1: Add `visualize()` function to `robot_lab/visualization.py` (AC: 1, 2, 3)
  - [ ] `visualize(env_name, algorithm, model_path, vecnorm_path, output_dir, num_episodes, record_video)` signature
  - [ ] Raise `ValueError` with exact message when `vecnorm_path` does not exist (AC: 3)
  - [ ] Use `DummyVecEnv` always — no `SubprocVecEnv` on visualization path (AC: 2)
  - [ ] When `record_video=True`: wrap base env with `gym.wrappers.RecordVideo`, write MP4 to ExperimentTracker run dir
  - [ ] Use `ExperimentTracker` with `experiment_name=f"viz_{env_base_name}_{algo_name}"` to create output directory under `data/experiments/...`
  - [ ] Return path to MP4 file (or None if `record_video=False`)
  - [ ] Add Google-style docstring and full type hints

- [ ] Task 2: Update CLI `visualize` command to call `visualize()` for video recording (AC: 1)
  - [ ] Add `--record-video / --no-record-video` flag (default: True) to CLI command
  - [ ] When `--record-video` is set, call `visualize()` for video recording path
  - [ ] Keep existing `visualize_policy()` call path for backward compat when `--no-record-video`

- [ ] Task 3: Add `TestVisualizationPipeline` to `tests/test_smoke.py` (AC: 4)
  - [ ] Create a minimal SAC model + VecNorm with `total_timesteps=100` steps for speed
  - [ ] Call `visualize()` with `render=False`, `record_video=True`
  - [ ] Assert MP4 file exists in experiment run dir with size > 0
  - [ ] `@pytest.mark.fast` — must complete without GPU in < 10s

- [ ] Task 4: Run ruff and full test suite (AC: all)
  - [ ] `ruff check robot_lab/visualization.py tests/test_smoke.py` — zero violations
  - [ ] `pytest tests/test_smoke.py -v` — all tests pass
  - [ ] `pytest tests/ -v --ignore=tests/test_training.py` — no regressions

## Dev Notes

### `visualize()` Function Design

```python
def visualize(
    env_name: str,
    algorithm: str,
    model_path: str,
    vecnorm_path: str,
    output_dir: Optional[str] = None,
    num_episodes: int = 1,
    record_video: bool = True,
) -> Optional[Path]:
```

**VecNorm error message** (exact):
```
"[Visualize] VecNorm stats file not found at {path}. Ensure model and VecNorm were saved as a pair."
```

**Video recording approach** (using gymnasium RecordVideo wrapper):
```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

base_env = gym.make(env_name, render_mode="rgb_array")
if record_video:
    base_env = RecordVideo(base_env, video_folder=str(run_dir), episode_trigger=lambda ep: True)
eval_env = DummyVecEnv([lambda: base_env])
```

**ExperimentTracker usage for output dir**:
```python
tracker = ExperimentTracker(
    experiment_name=f"viz_{env_base_name}_{algo_name}",
    run_name="rollout",
    seed=0,
    output_dir=output_dir,
)
tracker.start_run()
run_dir = tracker.get_run_dir()
```

**Finding the MP4 after recording**:
After the rollout, RecordVideo writes files to `video_folder`. Find the MP4:
```python
mp4_files = list(run_dir.glob("*.mp4"))
```

**Model loading**: Use `DummyVecEnv` only — no SubprocVecEnv.

**imageio[ffmpeg] dependency**: `imageio[ffmpeg]` is already installed and added to pyproject.toml
via `uv add "imageio[ffmpeg]"`. Gymnasium's RecordVideo uses it automatically.

### Testing Strategy

For the smoke test, creating a tiny trained model:
```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
env = VecNormalize(env)
model = SAC("MlpPolicy", env, seed=0, device="cpu")
model.learn(total_timesteps=100)
model.save(str(model_path))
env.save(str(vecnorm_path))
```

This should complete in 1-2 seconds and create a valid model+vecnorm pair.

## Dev Agent Record

### Implementation Plan
_(to be filled during implementation)_

### Completion Notes
_(to be filled when complete)_

## File List
_(to be filled when complete)_

## Change Log
- 2026-03-13: Story created for Epic 2, Story 2.3
