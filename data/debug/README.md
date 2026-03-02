# Debug Configurations

This directory contains JSON configuration files for debugging training and visualization runs in VS Code.

## Structure

Debug configuration files follow the naming convention:
- Training: `train_{environment}_{algorithm}.json`
- Visualization: `visualize_{environment}_{algorithm}.json`

## Usage

### From VS Code (F5)

1. Open the Run and Debug panel (Ctrl+Shift+D)
2. Select a debug configuration from the dropdown
3. Press F5 to start debugging

All debug configurations automatically load parameters from these JSON files.

### From Command Line

You can also use these configs from the command line:

```bash
# Training
robot-lab train-cmd --debug-config train_walker2d_sac

# Visualization
robot-lab visualize --debug-config visualize_walker2d_sac
```

### Overriding Parameters

Command line arguments take precedence over debug config values:

```bash
# Use debug config but override the seed
robot-lab train-cmd --debug-config train_walker2d_sac --seed 123
```

## Configuration Format

### Training Configuration

```json
{
  "env": "Walker2d-v5",
  "algo": "SAC",
  "seed": 42,
  "config": null,
  "output_dir": null,
  "eval_freq": 10000,
  "eval_episodes": 10,
  "save_freq": null,
  "checkpoints": false
}
```

### Visualization Configuration

```json
{
  "env": "Walker2d-v5",
  "algo": "SAC",
  "model_path": null,
  "vecnorm_path": null,
  "episodes": 3,
  "no_render": false,
  "no_plot": false,
  "output_dir": null
}
```

## Creating New Configs

You can create new debug configurations by copying and modifying existing files, or use the utility:

```python
from robot_lab.utils import create_debug_config_template

# Create a new training config
create_debug_config_template("train_myenv_sac", config_type="train")

# Create a new visualization config
create_debug_config_template("visualize_myenv_sac", config_type="visualize")
```

## Notes

- `null` values mean "use default" or "auto-detect"
- The `data/` directory is gitignored except for these debug config files
- Edit these files to customize your debugging workflow
