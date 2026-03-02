# Experiment Documentation Structure

This directory contains detailed documentation for all research experiments conducted in the robot_lab project.

## Directory Structure

```
experiments/
├── README.md (this file)
├── 0_foundations/              # Core infrastructure experiments
│   ├── 001_smooth_locomotion.md
│   ├── 002_*.md
│   └── ...
├── 1_curriculum/               # Curriculum learning basics
│   ├── 010_manual_curriculum.md
│   ├── 011_*.md
│   └── ...
├── 2_representation/           # Representation and generalization
│   ├── 020_*.md
│   └── ...
├── 3_sim2real/                 # Sim2real oriented research
│   ├── 030_*.md
│   └── ...
└── 4_scientific/               # Scientific contribution experiments
    ├── 040_*.md
    └── ...
```

## Naming Convention

### Phase Folders
- **Format**: `<phaseID>_<phase_shortname>/`
- **Examples**: 
  - `0_foundations/`
  - `1_curriculum/`
  - `2_representation/`
  - `3_sim2real/`
  - `4_scientific/`

### Experiment Documents
- **Format**: `<NNN>_<descriptive_shortname>.md`
- **NNN**: Three-digit experiment ID (001, 002, ..., 099, 100, ...)
- **descriptive_shortname**: Short, descriptive name using underscores
- **Examples**:
  - `001_smooth_locomotion.md`
  - `002_sac_vs_ppo.md`
  - `010_manual_curriculum.md`
  - `015_adaptive_difficulty.md`

### Experiment ID Ranges (Guidelines)
- **001-009**: Phase 0 - Core foundations
- **010-019**: Phase 1 - Curriculum learning basics
- **020-029**: Phase 2 - Representation & generalization
- **030-039**: Phase 3 - Sim2real oriented research
- **040-049**: Phase 4 - Scientific contributions
- **050-099**: Additional experiments as needed

## Document Structure

Each experiment document must include the following sections:

### Required Sections

1. **Header**
   - Title
   - Research goal statement
   - Date started
   - Status (Planning, In Progress, Completed, Failed)

2. **Background & Motivation**
   - Why this experiment matters
   - What problem it addresses
   - Context within the research roadmap

3. **Research Questions / Hypotheses**
   - Specific, testable hypotheses
   - Clear research questions
   - Expected outcomes

4. **Experimental Design**
   - Independent variables (what you're changing)
   - Dependent variables (what you're measuring)
   - Control variables (what stays constant)
   - Baseline comparisons
   - Number of seeds/trials

5. **Quantitative Metrics**
   - Exact formulas for metrics
   - How metrics are computed
   - Statistical tests to be used
   - Significance thresholds

6. **Expected Outcomes**
   - What results would confirm hypothesis
   - What results would reject hypothesis
   - Potential confounding factors

7. **Implementation Details**
   - Commands to run experiments
   - Configuration files used
   - Code modifications required
   - Control parameters (kp, actuator type, etc.)

8. **Results Section** (filled during/after experiment)
   - Tables of results
   - Plots and visualizations
   - Statistical analysis
   - Observations

9. **Conclusions & Next Steps**
   - Interpretation of results
   - Lessons learned
   - Future experiments suggested
   - Decision points (go/no-go)

### Optional Sections
- **Related Research**: Papers and prior work
- **Appendix**: Implementation code, detailed derivations
- **Notes & Observations**: Informal notes during execution

## Metadata Requirements

### Control Parameters
**CRITICAL**: All experiments involving physics simulation must document:

```yaml
# environment_config.yaml
control_parameters:
  actuator_type: position  # position | velocity | torque | motor
  kp: 100.0
  kd: 0.5
  gear_ratio: 1.0
  ctrl_range: [-1.0, 1.0]
physics_parameters:
  timestep: 0.01
  gravity: [0, 0, -9.81]
  integrator: euler  # euler | rk4
environment_parameters:
  max_episode_steps: 1000
  physics_steps_per_action: 5
```

**Why YAML?**
- Human-readable and easy to edit
- Better for configuration files
- Supports comments for documentation
- Validated before saving to catch errors early
```

### Git Commit Tracking
Each experiment run should capture the git commit hash for full reproducibility.

## Linking to Results

### Convention
- **Documentation**: `experiments/<phase>/<experiment>.md`
- **Result Data**: `data/experiments/<experiment_folder>/`
- **Models**: `data/experiments/<experiment_folder>/models/`
- **Logs**: `data/experiments/<experiment_folder>/logs/`

### Cross-References
Always include links in experiment docs:
```markdown
**Result Directory**: [data/experiments/smooth_locomotion](../../data/experiments/smooth_locomotion/)
**TensorBoard Logs**: Run `robot-lab tensorboard --logdir data/experiments/smooth_locomotion/logs/`
```

## Status Tracking

Use status badges in experiment documents:
- 🎯 **Planning**: Experiment designed but not started
- 🔄 **In Progress**: Currently running
- ✅ **Completed**: Finished with documented results
- ❌ **Failed**: Did not produce usable results
- ⏸️ **Paused**: Temporarily suspended
- 🔀 **Superseded**: Replaced by newer experiment

## Example Usage

### Creating a New Experiment

1. **Choose experiment ID**: Next available in your phase (e.g., 003)
2. **Create document**: `experiments/0_foundations/003_description.md`
3. **Use template**: Copy structure from existing experiment doc
4. **Fill in design**: Complete all required sections before running
5. **Create data folder**: `mkdir -p data/experiments/003_description`
6. **Run experiments**: Use `--output-dir data/experiments/003_description`
7. **Document results**: Fill in results section as data comes in
8. **Update status**: Change from Planning → In Progress → Completed

### Comparing Experiments

To compare experiments, reference them by ID:
```markdown
This experiment builds on [001](0_foundations/001_smooth_locomotion.md) 
by testing higher kp values...

Comparison with [002](0_foundations/002_sac_vs_ppo.md) shows...
```

## Tools & Automation

### Experiment Tracker Integration
Use `ExperimentTracker` to automatically log metadata:
```python
from robot_lab.experiments import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="001_smooth_locomotion",
    run_name="kp50_seed42",
    base_dir="data/experiments"
)

# Log control parameters
tracker.log_env_config({
    "control_parameters": {"kp": 50, "actuator_type": "position"},
    "physics_parameters": {"timestep": 0.01},
})
```

### Analysis Scripts
Place analysis scripts in `scripts/analysis/` with clear naming:
- `analyze_001_smoothness.py`
- `plot_002_comparison.py`
- `stats_003_significance.py`

## Best Practices

1. **Document before running**: Design experiment fully before execution
2. **Track everything**: Control params, git commit, system info, seeds
3. **Use standardized metrics**: Define metrics in advance with exact formulas
4. **Multiple seeds**: Always run with multiple seeds (minimum 3, preferably 5+)
5. **Statistical rigor**: Use proper significance tests, not just eyeballing plots
6. **Version control**: Commit experiment docs before starting runs
7. **Fail fast**: If experiment design is flawed, stop and redesign
8. **Learn from failures**: Document failed experiments too (what went wrong?)

## Questions?

See [docs/PLAN.md](../docs/PLAN.md) for overall research roadmap and [docs/TODO.md](../docs/TODO.md) for current tasks.
