# Robot Lab Examples

This directory contains example scripts demonstrating key features and workflows of robot_lab.

## Available Examples

### Control Parameter Tracking
**File**: [`track_control_params.py`](track_control_params.py)

Demonstrates how to:
- Extract environment configuration (control & physics parameters)
- Log configuration with ExperimentTracker
- Verify saved metadata for reproducibility

**Usage**:
```bash
# From repository root
python examples/track_control_params.py

# Or with uv
uv run examples/track_control_params.py
```

**Output**: Creates experiment metadata in `data/experiments/example_control_tracking/`

**Key concepts**:
- `extract_environment_config()`: Get full env config including MuJoCo parameters
- `ExperimentTracker.log_env_config()`: Track control parameters for reproducibility
- `summarize_control_config()`: Generate human-readable summaries

## Adding New Examples

When adding a new example:

1. **Name clearly**: `example_feature_name.py`
2. **Add docstring**: Explain what the example demonstrates
3. **Keep simple**: Focus on one concept per example
4. **Add to this README**: Document usage and key concepts
5. **Use existing API**: Don't introduce new dependencies
6. **Print progress**: Use emoji and clear output for user feedback

## Example Template

```python
"""
Brief description of what this example demonstrates.

Usage:
    python examples/my_example.py
"""

def main():
    """Main function with clear steps."""
    print("🔧 Step 1: Setup...")
    # Implementation
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
```

## Related Documentation

- [Main README](../README.md): Project overview and quickstart
- [Experiment Documentation](../docs/experiments/): Structured experiment plans
- [Metadata System](../docs/metadata_system.md): Overview of metadata tracking
- [API Documentation](../docs/): Detailed API docs
