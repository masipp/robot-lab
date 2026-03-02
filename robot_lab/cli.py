"""Command-line interface for robot_lab."""

import typer
from typing import Optional
from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from loguru import logger

from robot_lab.training import train as train_agent
from robot_lab.visualization import visualize_policy
from robot_lab.utils.logger import configure_logger
from robot_lab.utils.debug_config import load_debug_config
from robot_lab.utils.paths import get_logs_dir
from robot_lab.utils.run_selector import list_training_runs, format_run_option, get_full_env_name
from robot_lab.envs import get_env_registry, get_env_info, EnvCategory, EnvDifficulty

app = typer.Typer(
    name="robot-lab",
    help="Reinforcement learning playground for robotic environments",
    add_completion=False
)
console = Console()


@app.command()
def train(
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment name (e.g., 'Walker2d-v5', 'HalfCheetah-v5')"
    ),
    algo: Optional[str] = typer.Option(
        None,
        "--algo",
        "-a",
        help="Algorithm to use: 'SAC' or 'PPO'"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to custom hyperparameters JSON file"
    ),
    env_config: Optional[Path] = typer.Option(
        None,
        "--env-config",
        help="Path to environment configuration YAML file (control params, physics, etc.)"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom output directory for models and logs"
    ),
    eval_freq: Optional[int] = typer.Option(
        None,
        "--eval-freq",
        help="Evaluation frequency in timesteps"
    ),
    eval_episodes: Optional[int] = typer.Option(
        None,
        "--eval-episodes",
        help="Number of episodes for evaluation"
    ),
    save_freq: Optional[int] = typer.Option(
        None,
        "--save-freq",
        help="Checkpoint save frequency in timesteps (None = disabled)"
    ),
    checkpoints: Optional[bool] = typer.Option(
        None,
        "--checkpoints",
        help="Enable intermediate checkpoint saving"
    ),
    debug_config: Optional[str] = typer.Option(
        None,
        "--debug-config",
        "-d",
        help="Load parameters from debug config file (e.g., 'train_walker2d_sac')"
    ),
):
    """Train a reinforcement learning agent on a robotic environment."""
    
    # Load debug config if provided
    if debug_config:
        debug_params = load_debug_config(debug_config)
        logger.info(f"Loaded debug config: {debug_config}")
        
        # Use debug config values as defaults, but allow CLI args to override
        env = env or debug_params.get("env")
        algo = algo or debug_params.get("algo")
        config = config or (Path(debug_params["config"]) if debug_params.get("config") else None)
        seed = seed if seed is not None else debug_params.get("seed", 42)
        output_dir = output_dir or (Path(debug_params["output_dir"]) if debug_params.get("output_dir") else None)
        eval_freq = eval_freq if eval_freq is not None else debug_params.get("eval_freq", 10000)
        eval_episodes = eval_episodes if eval_episodes is not None else debug_params.get("eval_episodes", 10)
        save_freq = save_freq if save_freq is not None else debug_params.get("save_freq")
        checkpoints = checkpoints if checkpoints is not None else debug_params.get("checkpoints", False)
    else:
        # Use default values if not provided via CLI or debug config
        seed = seed if seed is not None else 42
        eval_freq = eval_freq if eval_freq is not None else 10000
        eval_episodes = eval_episodes if eval_episodes is not None else 10
        checkpoints = checkpoints if checkpoints is not None else False
    
    # Validate required parameters
    if not env:
        rprint("[bold red]✗ Error:[/bold red] --env is required (or use --debug-config)")
        raise typer.Exit(code=1)
    if not algo:
        rprint("[bold red]✗ Error:[/bold red] --algo is required (or use --debug-config)")
        raise typer.Exit(code=1)
    
    # Configure logger
    configure_logger(output_dir=str(output_dir) if output_dir else None)
    
    logger.info("Starting training command")
    logger.info(f"Environment: {env}, Algorithm: {algo.upper()}, Seed: {seed}")
    
    # Display training configuration
    table = Table(title="Training Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Environment", env)
    table.add_row("Algorithm", algo.upper())
    table.add_row("Config File", str(config) if config else "Auto-detect")
    table.add_row("Env Config File", str(env_config) if env_config else "None")
    table.add_row("Seed", str(seed))
    table.add_row("Output Directory", str(output_dir) if output_dir else "Current directory")
    table.add_row("Eval Frequency", f"{eval_freq:,} timesteps")
    table.add_row("Eval Episodes", str(eval_episodes))
    table.add_row("Checkpoints", "Enabled" if checkpoints else "Disabled")
    if save_freq:
        table.add_row("Save Frequency", f"{save_freq:,} timesteps")
    
    console.print(table)
    console.print()
    
    try:
        # Run training
        model, model_path, vecnorm_path = train_agent(
            env_name=env,
            algorithm=algo,
            config_path=str(config) if config else None,
            env_config_path=str(env_config) if env_config else None,
            seed=seed,
            output_dir=str(output_dir) if output_dir else None,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            save_freq=save_freq,
            use_checkpoints=checkpoints,
        )
        
        logger.success("Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"VecNormalize saved to: {vecnorm_path}")
        
        rprint("[bold green]✓ Training completed successfully![/bold green]")
        rprint(f"[cyan]Model saved to:[/cyan] {model_path}")
        rprint(f"[cyan]VecNormalize saved to:[/cyan] {vecnorm_path}")
        rprint("\n[yellow]To visualize the learned policy, run:[/yellow]")
        rprint(f"[bold]robot-lab visualize --env {env} --algo {algo}[/bold]")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        rprint(f"[bold red]✗ Training failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def visualize(
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment name (must match training)"
    ),
    algo: Optional[str] = typer.Option(
        None,
        "--algo",
        "-a",
        help="Algorithm used: 'SAC' or 'PPO'"
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to trained model (None = auto-detect)"
    ),
    vecnorm_path: Optional[Path] = typer.Option(
        None,
        "--vecnorm-path",
        "-v",
        help="Path to VecNormalize stats (None = auto-detect)"
    ),
    episodes: Optional[int] = typer.Option(
        None,
        "--episodes",
        "-n",
        help="Number of episodes to run"
    ),
    no_render: Optional[bool] = typer.Option(
        None,
        "--no-render",
        help="Disable environment rendering"
    ),
    no_plot: Optional[bool] = typer.Option(
        None,
        "--no-plot",
        help="Disable performance plots"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom output directory"
    ),
    env_config: Optional[Path] = typer.Option(
        None,
        "--env-config",
        help="Path to environment configuration YAML file (control params, physics, etc.)"
    ),
    debug_config: Optional[str] = typer.Option(
        None,
        "--debug-config",
        "-d",
        help="Load parameters from debug config file (e.g., 'visualize_walker2d_sac')"
    ),
):
    """Visualize a trained policy."""
    
    # Load debug config if provided
    if debug_config:
        debug_params = load_debug_config(debug_config)
        logger.info(f"Loaded debug config: {debug_config}")
        
        # Use debug config values as defaults, but allow CLI args to override
        env = env or debug_params.get("env")
        algo = algo or debug_params.get("algo")
        model_path = model_path or (Path(debug_params["model_path"]) if debug_params.get("model_path") else None)
        vecnorm_path = vecnorm_path or (Path(debug_params["vecnorm_path"]) if debug_params.get("vecnorm_path") else None)
        episodes = episodes if episodes is not None else debug_params.get("episodes", 3)
        no_render = no_render if no_render is not None else debug_params.get("no_render", False)
        no_plot = no_plot if no_plot is not None else debug_params.get("no_plot", False)
        output_dir = output_dir or (Path(debug_params["output_dir"]) if debug_params.get("output_dir") else None)
    
    # Interactive run selection if env/algo not provided
    if not env or not algo:
        logs_dir = get_logs_dir(str(output_dir) if output_dir else None)
        runs = list_training_runs(logs_dir, limit=20)
        
        if not runs:
            rprint("[bold red]✗ Error:[/bold red] No training runs found. Train a model first.")
            raise typer.Exit(code=1)
        
        # Display available runs
        console.print("\n[bold cyan]Available Training Runs:[/bold cyan]")
        console.print("=" * 80)
        
        for idx, (run_dir, run_info) in enumerate(runs, 1):
            is_latest = (idx == 1)
            option_text = format_run_option(run_info, idx, is_latest)
            if is_latest:
                console.print(f"[bold green]{option_text}[/bold green]")
            else:
                console.print(option_text)
        
        console.print("=" * 80)
        
        # Prompt for selection
        selection = typer.prompt(
            "\nSelect run number (or press Enter for latest)",
            default="1",
            type=str
        )
        
        try:
            selected_idx = int(selection) - 1
            if selected_idx < 0 or selected_idx >= len(runs):
                rprint("[bold red]✗ Error:[/bold red] Invalid selection")
                raise typer.Exit(code=1)
        except ValueError:
            rprint("[bold red]✗ Error:[/bold red] Please enter a valid number")
            raise typer.Exit(code=1)
        
        # Get selected run info
        selected_run_dir, selected_info = runs[selected_idx]
        
        # Set parameters from selected run
        env = get_full_env_name(selected_info)
        if not env:
            # Fallback: try to infer from training log or prompt user
            rprint(f"[yellow]⚠ Warning:[/yellow] Could not auto-detect full environment name for '{selected_info['env']}'")
            env = typer.prompt("Enter full environment name (e.g., 'Walker2d-v5')")
        
        algo = selected_info['algo']
        model_path = selected_info['model_file']
        vecnorm_path = selected_info['vecnorm_file']
        
        console.print(f"\n[bold green]✓ Selected:[/bold green] {selected_info['run_id']}")
    
    # Set default values if not provided
    episodes = episodes if episodes is not None else 3
    no_render = no_render if no_render is not None else False
    no_plot = no_plot if no_plot is not None else False
    
    # Validate required parameters
    if not env:
        rprint("[bold red]✗ Error:[/bold red] --env is required (or use --debug-config)")
        raise typer.Exit(code=1)
    if not algo:
        rprint("[bold red]✗ Error:[/bold red] --algo is required (or use --debug-config)")
        raise typer.Exit(code=1)
    
    # Configure logger
    configure_logger(output_dir=str(output_dir) if output_dir else None)
    
    logger.info("Starting visualization command")
    logger.info(f"Environment: {env}, Algorithm: {algo.upper()}, Episodes: {episodes}")
    
    # Display visualization configuration
    table = Table(title="Visualization Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Environment", env)
    table.add_row("Algorithm", algo.upper())
    table.add_row("Model Path", str(model_path) if model_path else "Auto-detect")
    table.add_row("VecNorm Path", str(vecnorm_path) if vecnorm_path else "Auto-detect")
    table.add_row("Env Config File", str(env_config) if env_config else "None")
    table.add_row("Episodes", str(episodes))
    table.add_row("Render", "No" if no_render else "Yes")
    table.add_row("Save Plot", "No" if no_plot else "Yes")
    
    console.print(table)
    console.print()
    
    try:
        # Run visualization
        rewards, lengths = visualize_policy(
            env_name=env,
            algorithm=algo,
            model_path=str(model_path) if model_path else None,
            vecnorm_path=str(vecnorm_path) if vecnorm_path else None,
            env_config_path=str(env_config) if env_config else None,
            num_episodes=episodes,
            render=not no_render,
            save_plot=not no_plot,
            output_dir=str(output_dir) if output_dir else None,
        )
        
        logger.success("Visualization completed successfully!")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        rprint(f"[bold red]✗ Visualization failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def tensorboard(
    logdir: Optional[Path] = typer.Option(
        None,
        "--logdir",
        "-l",
        help="Path to logs directory (None = ./logs)"
    ),
    port: int = typer.Option(
        6006,
        "--port",
        "-p",
        help="Port to run TensorBoard on"
    ),
):
    """Launch TensorBoard to view training logs."""
    import subprocess
    
    logger.info("Starting TensorBoard command")
    
    if logdir is None:
        logdir = Path("logs")
    
    if not logdir.exists():
        logger.error(f"Log directory not found: {logdir}")
        rprint(f"[bold red]✗ Log directory not found:[/bold red] {logdir}")
        raise typer.Exit(code=1)
    
    logger.info(f"Launching TensorBoard on port {port} with logdir: {logdir}")
    rprint(f"[cyan]Launching TensorBoard on port {port}...[/cyan]")
    rprint(f"[cyan]Log directory:[/cyan] {logdir}")
    rprint(f"\n[yellow]Access TensorBoard at:[/yellow] http://localhost:{port}")
    rprint("[yellow]Press Ctrl+C to stop[/yellow]\n")
    
    try:
        subprocess.run(
            ["tensorboard", "--logdir", str(logdir), "--port", str(port)],
            check=True
        )
    except KeyboardInterrupt:
        logger.info("TensorBoard stopped by user")
        rprint("\n[cyan]TensorBoard stopped.[/cyan]")
    except FileNotFoundError:
        logger.error("TensorBoard not found")
        rprint("[bold red]✗ TensorBoard not found. Install with:[/bold red]")
        rprint("[bold]pip install tensorboard[/bold]")
        raise typer.Exit(code=1)


@app.command()
def info():
    """Display information about robot_lab."""
    from robot_lab import __version__
    
    console.print(f"\n[bold cyan]robot-lab v{__version__}[/bold cyan]")
    console.print("[yellow]Reinforcement learning playground for robotic environments[/yellow]\n")
    
    console.print("[bold]Available Commands:[/bold]")
    console.print("  [cyan]train[/cyan]       - Train a new agent")
    console.print("  [cyan]visualize[/cyan]   - Visualize a trained policy")
    console.print("  [cyan]tensorboard[/cyan] - Launch TensorBoard")
    console.print("  [cyan]list-envs[/cyan]   - List available environments")
    console.print("  [cyan]env-info[/cyan]    - Show detailed environment information")
    console.print("  [cyan]info[/cyan]        - Show this information")
    
    console.print("\n[bold]Supported Algorithms:[/bold]")
    console.print("  • SAC (Soft Actor-Critic) - Continuous actions only")
    console.print("  • PPO (Proximal Policy Optimization) - Discrete & continuous actions")
    
    console.print("\n[bold]Example Environments:[/bold]")
    console.print("  • Walker2d-v5")
    console.print("  • HalfCheetah-v5")
    console.print("  • MountainCarContinuous-v0")
    console.print("  • GripperEnv-v0 (custom)")
    console.print("  • A1Quadruped-v0 (custom)")
    
    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  [bold cyan]robot-lab train --env Walker2d-v5 --algo SAC[/bold cyan]")
    console.print("  [bold cyan]robot-lab visualize --env Walker2d-v5 --algo SAC[/bold cyan]")
    console.print("  [bold cyan]robot-lab list-envs --category mujoco[/bold cyan]")
    console.print()


@app.command()
def list_envs(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category: classic_control, mujoco, locomotion, manipulation, custom"
    ),
    difficulty: Optional[str] = typer.Option(
        None,
        "--difficulty",
        "-d",
        help="Filter by difficulty: easy, medium, hard, expert"
    ),
    custom_only: bool = typer.Option(
        False,
        "--custom-only",
        help="Show only custom robot_lab environments"
    ),
    builtin_only: bool = typer.Option(
        False,
        "--builtin-only",
        help="Show only built-in Gymnasium environments"
    ),
):
    """List available environments in the registry."""
    
    logger.info("Listing available environments")
    
    registry = get_env_registry()
    
    # Parse filters
    category_filter = None
    if category:
        try:
            category_filter = EnvCategory(category.lower())
        except ValueError:
            rprint(f"[bold red]✗ Invalid category:[/bold red] {category}")
            rprint(f"[yellow]Valid categories:[/yellow] {', '.join([c.value for c in EnvCategory])}")
            raise typer.Exit(code=1)
    
    difficulty_filter = None
    if difficulty:
        try:
            difficulty_filter = EnvDifficulty(difficulty.lower())
        except ValueError:
            rprint(f"[bold red]✗ Invalid difficulty:[/bold red] {difficulty}")
            rprint(f"[yellow]Valid difficulties:[/yellow] {', '.join([d.value for d in EnvDifficulty])}")
            raise typer.Exit(code=1)
    
    # Handle custom/builtin filters
    include_custom = not builtin_only
    
    # Get filtered environments
    envs = registry.list_envs(
        category=category_filter,
        difficulty=difficulty_filter,
        include_custom=include_custom
    )
    
    # Further filter if custom_only
    if custom_only:
        envs = [e for e in envs if e.is_custom]
    
    if not envs:
        rprint("[yellow]No environments match the specified filters.[/yellow]")
        return
    
    # Display results in a table
    table = Table(title=f"Available Environments ({len(envs)} found)")
    table.add_column("Environment ID", style="cyan", no_wrap=True)
    table.add_column("Category", style="blue")
    table.add_column("Difficulty", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Algorithm", style="green")
    table.add_column("Custom", style="yellow")
    
    for metadata in envs:
        custom_mark = "✓" if metadata.is_custom else ""
        table.add_row(
            metadata.env_id,
            metadata.category.value,
            metadata.difficulty.value,
            metadata.description[:50] + "..." if len(metadata.description) > 50 else metadata.description,
            metadata.default_algorithm,
            custom_mark
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(envs)} environments[/dim]")
    console.print(f"[dim]Use 'robot-lab env-info --env <ENV_ID>' for detailed information[/dim]\n")


@app.command()
def env_info(
    env: str = typer.Option(
        ...,
        "--env",
        "-e",
        help="Environment ID to get information about"
    ),
):
    """Display detailed information about a specific environment."""
    
    logger.info(f"Getting info for environment: {env}")
    
    registry = get_env_registry()
    metadata = registry.get_metadata(env)
    
    if metadata is None:
        rprint(f"[bold red]✗ Environment not found:[/bold red] {env}")
        rprint("[yellow]Use 'robot-lab list-envs' to see available environments[/yellow]")
        raise typer.Exit(code=1)
    
    # Display detailed information
    console.print()
    console.print(f"[bold cyan]{metadata.env_id}[/bold cyan]")
    console.print(f"[dim]{metadata.description}[/dim]\n")
    
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="cyan", width=25)
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Category", metadata.category.value)
    info_table.add_row("Difficulty", metadata.difficulty.value)
    info_table.add_row("Custom Environment", "Yes" if metadata.is_custom else "No")
    info_table.add_row("", "")
    info_table.add_row("[bold]Spaces", "")
    info_table.add_row("  Observation", metadata.observation_space_desc)
    info_table.add_row("  Action", metadata.action_space_desc)
    info_table.add_row("", "")
    info_table.add_row("[bold]Training", "")
    info_table.add_row("  Default Algorithm", metadata.default_algorithm)
    info_table.add_row("  Recommended Timesteps", f"{metadata.recommended_timesteps:,}")
    
    if metadata.requires_mujoco or metadata.requires_robot_descriptions:
        info_table.add_row("", "")
        info_table.add_row("[bold]Requirements", "")
        if metadata.requires_mujoco:
            info_table.add_row("  MuJoCo", "Required")
        if metadata.requires_robot_descriptions:
            info_table.add_row("  robot_descriptions", "Required")
    
    if metadata.tags:
        info_table.add_row("", "")
        info_table.add_row("Tags", ", ".join(metadata.tags))
    
    console.print(info_table)
    
    # Show example training command
    console.print(f"\n[bold]Example Training Command:[/bold]")
    console.print(f"  [cyan]robot-lab train --env {metadata.env_id} --algo {metadata.default_algorithm}[/cyan]\n")


@app.command(name="run-experiment")
def run_experiment_cmd(
    config: Path = typer.Argument(
        ...,
        help="Path to YAML experiment configuration file",
        exists=True,
    ),
    experiment_id: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Specific experiment ID to run (runs all if not specified)"
    ),
    output_dir: Optional[Path] = typer.Option(
        "data/experiments",
        "--output-dir",
        "-o",
        help="Base output directory for experiment results"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show execution plan without running experiments"
    ),
    list_only: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List available experiments in config and exit"
    ),
):
    """
    Run experiments from YAML configuration file.
    
    This command orchestrates multiple training runs based on a structured YAML
    experiment specification. Useful for systematic hyperparameter sweeps,
    ablation studies, and comparative experiments.
    
    Examples:
    
        # List experiments in config
        robot-lab run-experiment experiments/0_foundations/configs/smooth_locomotion_experiments.yaml --list
        
        # Run all enabled experiments
        robot-lab run-experiment experiments/0_foundations/configs/smooth_locomotion_experiments.yaml
        
        # Run specific experiment
        robot-lab run-experiment experiments/0_foundations/configs/smooth_locomotion_experiments.yaml -e exp0_baseline
        
        # Dry run to see execution plan
        robot-lab run-experiment experiments/0_foundations/configs/smooth_locomotion_experiments.yaml --dry-run
    """
    from robot_lab.experiments.runner import ExperimentRunner
    
    # Configure logger
    configure_logger()
    
    runner = ExperimentRunner(str(config), str(output_dir))
    
    if list_only:
        # List experiments grouped by tags
        experiments = runner.list_experiments(enabled_only=False)
        
        console.print(f"\n[bold]Experiments in {config.name}:[/bold]\n")
        
        # Group experiments by tag
        by_tag = {}
        no_tag = []
        for exp_id in experiments:
            exp_config = runner.experiments[exp_id]
            tag = exp_config.get('tag')
            if tag:
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append((exp_id, exp_config))
            else:
                no_tag.append((exp_id, exp_config))
        
        # Print experiments grouped by tag
        for tag, exp_list in sorted(by_tag.items()):
            console.print(f"[bold cyan]Tag: {tag}[/bold cyan]")
            for exp_id, exp_config in exp_list:
                enabled = exp_config.get("enabled", True)
                status_icon = "✓" if enabled else "✗"
                status_color = "green" if enabled else "red"
                
                console.print(f"  [{status_color}]{status_icon}[/{status_color}] {exp_id}")
                console.print(f"     {exp_config.get('description', 'No description')}")
                if exp_config.get('notes'):
                    console.print(f"     [dim]Note: {exp_config['notes']}[/dim]")
            console.print()
        
        # Print untagged experiments
        if no_tag:
            console.print(f"[bold yellow]Untagged:[/bold yellow]")
            for exp_id, exp_config in no_tag:
                enabled = exp_config.get("enabled", True)
                status_icon = "✓" if enabled else "✗"
                status_color = "green" if enabled else "red"
                
                console.print(f"  [{status_color}]{status_icon}[/{status_color}] {exp_id}")
                console.print(f"     {exp_config.get('description', 'No description')}")
                if exp_config.get('notes'):
                    console.print(f"     [dim]Note: {exp_config['notes']}[/dim]")
            console.print()
        
        enabled_count = len(runner.list_experiments(enabled_only=True))
        console.print(f"[bold]{enabled_count}/{len(experiments)} experiments enabled[/bold]\n")
        return
    
    # Run experiments
    try:
        if experiment_id:
            if dry_run:
                console.print(f"\n[bold cyan]Dry Run - Experiment Plan:[/bold cyan]\n")
                runner._print_experiment_plan(experiment_id)
            else:
                console.print(f"\n[bold green]Running Experiment: {experiment_id}[/bold green]\n")
                runner.run_experiment(experiment_id)
        else:
            if dry_run:
                console.print(f"\n[bold cyan]Dry Run - Full Campaign Plan:[/bold cyan]\n")
            else:
                console.print(f"\n[bold green]Running Experiment Campaign[/bold green]\n")
            
            results = runner.run_all(dry_run=dry_run)
            
            if not dry_run:
                # Print summary
                console.print(f"\n[bold]Campaign Complete![/bold]")
                successes = sum(1 for success in results.values() if success)
                console.print(f"  Success: {successes}/{len(results)}")
                
                if successes < len(results):
                    console.print(f"\n[bold red]Failed Experiments:[/bold red]")
                    for exp_id, success in results.items():
                        if not success:
                            console.print(f"  ✗ {exp_id}")
    
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
