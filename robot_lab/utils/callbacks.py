"""Custom callbacks for training monitoring and checkpointing."""

import time
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger


class TimeBasedCheckpointCallback(BaseCallback):
    """
    Callback that saves model checkpoints based on wall-clock time intervals.
    
    Unlike CheckpointCallback which saves based on timesteps, this saves based on
    actual elapsed time, making it useful for long training runs where you want
    regular backups regardless of training speed.
    
    Args:
        save_freq_seconds: Save frequency in seconds (e.g., 600 for 10 minutes)
        save_path: Path to save checkpoints
        name_prefix: Prefix for checkpoint files
        verbose: Whether to log checkpoint saves
    """
    
    def __init__(
        self,
        save_freq_seconds: int,
        save_path: str,
        name_prefix: str = "model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq_seconds = save_freq_seconds
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.last_save_time = None
        self.checkpoint_count = 0
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _init_callback(self) -> None:
        """Initialize callback at the start of training."""
        # Record initial time
        self.last_save_time = time.time()
        self.start_time = time.time()  # Store start time for elapsed time calculation
        if self.verbose > 0:
            logger.info(
                f"Time-based checkpoints enabled: saving every "
                f"{self.save_freq_seconds}s ({self.save_freq_seconds / 60:.1f} minutes)"
            )
    
    def _on_step(self) -> bool:
        """
        Check if enough time has elapsed to save a checkpoint.
        This is called after each training step.
        """
        current_time = time.time()
        elapsed_since_last_save = current_time - self.last_save_time
        
        # Check if it's time to save
        if elapsed_since_last_save >= self.save_freq_seconds:
            self._save_checkpoint()
            self.last_save_time = current_time
        
        return True  # Continue training
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint with timestamp and step count."""
        self.checkpoint_count += 1
        
        # Create filename with checkpoint number and timestep
        checkpoint_name = (
            f"{self.name_prefix}_checkpoint_{self.checkpoint_count}_"
            f"steps_{self.num_timesteps}"
        )
        checkpoint_path = self.save_path / f"{checkpoint_name}.zip"
        
        # Save the model
        self.model.save(checkpoint_path)
        
        if self.verbose > 0:
            elapsed_minutes = (time.time() - self.start_time) / 60
            logger.success(
                f"✓ Checkpoint {self.checkpoint_count} saved at {self.num_timesteps:,} timesteps "
                f"({elapsed_minutes:.1f} minutes elapsed) → {checkpoint_path.name}"
            )
    
    def _on_training_start(self) -> None:
        """Called when training starts (not needed anymore, using _init_callback)."""
        pass


class VecNormalizeSaveCallback(BaseCallback):
    """
    Callback that saves VecNormalize statistics alongside model checkpoints.
    
    This ensures that the normalization statistics are saved whenever the model
    is saved, which is critical for proper evaluation and deployment.
    
    Args:
        save_path: Path to save VecNormalize statistics
        name_prefix: Prefix for VecNormalize files
        verbose: Whether to log saves
    """
    
    def __init__(
        self,
        save_path: str,
        name_prefix: str = "vecnorm",
        save_freq_seconds: Optional[int] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_freq_seconds = save_freq_seconds
        self.last_save_time = None
        self.checkpoint_count = 0
        
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _init_callback(self) -> None:
        """Initialize callback."""
        if self.save_freq_seconds:
            self.last_save_time = time.time()
    
    def _on_step(self) -> bool:
        """Check if VecNormalize should be saved."""
        if self.save_freq_seconds:
            current_time = time.time()
            elapsed_since_last_save = current_time - self.last_save_time
            
            if elapsed_since_last_save >= self.save_freq_seconds:
                self._save_vecnormalize()
                self.last_save_time = current_time
        
        return True
    
    def _save_vecnormalize(self) -> None:
        """Save VecNormalize statistics."""
        # Check if environment has VecNormalize wrapper
        if hasattr(self.training_env, 'save'):
            self.checkpoint_count += 1
            vecnorm_name = (
                f"{self.name_prefix}_checkpoint_{self.checkpoint_count}_"
                f"steps_{self.num_timesteps}.pkl"
            )
            vecnorm_path = self.save_path / vecnorm_name
            
            self.training_env.save(str(vecnorm_path))
            
            if self.verbose > 0:
                logger.debug(f"VecNormalize stats saved → {vecnorm_path.name}")
