"""Logger configuration and utilities for robot_lab using loguru."""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def configure_logger(
    output_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    rotation: str = "10 MB",
    retention: str = "14 days",
) -> None:
    """Configure loguru logger for robot_lab.
    
    Sets up console and file logging with appropriate formatting.
    Logs are saved to the output directory if specified.
    
    Args:
        output_dir: Optional output directory for log files
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to save logs to file
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep old log files (default: 14 days)
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger with colors and formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    
    # Add file logger if requested
    if log_to_file and output_dir:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "robot_lab_{time:YYYY-MM-DD}.log"
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
        logger.debug(f"File logging enabled: {log_file}")


def get_logger():
    """Get the configured logger instance.
    
    Returns:
        The loguru logger instance
    """
    return logger
