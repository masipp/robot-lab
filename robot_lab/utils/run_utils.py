"""Utilities for generating run identifiers and managing run-related operations."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
from loguru import logger


def generate_run_id(suffix: str = "") -> str:
    """Generate a unique run identifier with timestamp and hash.
    
    Format: {timestamp}_{hash}_{suffix}
    - Timestamp: YYYYMMDD_HHMMSS
    - Hash: 8-character hash derived from timestamp (for uniqueness)
    - Suffix: Optional descriptor (e.g., "sac_walker2d")
    
    Args:
        suffix: Optional suffix for the run ID (e.g., "sac_walker2d")
    
    Returns:
        Formatted run identifier string
    """
    # Get timestamp with microseconds for uniqueness
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # Generate 8-character hash from timestamp + microseconds
    timestamp_with_micro = now.strftime("%Y%m%d_%H%M%S_%f")
    sha = hashlib.sha256(timestamp_with_micro.encode()).hexdigest()[:8]
    
    # Build run ID
    if suffix:
        return f"{timestamp}_{sha}_{suffix}"
    else:
        return f"{timestamp}_{sha}"


def cleanup_old_runs(
    directory: Path,
    max_age_days: int = 14,
    pattern: str = "*",
    dry_run: bool = False
) -> Tuple[int, int]:
    """Remove old run directories/files older than max_age_days.
    
    Args:
        directory: Directory to clean up
        max_age_days: Maximum age in days before deletion
        pattern: Glob pattern to match files/directories
        dry_run: If True, only report what would be deleted without deleting
    
    Returns:
        Tuple of (number_deleted, total_size_freed_mb)
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return 0, 0
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    deleted_count = 0
    total_size = 0
    
    # Find all matching items
    for item in directory.glob(pattern):
        try:
            # Get modification time
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            
            if mtime < cutoff_time:
                # Calculate size
                if item.is_file():
                    size = item.stat().st_size
                elif item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                else:
                    continue
                
                total_size += size
                
                if dry_run:
                    logger.info(f"Would delete: {item} ({size / 1024 / 1024:.2f} MB, age: {(datetime.now() - mtime).days} days)")
                else:
                    # Delete the item
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                    
                    logger.info(f"Deleted: {item} ({size / 1024 / 1024:.2f} MB, age: {(datetime.now() - mtime).days} days)")
                
                deleted_count += 1
        
        except Exception as e:
            logger.warning(f"Failed to process {item}: {e}")
    
    total_size_mb = total_size / 1024 / 1024
    
    if deleted_count > 0:
        if dry_run:
            logger.info(f"Dry run: Would delete {deleted_count} items, freeing {total_size_mb:.2f} MB")
        else:
            logger.success(f"Deleted {deleted_count} old items, freed {total_size_mb:.2f} MB")
    else:
        logger.info(f"No items older than {max_age_days} days found in {directory}")
    
    return deleted_count, int(total_size_mb)
