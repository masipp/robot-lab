"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for tests.
    
    Yields:
        Path to temporary directory that will be cleaned up after test
    """
    temp_dir = tempfile.mkdtemp(prefix="robot_lab_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_seed() -> int:
    """Consistent seed for reproducible tests.
    
    Returns:
        Random seed value
    """
    return 42
