"""Shared, non-personal path configuration for the reproduction scripts."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("SUSTAIN_PROJECT_DIR", REPOSITORY_DIR)).expanduser().resolve()
HPC_DIR = (PROJECT_DIR / "HPC").resolve()
INPUT_DIR = (HPC_DIR / "input").resolve()
OUTPUT_DIR = (PROJECT_DIR / "Output_sustainrun").resolve()


def require_file(path: Path) -> Path:
    """Return an existing file or raise a concise, actionable error."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}. Configure SUSTAIN_PROJECT_DIR "
            "and use the directory structure described in README.md."
        )
    return path
