# src/config_loader.py

from pathlib import Path
from typing import Optional

import yaml


def load_config(root: Path, profile: Optional[str] = None) -> dict:
    """
    Load a YAML config dynamically.

    - If profile is None: determine from market outlook and select stocks dynamically
    - If profile is specified: loads config_{profile}.yaml with dynamic stock selection
    """
    from src.dynamic_config import load_dynamic_config
    return load_dynamic_config(root, profile)
