"""
Dynamic Configuration Loader
============================

Loads configuration dynamically based on market outlook and stock selection.
"""

from pathlib import Path
from typing import Optional, Dict
import yaml
from src.market_outlook import get_dynamic_profile
from src.stock_selector import get_selected_tickers


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins)."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_dynamic_config(root: Path, base_profile: Optional[str] = None) -> Dict:
    """
    Load configuration dynamically.

    - If base_profile is provided, use it; otherwise determine from market outlook
    - Select stocks dynamically
    - Return modified config
    """
    # Determine profile
    if base_profile is None:
        profile = get_dynamic_profile(root)
    else:
        profile = base_profile

    # Always load base config, then optionally overlay profile-specific config.
    base_path = root / "configs" / "config.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    if profile is None:
        cfg = base_cfg
    else:
        # Preferred path: single-file profile overrides in config.yaml.
        dynamic_profiles = base_cfg.get("dynamic_profiles", {}) or {}
        if profile in dynamic_profiles:
            cfg = _deep_merge(base_cfg, dynamic_profiles[profile] or {})
        else:
            # Backward compatibility: config_{profile}.yaml files.
            profile_path = root / "configs" / f"config_{profile}.yaml"
            if not profile_path.exists():
                raise FileNotFoundError(f"Config file not found: {profile_path}")
            with profile_path.open("r", encoding="utf-8") as f:
                profile_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(base_cfg, profile_cfg)

    # Update tickers with selected stocks
    selected_tickers = get_selected_tickers(root)
    cfg["market"]["tickers"] = selected_tickers

    # Optionally set the ticker to the first one
    if selected_tickers:
        cfg["market"]["ticker"] = selected_tickers[0]

    cfg["active_profile"] = profile

    return cfg


def load_config(root: Path, profile: Optional[str] = None) -> Dict:
    """
    Backwards compatible load_config that uses dynamic loading if profile is None.
    """
    return load_dynamic_config(root, profile)