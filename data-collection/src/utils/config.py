"""YAML configuration loader for the DF project."""

import yaml
from pathlib import Path
from typing import Optional


# Project root: two levels up from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to YAML file. If None, loads configs/default.yaml.
                     Relative paths are resolved from PROJECT_ROOT.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def get_data_dir(config: dict, subdir: str = "") -> Path:
    """Resolve a data directory path relative to PROJECT_ROOT.

    Args:
        config: Loaded configuration dictionary.
        subdir: Optional subdirectory under the collected data path.

    Returns:
        Absolute Path to the directory.
    """
    base = PROJECT_ROOT / config["data"]["collected_dir"]
    if subdir:
        return base / subdir
    return base


def ensure_dirs(config: dict) -> None:
    """Create all required data directories if they do not exist."""
    dirs = [
        PROJECT_ROOT / config["data"]["raw_dir"] / "ClosedWorld" / "NoDef",
        PROJECT_ROOT / config["data"]["raw_dir"] / "OpenWorld" / "NoDef",
        PROJECT_ROOT / config["data"]["collected_dir"] / "pcap",
        PROJECT_ROOT / config["data"]["collected_dir"] / "traces",
        PROJECT_ROOT / config["data"]["collected_dir"] / "pickle",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
