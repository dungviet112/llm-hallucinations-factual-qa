"""Configuration loading and small configuration helpers."""

from pathlib import Path

import yaml


def load_config(path="config/experiment.yaml"):
    """Load experiment settings from a YAML file."""
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_config_directories(config):
    """Create configured output directories and return them as Paths."""
    paths = config.get("paths", {})
    directories = {
        key: Path(value)
        for key, value in paths.items()
        if key.endswith("_dir")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories