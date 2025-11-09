"""Configuration management utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        if filepath.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif filepath.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")

    return config


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration dictionary
        filepath: Path to save config file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        if filepath.suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False)
        elif filepath.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "data_collection": {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "delay": 1.0,
        },
        "nlp_processing": {
            "model_name": "yiyanghkust/finbert-tone",
            "max_length": 512,
            "batch_size": 8,
            "pooling": "cls",
        },
        "dimensionality_reduction": {
            "method": "pca",
            "variance_threshold": 0.95,
            "standardize": True,
        },
        "clustering": {
            "method": "gmm",
            "n_components": 3,
            "covariance_type": "full",
        },
        "classification": {
            "model_type": "xgboost",
            "test_size": 0.2,
            "random_state": 42,
        },
        "output": {
            "save_models": True,
            "output_dir": "output",
            "save_plots": True,
        },
    }
