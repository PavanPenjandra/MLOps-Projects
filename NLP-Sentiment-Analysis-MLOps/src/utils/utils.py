"""
Utility functions for the project.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
import yaml


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """
    Configure logging.

    Args:
        log_level: Logging level
        log_file: Path to log file
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    logger.info(f"Config saved to {output_path}")


def get_env_var(var_name: str, default: Any = None) -> Any:
    """
    Get environment variable with optional default.

    Args:
        var_name: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    return os.getenv(var_name, default)


def create_directories(paths: list) -> None:
    """
    Create multiple directories.

    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict:
    """
    Flatten nested dictionary.

    Args:
        d: Nested dictionary
        parent_key: Parent key prefix
        sep: Separator

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
