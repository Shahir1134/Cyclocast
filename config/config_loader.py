"""
config/config_loader.py
=========================
Utilities for loading and validating the YAML configuration file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and return the YAML configuration as a Python dict.

    Parameters
    ----------
    config_path : str | Path
        Path to ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Config loaded from '%s'.", config_path)
    return cfg


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to a ``torch.device``.

    Parameters
    ----------
    device_str : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.
        ``"auto"`` picks CUDA > MPS > CPU in that priority order.

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info("Using device: %s", device)
    return device
