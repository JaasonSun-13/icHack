# src/utils.py
"""Logging, config, and saving utilities."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import joblib
import numpy as np
import yaml


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup and return root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("svf")


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_model(model: Any, metadata: Dict, path: str):
    """Save model with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_model(path: str) -> Dict:
    """Load model with metadata."""
    return joblib.load(path)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Dict, path: str):
    """Save dict as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def get_timestamp() -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
