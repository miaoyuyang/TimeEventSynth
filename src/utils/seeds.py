"""Central seed management for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional in some environments
    torch = None


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, common env seeds, and PyTorch when available."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def detector_random_state(config: dict) -> int:
    """Resolve detector random state from normalized config."""
    split_seed = config.get("split", {}).get("seed")
    if split_seed is not None:
        return int(split_seed)
    return int(config.get("seed", 42))
