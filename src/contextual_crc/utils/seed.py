from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed available random number generators without requiring GPU libraries."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
