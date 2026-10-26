"""Shared settings for deterministic LV processing."""

from dataclasses import dataclass, field
import random

import numpy as np


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for inference and beat analysis."""

    seed: int = 42
    threshold: float = 0.5
    denoise: bool = False
    batch_size: int = 16
    model_widths: tuple[int, ...] = (64, 128, 256, 512)
    disk_count: int = 20
    savgol_seconds: float = 0.1
    savgol_order: int = 2
    min_beat_seconds: float = 0.4


def set_seed(seed: int = 42) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
