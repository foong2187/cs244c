"""Reproducibility utilities for seeding random number generators."""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Seeds random and numpy. Torch seeding is handled separately
    in training code since it is not needed for data collection.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
