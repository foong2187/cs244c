"""Tests for src/utils/reproducibility.py."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.reproducibility import set_seed


class TestSetSeed:
    def test_deterministic(self):
        """Same seed produces same random sequence."""
        set_seed(123)
        py_vals_1 = [random.random() for _ in range(5)]
        np_vals_1 = np.random.rand(5).tolist()

        set_seed(123)
        py_vals_2 = [random.random() for _ in range(5)]
        np_vals_2 = np.random.rand(5).tolist()

        assert py_vals_1 == py_vals_2
        assert np_vals_1 == np_vals_2

    def test_different_seeds(self):
        """Different seeds produce different sequences."""
        set_seed(1)
        vals_1 = np.random.rand(5).tolist()

        set_seed(2)
        vals_2 = np.random.rand(5).tolist()

        assert vals_1 != vals_2
