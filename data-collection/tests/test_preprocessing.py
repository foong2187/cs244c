"""Tests for src/data/preprocessing.py."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    load_closed_world_data,
    load_pickle,
    pad_or_truncate,
)


class TestLoadPickle:
    def test_valid_pickle(self, tmp_path):
        """Loads a Python 3 pickle correctly."""
        data = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        result = load_pickle(pkl_path)
        np.testing.assert_array_equal(result, data)

    def test_file_not_found(self):
        """Raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_pickle("/nonexistent/file.pkl")

    def test_2d_array(self, tmp_path):
        """Loads a 2D array pickle correctly."""
        data = np.random.rand(10, 5000).astype(np.float32)
        pkl_path = tmp_path / "test2d.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        result = load_pickle(pkl_path)
        np.testing.assert_array_equal(result, data)
        assert result.shape == (10, 5000)


class TestPadOrTruncate:
    def test_short_trace(self):
        """100-element array padded to 5000 with zeros."""
        trace = np.ones(100, dtype=np.float32)
        result = pad_or_truncate(trace, 5000)
        assert result.shape == (5000,)
        assert np.all(result[:100] == 1.0)
        assert np.all(result[100:] == 0.0)

    def test_exact_length(self):
        """5000-element array returned unchanged."""
        trace = np.ones(5000, dtype=np.float32)
        result = pad_or_truncate(trace, 5000)
        assert result.shape == (5000,)
        np.testing.assert_array_equal(result, trace)

    def test_long_trace(self):
        """7000-element array truncated to 5000."""
        trace = np.arange(7000, dtype=np.float32)
        result = pad_or_truncate(trace, 5000)
        assert result.shape == (5000,)
        np.testing.assert_array_equal(result, trace[:5000])

    def test_empty_trace(self):
        """Empty array produces all zeros."""
        trace = np.array([], dtype=np.float32)
        result = pad_or_truncate(trace, 5000)
        assert result.shape == (5000,)
        assert np.all(result == 0.0)

    def test_dtype_always_float32(self):
        """Output is always float32 regardless of input dtype."""
        trace_int = np.array([1, -1, 1], dtype=np.int64)
        result = pad_or_truncate(trace_int, 10)
        assert result.dtype == np.float32

    def test_custom_length(self):
        """Non-default length works."""
        trace = np.ones(5, dtype=np.float32)
        result = pad_or_truncate(trace, 10)
        assert result.shape == (10,)
        assert np.all(result[:5] == 1.0)
        assert np.all(result[5:] == 0.0)


class TestLoadClosedWorldData:
    def test_loads_all_splits(self, sample_pickle_dir):
        """Loads 6 pickle files and returns correct shapes."""
        result = load_closed_world_data(sample_pickle_dir)
        assert len(result) == 6
        X_train, y_train, X_valid, y_valid, X_test, y_test = result

        assert X_train.shape == (40, 5000)
        assert y_train.shape == (40,)
        assert X_valid.shape == (5, 5000)
        assert y_valid.shape == (5,)
        assert X_test.shape == (5, 5000)
        assert y_test.shape == (5,)

    def test_missing_file(self, tmp_path):
        """Raises FileNotFoundError when a file is missing."""
        with pytest.raises(FileNotFoundError):
            load_closed_world_data(tmp_path)
