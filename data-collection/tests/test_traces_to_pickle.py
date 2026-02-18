"""Tests for scripts/traces_to_pickle.py."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.traces_to_pickle import (
    load_trace_file,
    save_splits,
    split_data,
    validate_traces,
)


class TestLoadTraceFile:
    def test_valid_file(self, tmp_path):
        """Parses valid trace file, returns correct directions."""
        trace_file = tmp_path / "0-0"
        trace_file.write_text("0.000000\t1\n0.012000\t-1\n0.045000\t1\n")

        result = load_trace_file(trace_file)
        assert result == [1.0, -1.0, 1.0]

    def test_with_comments(self, tmp_path):
        """Skips comment and blank lines."""
        trace_file = tmp_path / "0-1"
        trace_file.write_text(
            "# This is a comment\n"
            "\n"
            "0.000000\t1\n"
            "0.012000\t-1\n"
        )
        result = load_trace_file(trace_file)
        assert result == [1.0, -1.0]

    def test_malformed_lines(self, tmp_path):
        """Warns and skips bad lines, parses good ones."""
        trace_file = tmp_path / "0-2"
        trace_file.write_text(
            "0.000000\t1\n"
            "bad line\n"
            "0.012000\t-1\n"
            "0.045\tnotanumber\n"
            "0.050000\t1\n"
        )
        result = load_trace_file(trace_file)
        assert result == [1.0, -1.0, 1.0]

    def test_empty_file(self, tmp_path):
        """Returns empty list for empty file."""
        trace_file = tmp_path / "0-3"
        trace_file.write_text("")
        result = load_trace_file(trace_file)
        assert result == []


class TestValidateTraces:
    def _make_traces(self, lengths: list[int], num_labels: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Helper to create traces of specified non-zero lengths."""
        seq_len = 5000
        X = np.zeros((len(lengths), seq_len), dtype=np.float32)
        y = np.array([i % num_labels for i in range(len(lengths))], dtype=np.int64)

        rng = np.random.RandomState(42)
        for i, length in enumerate(lengths):
            directions = rng.choice([-1.0, 1.0], size=length)
            X[i, :length] = directions

        return X, y

    def test_removes_short_traces(self):
        """Traces below min_packets are removed."""
        X, y = self._make_traces([200, 10, 300, 5])  # indices 1 and 3 are short
        X_f, y_f, stats = validate_traces(X, y, min_packets=50)
        assert len(X_f) == 2
        assert stats["num_removed_short"] == 2

    def test_keeps_long_traces(self):
        """Traces at or above min_packets are kept."""
        X, y = self._make_traces([100, 200, 300])
        X_f, y_f, stats = validate_traces(X, y, min_packets=50)
        assert len(X_f) == 3
        assert stats["num_removed_short"] == 0

    def test_stats_dict(self):
        """Stats dict has correct structure and values."""
        X, y = self._make_traces([200, 10, 300])
        _, _, stats = validate_traces(X, y, min_packets=50)

        assert stats["total_input"] == 3
        assert stats["num_removed_short"] == 1
        assert stats["total_after_filter"] == 2
        assert "length_mean" in stats
        assert "length_min" in stats
        assert "length_max" in stats
        assert "num_labels" in stats
        assert "flagged_labels" in stats

    def test_empty_input(self):
        """Handles empty X/y gracefully."""
        X = np.zeros((0, 5000), dtype=np.float32)
        y = np.array([], dtype=np.int64)
        X_f, y_f, stats = validate_traces(X, y)
        assert len(X_f) == 0
        assert stats["total_input"] == 0

    def test_all_removed(self):
        """All traces below threshold returns empty arrays."""
        X, y = self._make_traces([5, 10, 3])
        X_f, y_f, stats = validate_traces(X, y, min_packets=50)
        assert len(X_f) == 0
        assert stats["total_after_filter"] == 0


class TestSplitData:
    def _make_dataset(self, num_labels=5, per_label=20):
        """Create a balanced dataset."""
        total = num_labels * per_label
        X = np.random.rand(total, 100).astype(np.float32)
        y = np.repeat(np.arange(num_labels), per_label).astype(np.int64)
        return X, y

    def test_proportions(self):
        """80/10/10 split with correct sizes."""
        X, y = self._make_dataset(num_labels=5, per_label=20)
        splits = split_data(X, y, 0.8, 0.1, seed=42)

        total = len(X)
        assert len(splits["train"][0]) + len(splits["valid"][0]) + len(splits["test"][0]) == total

        # Each label has 20 instances: 16 train, 2 valid, 2 test
        assert len(splits["train"][0]) == 80  # 5 labels * 16
        assert len(splits["valid"][0]) == 10  # 5 labels * 2
        assert len(splits["test"][0]) == 10   # 5 labels * 2

    def test_stratified(self):
        """Every label appears in all 3 splits."""
        X, y = self._make_dataset(num_labels=5, per_label=20)
        splits = split_data(X, y, 0.8, 0.1, seed=42)

        for split_name in ["train", "valid", "test"]:
            labels_in_split = set(splits[split_name][1].tolist())
            assert labels_in_split == {0, 1, 2, 3, 4}, (
                f"Split '{split_name}' missing labels: {set(range(5)) - labels_in_split}"
            )

    def test_reproducible(self):
        """Same seed produces identical splits."""
        X, y = self._make_dataset()
        splits_1 = split_data(X, y, 0.8, 0.1, seed=42)
        splits_2 = split_data(X, y, 0.8, 0.1, seed=42)

        for key in ["train", "valid", "test"]:
            np.testing.assert_array_equal(splits_1[key][0], splits_2[key][0])
            np.testing.assert_array_equal(splits_1[key][1], splits_2[key][1])

    def test_different_seeds(self):
        """Different seeds produce different order."""
        X, y = self._make_dataset()
        splits_1 = split_data(X, y, 0.8, 0.1, seed=1)
        splits_2 = split_data(X, y, 0.8, 0.1, seed=2)

        # The content sets should be the same but order should differ
        # (with high probability for different seeds)
        assert not np.array_equal(splits_1["train"][0], splits_2["train"][0])


class TestSaveSplits:
    def test_creates_files(self, tmp_path):
        """6 pickle files created with correct names."""
        X = np.random.rand(10, 100).astype(np.float32)
        y = np.arange(10, dtype=np.int64)
        splits = {
            "train": (X[:6], y[:6]),
            "valid": (X[6:8], y[6:8]),
            "test": (X[8:], y[8:]),
        }
        save_splits(splits, tmp_path, "TestSuffix")

        expected_files = [
            "X_train_TestSuffix.pkl", "y_train_TestSuffix.pkl",
            "X_valid_TestSuffix.pkl", "y_valid_TestSuffix.pkl",
            "X_test_TestSuffix.pkl", "y_test_TestSuffix.pkl",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_roundtrip(self, tmp_path):
        """Save then load produces identical arrays."""
        X = np.random.rand(10, 100).astype(np.float32)
        y = np.arange(10, dtype=np.int64)
        splits = {
            "train": (X[:6], y[:6]),
            "valid": (X[6:8], y[6:8]),
            "test": (X[8:], y[8:]),
        }
        save_splits(splits, tmp_path, "RT")

        for split_name, (X_orig, y_orig) in splits.items():
            with open(tmp_path / f"X_{split_name}_RT.pkl", "rb") as f:
                X_loaded = pickle.load(f)
            with open(tmp_path / f"y_{split_name}_RT.pkl", "rb") as f:
                y_loaded = pickle.load(f)

            np.testing.assert_array_equal(X_loaded, X_orig)
            np.testing.assert_array_equal(y_loaded, y_orig)
