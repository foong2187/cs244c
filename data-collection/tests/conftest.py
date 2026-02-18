"""Shared pytest fixtures for the data collection test suite."""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_config(tmp_path):
    """Returns a config dict with tmp_path-based directories."""
    return {
        "data": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "collected_dir": str(tmp_path / "data" / "collected"),
            "site_list": str(tmp_path / "data" / "collected" / "site_list.txt"),
            "sequence_length": 5000,
        },
        "collection": {
            "num_batches": 90,
            "page_load_timeout": 60,
            "post_load_wait": 5,
            "newnym_wait": 10,
            "max_retries": 3,
            "min_trace_packets": 50,
            "tor_browser_path": "/opt/tor-browser",
        },
        "preprocessing": {
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "seed": 42,
    }


@pytest.fixture
def config_file(tmp_path, sample_config):
    """Writes a valid YAML config to tmp_path and returns the path."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_traces(tmp_path):
    """Creates synthetic trace text files (3 labels x 10 instances).

    Returns the directory containing the trace files.
    """
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    rng = np.random.RandomState(42)
    for label in range(3):
        for batch in range(10):
            trace_file = traces_dir / f"{label}-{batch}"
            num_packets = rng.randint(100, 500)
            with open(trace_file, "w") as f:
                t = 0.0
                for _ in range(num_packets):
                    direction = rng.choice([1, -1])
                    f.write(f"{t:.6f}\t{direction}\n")
                    t += rng.exponential(0.02)

    return traces_dir


@pytest.fixture
def sample_pickle_dir(tmp_path):
    """Creates 6 mock pickle files for closed-world testing.

    Returns the directory containing the pickle files.
    """
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()

    rng = np.random.RandomState(42)
    num_classes = 5

    for split, n_samples in [("train", 40), ("valid", 5), ("test", 5)]:
        X = rng.choice([-1.0, 1.0], size=(n_samples, 5000)).astype(np.float32)
        y = np.repeat(np.arange(num_classes), n_samples // num_classes).astype(np.int64)

        with open(pickle_dir / f"X_{split}_NoDef.pkl", "wb") as f:
            pickle.dump(X, f)
        with open(pickle_dir / f"y_{split}_NoDef.pkl", "wb") as f:
            pickle.dump(y, f)

    return pickle_dir


@pytest.fixture
def sample_site_list(tmp_path):
    """Writes a site_list.txt with 3 test sites, returns the file path."""
    site_list = tmp_path / "site_list.txt"
    site_list.write_text(
        "# Test sites\n"
        "0\thttps://www.example.com\n"
        "1\thttps://www.google.com\n"
        "2\thttps://www.wikipedia.org\n"
    )
    return site_list


@pytest.fixture
def sample_progress_csv(tmp_path):
    """Writes a progress.csv with mix of success/failed rows."""
    progress_file = tmp_path / "progress.csv"
    with open(progress_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "batch", "timestamp", "status",
            "duration_sec", "pcap_size_bytes", "error_msg",
        ])
        writer.writerow(["0", "0", "2026-02-17T10:00:00", "success", "45.2", "12345", ""])
        writer.writerow(["1", "0", "2026-02-17T10:01:00", "failed", "60.0", "0", "TimeoutException"])
        writer.writerow(["2", "0", "2026-02-17T10:02:00", "success", "38.7", "9876", ""])
        writer.writerow(["0", "1", "2026-02-17T10:03:00", "success", "42.1", "11111", ""])
    return progress_file
