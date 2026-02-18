"""Data preprocessing utilities for loading and transforming DF traces."""

import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_pickle(path: str | Path) -> np.ndarray:
    """Load a pickle file with Python 2 compatibility fallback.

    The original DF dataset was serialized with Python 2's cPickle.
    Python 3 cannot read these directly; encoding='latin1' is needed.

    Args:
        path: Path to the .pkl file.

    Returns:
        Numpy array loaded from the pickle.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")

    return np.array(data)


def pad_or_truncate(trace: np.ndarray, length: int = 5000) -> np.ndarray:
    """Pad with 0.0 or truncate a trace to a fixed length.

    Args:
        trace: 1D numpy array of direction values (+1.0, -1.0).
        length: Target length (default 5000, per DF paper).

    Returns:
        1D numpy array of exactly `length` elements, dtype float32.
    """
    trace = np.asarray(trace, dtype=np.float32)
    if len(trace) >= length:
        return trace[:length]
    padded = np.zeros(length, dtype=np.float32)
    padded[: len(trace)] = trace
    return padded


def load_closed_world_data(
    data_dir: str | Path, defense: str = "NoDef"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all closed-world dataset splits.

    Expects files named X_{split}_{defense}.pkl and y_{split}_{defense}.pkl
    for splits: train, valid, test.

    Args:
        data_dir: Directory containing the pickle files.
        defense: Defense name string (e.g., "NoDef").

    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test).
    """
    data_dir = Path(data_dir)
    results = []
    for split in ["train", "valid", "test"]:
        X = load_pickle(data_dir / f"X_{split}_{defense}.pkl")
        y = load_pickle(data_dir / f"y_{split}_{defense}.pkl")
        logger.info(f"Loaded {split}: X={X.shape}, y={y.shape}")
        results.extend([X, y])
    return tuple(results)


def load_open_world_data(
    data_dir: str | Path, defense: str = "NoDef"
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Load open-world dataset splits.

    Uses the same train/valid files as closed-world, plus separate
    monitored and unmonitored test files.

    Args:
        data_dir: Directory containing the pickle files.
        defense: Defense name string.

    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid,
                  X_test_mon, y_test_mon, X_test_unmon, y_test_unmon).
    """
    data_dir = Path(data_dir)

    X_train = load_pickle(data_dir / f"X_train_{defense}.pkl")
    y_train = load_pickle(data_dir / f"y_train_{defense}.pkl")
    X_valid = load_pickle(data_dir / f"X_valid_{defense}.pkl")
    y_valid = load_pickle(data_dir / f"y_valid_{defense}.pkl")
    X_test_mon = load_pickle(data_dir / f"X_test_Mon_{defense}.pkl")
    y_test_mon = load_pickle(data_dir / f"y_test_Mon_{defense}.pkl")
    X_test_unmon = load_pickle(data_dir / f"X_test_Unmon_{defense}.pkl")
    y_test_unmon = load_pickle(data_dir / f"y_test_Unmon_{defense}.pkl")

    logger.info(
        f"Loaded open-world: train={X_train.shape}, valid={X_valid.shape}, "
        f"test_mon={X_test_mon.shape}, test_unmon={X_test_unmon.shape}"
    )

    return (
        X_train, y_train, X_valid, y_valid,
        X_test_mon, y_test_mon, X_test_unmon, y_test_unmon,
    )
