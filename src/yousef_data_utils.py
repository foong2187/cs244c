"""
Data loading for Yousef's preprocessed DF-style dataset.

Yousef's pickles live under data/yousef-data/pickle/ with either:
  - X_train_Yousef.pkl, y_train_Yousef.pkl, ... (Yousef suffix), or
  - X_train_Fresh2026.pkl, y_train_Fresh2026.pkl, ... (Fresh2026 suffix)

Format matches the DF model:
  - X: (n, 5000) direction sequences (+1, -1, 0), float32
  - y: (n,) integer labels in [0, num_classes-1]
"""

import os
import pickle
import numpy as np


def _load_pickle(filepath):
    with open(filepath, "rb") as f:
        try:
            return np.array(pickle.load(f, encoding="latin1"))
        except TypeError:
            return np.array(pickle.load(f))


def load_yousef_closed_world(data_dir=None):
    """Load Yousef closed-world train/valid/test from pickle files.

    Args:
        data_dir: Root directory containing the pickle files. If None, uses
                  ../data/yousef-data/pickle relative to this file.

    Returns:
        Tuple (X_train, y_train, X_valid, y_valid, X_test, y_test).
        X arrays shape (n, 5000), y arrays shape (n,). Dtypes are preserved
        (float32/int64) for compatibility with preprocess_data.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "yousef-data", "pickle"
        )
    data_dir = os.path.abspath(data_dir)

    # Try Yousef suffix first, then Fresh2026 (actual files use Fresh2026)
    for suffix in ("Yousef", "Fresh2026"):
        train_x = os.path.join(data_dir, f"X_train_{suffix}.pkl")
        if os.path.isfile(train_x):
            break
    else:
        raise FileNotFoundError(
            f"No Yousef/Fresh2026 train pickles found in {data_dir}. "
            "Expected X_train_Yousef.pkl or X_train_Fresh2026.pkl, etc."
        )

    def _lp(name):
        path = os.path.join(data_dir, name)
        return _load_pickle(path)

    X_train = _lp(f"X_train_{suffix}.pkl")
    y_train = _lp(f"y_train_{suffix}.pkl")
    X_valid = _lp(f"X_valid_{suffix}.pkl")
    y_valid = _lp(f"y_valid_{suffix}.pkl")
    X_test = _lp(f"X_test_{suffix}.pkl")
    y_test = _lp(f"y_test_{suffix}.pkl")

    # Ensure sequence length 5000 (truncate or pad)
    target_len = 5000
    if X_train.shape[1] != target_len:
        if X_train.shape[1] > target_len:
            X_train = X_train[:, :target_len]
            X_valid = X_valid[:, :target_len]
            X_test = X_test[:, :target_len]
        else:
            pad_width = ((0, 0), (0, target_len - X_train.shape[1]))
            X_train = np.pad(X_train, pad_width, constant_values=0)
            X_valid = np.pad(X_valid, pad_width, constant_values=0)
            X_test = np.pad(X_test, pad_width, constant_values=0)

    # Cast to float32 for consistency with data_utils
    X_train = X_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_valid = np.asarray(y_valid, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    print("Loaded Yousef closed-world splits:")
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_valid {X_valid.shape}, y_valid {y_valid.shape}")
    print(f"  X_test  {X_test.shape}, y_test  {y_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_yousef_open_world(data_dir=None, num_monitored=70):
    """Load Yousef data as open-world splits: monitored vs unmonitored.

    Uses the same pickle files as closed-world. Classes 0..num_monitored-1
    are treated as monitored; classes num_monitored..max_label are pooled
    into a single unmonitored class (label = num_monitored).

    Args:
        data_dir: Root directory containing pickle files (default: ../data/yousef-data/pickle).
        num_monitored: Number of monitored site classes (default: 70).

    Returns:
        Tuple (X_train, y_train, X_valid, y_valid,
               X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon).
        Train/valid labels are in [0, num_monitored] (unmonitored = num_monitored).
        Test Mon: only samples from monitored classes, y in [0, num_monitored-1].
        Test Unmon: samples from pooled unmonitored classes, y all num_monitored.
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_yousef_closed_world(
        data_dir=data_dir
    )
    # Avoid re-printing from load_yousef_closed_world by not calling it again for same data
    # Remap labels: monitored 0..num_monitored-1 unchanged; rest -> num_monitored
    def remap(y):
        y = np.asarray(y, dtype=np.int64)
        out = np.where(y < num_monitored, y, num_monitored)
        return out

    y_train = remap(y_train)
    y_valid = remap(y_valid)

    # Test split: monitored vs unmonitored
    mon_mask = y_test < num_monitored
    unmon_mask = ~mon_mask
    X_test_Mon = X_test[mon_mask]
    y_test_Mon = np.asarray(y_test[mon_mask], dtype=np.int64)
    X_test_Unmon = X_test[unmon_mask]
    y_test_Unmon = np.full(X_test_Unmon.shape[0], num_monitored, dtype=np.int64)

    print("Yousef open-world splits:")
    print(f"  Train {X_train.shape[0]}, Valid {X_valid.shape[0]}")
    print(f"  Test Mon {X_test_Mon.shape[0]}, Test Unmon {X_test_Unmon.shape[0]}")
    print(f"  num_monitored={num_monitored}, num_classes={num_monitored + 1}")

    return (
        X_train, y_train, X_valid, y_valid,
        X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon,
    )
