"""
Data loading and preprocessing utilities for Deep Fingerprinting.

Supports loading the original pickle-format datasets from the DF paper,
as well as generating synthetic data for architecture testing.

Dataset format (from the paper):
  - X: Array of shape (n, 5000) containing packet direction sequences
       where +1 = outgoing, -1 = incoming, 0 = padding
  - y: Array of shape (n,) containing website class labels (integers)

Data split: 80% train / 10% validation / 10% test
"""

import os
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_LENGTH = 5000
BASE_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')


# ---------------------------------------------------------------------------
# Generic pickle loader
# ---------------------------------------------------------------------------
def _load_pickle(filepath):
    """Load a pickle file, supporting both Python 2 and 3 formats."""
    with open(filepath, 'rb') as f:
        try:
            return np.array(pickle.load(f, encoding='latin1'))
        except TypeError:
            return np.array(pickle.load(f))


# ---------------------------------------------------------------------------
# Closed-World Data Loading
# ---------------------------------------------------------------------------
def load_closed_world_data(defense='NoDef'):
    """Load closed-world dataset for a given defense type.

    Args:
        defense: One of 'NoDef', 'WTFPAD', or 'WalkieTalkie'.

    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
        where X arrays have shape (n, 5000) and y arrays have shape (n,).
    """
    dataset_dir = os.path.join(BASE_DATASET_DIR, 'ClosedWorld', defense)

    suffix = defense
    if defense == 'NoDef':
        suffix = 'NoDef'

    print(f"Loading {defense} dataset for closed-world scenario")
    print(f"Dataset directory: {dataset_dir}")

    X_train = _load_pickle(os.path.join(dataset_dir, f'X_train_{suffix}.pkl'))
    y_train = _load_pickle(os.path.join(dataset_dir, f'y_train_{suffix}.pkl'))
    X_valid = _load_pickle(os.path.join(dataset_dir, f'X_valid_{suffix}.pkl'))
    y_valid = _load_pickle(os.path.join(dataset_dir, f'y_valid_{suffix}.pkl'))
    X_test = _load_pickle(os.path.join(dataset_dir, f'X_test_{suffix}.pkl'))
    y_test = _load_pickle(os.path.join(dataset_dir, f'y_test_{suffix}.pkl'))

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
    print(f"X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# ---------------------------------------------------------------------------
# Open-World Data Loading
# ---------------------------------------------------------------------------
def load_open_world_training_data(defense='NoDef'):
    """Load open-world training dataset (monitored + unmonitored for training).

    Args:
        defense: One of 'NoDef', 'WTFPAD', or 'WalkieTalkie'.

    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid).
    """
    dataset_dir = os.path.join(BASE_DATASET_DIR, 'OpenWorld', defense)
    suffix = defense

    print(f"Loading {defense} dataset for open-world training")

    X_train = _load_pickle(os.path.join(dataset_dir, f'X_train_{suffix}.pkl'))
    y_train = _load_pickle(os.path.join(dataset_dir, f'y_train_{suffix}.pkl'))
    X_valid = _load_pickle(os.path.join(dataset_dir, f'X_valid_{suffix}.pkl'))
    y_valid = _load_pickle(os.path.join(dataset_dir, f'y_valid_{suffix}.pkl'))

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")

    return X_train, y_train, X_valid, y_valid


def load_open_world_evaluation_data(defense='NoDef'):
    """Load open-world evaluation dataset (separate monitored/unmonitored test sets).

    Args:
        defense: One of 'NoDef', 'WTFPAD', or 'WalkieTalkie'.

    Returns:
        Tuple of (X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon).
    """
    dataset_dir = os.path.join(BASE_DATASET_DIR, 'OpenWorld', defense)
    suffix = defense

    print(f"Loading {defense} dataset for open-world evaluation")

    X_test_Mon = _load_pickle(
        os.path.join(dataset_dir, f'X_test_Mon_{suffix}.pkl'))
    y_test_Mon = _load_pickle(
        os.path.join(dataset_dir, f'y_test_Mon_{suffix}.pkl'))
    X_test_Unmon = _load_pickle(
        os.path.join(dataset_dir, f'X_test_Unmon_{suffix}.pkl'))
    y_test_Unmon = _load_pickle(
        os.path.join(dataset_dir, f'y_test_Unmon_{suffix}.pkl'))

    print(f"X_test_Mon   shape: {X_test_Mon.shape}")
    print(f"X_test_Unmon shape: {X_test_Unmon.shape}")

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_data(X_train, y_train, X_valid, y_valid, X_test, y_test,
                    num_classes):
    """Preprocess data for the DF model.

    Converts to float32, reshapes to (n, 5000, 1) for Conv1D input,
    and one-hot encodes labels.

    Args:
        X_train, y_train, X_valid, y_valid, X_test, y_test: Raw data arrays.
        num_classes: Number of website classes.

    Returns:
        Preprocessed tuple (X_train, y_train, X_valid, y_valid, X_test, y_test).
    """
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')

    # Reshape to (n, LENGTH, 1) for Conv1D
    X_train = X_train[:, :, np.newaxis]
    X_valid = X_valid[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes)
    y_valid = to_categorical(y_valid, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"{X_train.shape[0]} train samples")
    print(f"{X_valid.shape[0]} validation samples")
    print(f"{X_test.shape[0]} test samples")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def preprocess_open_world_data(X_train, y_train, X_valid, y_valid,
                               num_classes):
    """Preprocess open-world training data.

    Args:
        X_train, y_train, X_valid, y_valid: Raw data arrays.
        num_classes: Total number of classes (monitored sites + 1 unmonitored).

    Returns:
        Preprocessed tuple (X_train, y_train, X_valid, y_valid).
    """
    X_train = X_train.astype('float32')[:, :, np.newaxis]
    X_valid = X_valid.astype('float32')[:, :, np.newaxis]

    y_train = to_categorical(y_train, num_classes)
    y_valid = to_categorical(y_valid, num_classes)

    return X_train, y_train, X_valid, y_valid


def preprocess_evaluation_data(*arrays):
    """Cast arrays to float32 and reshape for Conv1D input."""
    result = []
    for arr in arrays:
        arr = arr.astype('float32')
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        result.append(arr)
    return tuple(result)


# ---------------------------------------------------------------------------
# Synthetic Data Generation (for testing without real datasets)
# ---------------------------------------------------------------------------
def generate_synthetic_data(num_classes=95, samples_per_class=100,
                            length=INPUT_LENGTH, split_ratio=(0.8, 0.1, 0.1)):
    """Generate synthetic website fingerprinting data for testing.

    Creates random packet direction sequences with class-specific patterns
    to allow meaningful (though not realistic) training and validation.

    Args:
        num_classes: Number of website classes.
        samples_per_class: Number of traces per class.
        length: Length of each trace.
        split_ratio: (train, valid, test) split ratios.

    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test).
    """
    rng = np.random.RandomState(42)
    total = num_classes * samples_per_class

    X = np.zeros((total, length), dtype='float32')
    y = np.zeros(total, dtype='int32')

    for c in range(num_classes):
        start = c * samples_per_class
        end = start + samples_per_class

        # Each class gets a characteristic burst pattern
        pattern_length = rng.randint(100, 500)
        pattern_offset = rng.randint(0, length - pattern_length)
        base_pattern = rng.choice([-1, 1], size=pattern_length).astype('float32')

        for i in range(start, end):
            trace_len = rng.randint(length // 2, length)
            directions = rng.choice([-1, 1], size=trace_len).astype('float32')

            X[i, :trace_len] = directions
            # Inject class-specific pattern with some noise
            noise = rng.normal(0, 0.1, size=pattern_length)
            X[i, pattern_offset:pattern_offset + pattern_length] = (
                base_pattern + noise
            ).clip(-1, 1).round()
            y[i] = c

    # Shuffle
    indices = rng.permutation(total)
    X = X[indices]
    y = y[indices]

    # Split
    n_train = int(total * split_ratio[0])
    n_valid = int(total * split_ratio[1])

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:n_train + n_valid]
    y_valid = y[n_train:n_train + n_valid]
    X_test = X[n_train + n_valid:]
    y_test = y[n_train + n_valid:]

    print(f"Generated synthetic data: {num_classes} classes, "
          f"{samples_per_class} samples/class")
    print(f"  Train: {X_train.shape}, Valid: {X_valid.shape}, "
          f"Test: {X_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def generate_synthetic_open_world_data(num_monitored=95,
                                       mon_samples_per_class=100,
                                       num_unmonitored_train=5000,
                                       num_unmonitored_test=20000,
                                       mon_test_per_class=10,
                                       length=INPUT_LENGTH):
    """Generate synthetic open-world data for testing.

    Args:
        num_monitored: Number of monitored site classes.
        mon_samples_per_class: Training samples per monitored class.
        num_unmonitored_train: Number of unmonitored training traces.
        num_unmonitored_test: Number of unmonitored test traces.
        mon_test_per_class: Test samples per monitored class.
        length: Trace length.

    Returns:
        Dictionary with keys: X_train, y_train, X_valid, y_valid,
        X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon.
    """
    rng = np.random.RandomState(42)

    # Monitored training data: classes 0..num_monitored-1
    mon_total = num_monitored * mon_samples_per_class
    X_mon = rng.choice([-1, 0, 1], size=(mon_total, length)).astype('float32')
    y_mon = np.repeat(np.arange(num_monitored), mon_samples_per_class)

    # Unmonitored training data: class = num_monitored (single class)
    X_unmon_train = rng.choice([-1, 0, 1],
                               size=(num_unmonitored_train, length)).astype('float32')
    y_unmon_train = np.full(num_unmonitored_train, num_monitored, dtype='int32')

    # Combine and shuffle for training
    X_all = np.concatenate([X_mon, X_unmon_train])
    y_all = np.concatenate([y_mon, y_unmon_train])
    idx = rng.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    n_train = int(len(X_all) * 0.9)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_valid, y_valid = X_all[n_train:], y_all[n_train:]

    # Test data
    mon_test_total = num_monitored * mon_test_per_class
    X_test_Mon = rng.choice([-1, 0, 1],
                            size=(mon_test_total, length)).astype('float32')
    y_test_Mon = np.repeat(np.arange(num_monitored), mon_test_per_class)

    X_test_Unmon = rng.choice([-1, 0, 1],
                              size=(num_unmonitored_test, length)).astype('float32')
    y_test_Unmon = np.full(num_unmonitored_test, num_monitored, dtype='int32')

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_valid': X_valid, 'y_valid': y_valid,
        'X_test_Mon': X_test_Mon, 'y_test_Mon': y_test_Mon,
        'X_test_Unmon': X_test_Unmon, 'y_test_Unmon': y_test_Unmon,
    }
