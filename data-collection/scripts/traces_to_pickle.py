#!/usr/bin/env python3
"""Convert trace files to DF-compatible pickle format.

Usage:
    python scripts/traces_to_pickle.py [--config configs/default.yaml]
                                       [--traces-dir data/collected/traces]
                                       [--output-dir data/collected/pickle]
                                       [--sequence-length 5000]
                                       [--seed 42]
                                       [--suffix Fresh2026]
                                       [--min-packets 50]

Output files: X_train_Fresh2026.pkl, y_train_Fresh2026.pkl, etc.
"""

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from src.data.preprocessing import pad_or_truncate
from src.utils.config import PROJECT_ROOT, ensure_dirs, load_config
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def load_trace_file(path: Path) -> list[float]:
    """Load a trace text file and extract the direction-only sequence.

    Reads tab-separated lines (timestamp<TAB>direction), discards timestamps,
    returns list of direction values as floats.

    Args:
        path: Path to the trace file.

    Returns:
        List of direction values (+1.0 or -1.0).
    """
    directions = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning(f"{path.name}:{line_num}: malformed line, skipping")
                continue
            try:
                directions.append(float(parts[1]))
            except ValueError:
                logger.warning(
                    f"{path.name}:{line_num}: invalid direction '{parts[1]}', skipping"
                )
    return directions


def validate_traces(
    X: np.ndarray, y: np.ndarray, min_packets: int = 50
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validate and filter traces, removing malformed ones.

    Quality checks:
    1. Length: remove traces with fewer than min_packets non-zero elements.
    2. Direction balance: warn about traces with >95% one direction.
    3. Completeness: count instances per label, flag labels with < 80 instances.

    Args:
        X: Traces array of shape (num_samples, sequence_length).
        y: Labels array of shape (num_samples,).
        min_packets: Minimum non-zero packets to keep a trace.

    Returns:
        Tuple of (filtered_X, filtered_y, stats_dict).
    """
    non_zero_lengths = np.array([(trace != 0).sum() for trace in X])

    # Find traces that are too short
    short_mask = non_zero_lengths < min_packets
    num_short = short_mask.sum()

    # Check direction balance
    unidirectional = 0
    for i, trace in enumerate(X):
        real = trace[trace != 0]
        if len(real) == 0:
            continue
        out_frac = (real > 0).mean()
        if out_frac < 0.05 or out_frac > 0.95:
            logger.warning(
                f"Trace {i} (label={y[i]}): {out_frac:.0%} outgoing, "
                "possibly malformed"
            )
            unidirectional += 1

    # Remove short traces
    keep_mask = ~short_mask
    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]

    # Compute per-label counts after filtering
    label_counts = Counter(y_filtered.tolist())
    flagged_labels = [
        label for label in sorted(label_counts) if label_counts[label] < 80
    ]

    stats = {
        "total_input": len(X),
        "num_removed_short": int(num_short),
        "num_unidirectional_warnings": unidirectional,
        "total_after_filter": len(X_filtered),
        "length_mean": float(non_zero_lengths[keep_mask].mean()) if keep_mask.any() else 0,
        "length_min": int(non_zero_lengths[keep_mask].min()) if keep_mask.any() else 0,
        "length_max": int(non_zero_lengths[keep_mask].max()) if keep_mask.any() else 0,
        "num_labels": len(label_counts),
        "flagged_labels": flagged_labels,
    }

    if flagged_labels:
        logger.warning(
            f"Labels with < 80 instances after filtering: {flagged_labels}"
        )

    return X_filtered, y_filtered, stats


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stratified split into train/valid/test sets.

    Ensures each label is proportionally represented in each split.

    Args:
        X: Traces array of shape (num_samples, sequence_length).
        y: Labels array of shape (num_samples,).
        train_ratio: Fraction for training set.
        valid_ratio: Fraction for validation set.
        seed: Random seed for shuffling.

    Returns:
        Dict with keys 'train', 'valid', 'test', each containing (X, y) tuples.
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(y)

    train_indices = []
    valid_indices = []
    test_indices = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        n = len(label_indices)
        train_end = int(train_ratio * n)
        valid_end = train_end + int(valid_ratio * n)

        train_indices.extend(label_indices[:train_end])
        valid_indices.extend(label_indices[train_end:valid_end])
        test_indices.extend(label_indices[valid_end:])

    # Shuffle within each split
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    rng.shuffle(test_indices)

    return {
        "train": (X[train_indices], y[train_indices]),
        "valid": (X[valid_indices], y[valid_indices]),
        "test": (X[test_indices], y[test_indices]),
    }


def save_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    suffix: str,
) -> None:
    """Save X and y arrays for each split as pickle files.

    Args:
        splits: Dict from split_data().
        output_dir: Directory to save pickle files.
        suffix: Suffix for filenames (e.g., 'Fresh2026').
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (X_split, y_split) in splits.items():
        x_path = output_dir / f"X_{split_name}_{suffix}.pkl"
        y_path = output_dir / f"y_{split_name}_{suffix}.pkl"

        with open(x_path, "wb") as f:
            pickle.dump(X_split, f, protocol=4)
        with open(y_path, "wb") as f:
            pickle.dump(y_split, f, protocol=4)

        logger.info(
            f"Saved {split_name}: X={X_split.shape} ({x_path.name}), "
            f"y={y_split.shape} ({y_path.name})"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--traces-dir", type=str, default=None,
        help="Directory containing trace files (default: data/collected/traces)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for pickle files (default: data/collected/pickle)",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=None,
        help="Trace sequence length (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--suffix", type=str, default="Fresh2026",
        help="Suffix for output filenames (default: Fresh2026)",
    )
    parser.add_argument(
        "--min-packets", type=int, default=None,
        help="Minimum non-zero packets to keep a trace (default: 50)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    ensure_dirs(config)

    seed = args.seed if args.seed is not None else config["seed"]
    seq_length = args.sequence_length if args.sequence_length is not None else config["data"]["sequence_length"]
    min_packets = args.min_packets if args.min_packets is not None else config["collection"]["min_trace_packets"]

    set_seed(seed)

    traces_dir = (
        Path(args.traces_dir)
        if args.traces_dir
        else PROJECT_ROOT / config["data"]["collected_dir"] / "traces"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / config["data"]["collected_dir"] / "pickle"
    )

    # Find all trace files (files named {label}-{batch})
    trace_files = sorted(
        f for f in traces_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )

    if not trace_files:
        logger.error(f"No trace files found in {traces_dir}")
        sys.exit(1)

    logger.info(f"Found {len(trace_files)} trace files in {traces_dir}")

    # Load all traces
    all_traces = []
    all_labels = []
    for trace_file in tqdm(trace_files, desc="Loading traces"):
        name = trace_file.name  # e.g., "42-7" -> site 42, batch 7
        try:
            label = int(name.split("-")[0])
        except (ValueError, IndexError):
            logger.warning(f"Cannot parse label from filename '{name}', skipping")
            continue

        directions = load_trace_file(trace_file)
        if not directions:
            logger.warning(f"Empty trace file '{name}', skipping")
            continue

        padded = pad_or_truncate(np.array(directions, dtype=np.float32), seq_length)
        all_traces.append(padded)
        all_labels.append(label)

    if not all_traces:
        logger.error("No valid traces loaded")
        sys.exit(1)

    X = np.array(all_traces, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    logger.info(f"Loaded {len(X)} traces, {len(np.unique(y))} unique labels")

    # Validate and filter
    X, y, stats = validate_traces(X, y, min_packets=min_packets)

    logger.info(
        f"Validation: {stats['total_input']} input -> {stats['total_after_filter']} "
        f"after filtering ({stats['num_removed_short']} short traces removed)"
    )
    logger.info(
        f"Non-zero length: mean={stats['length_mean']:.0f}, "
        f"min={stats['length_min']}, max={stats['length_max']}"
    )

    if stats["total_after_filter"] == 0:
        logger.error("No traces remaining after filtering")
        sys.exit(1)

    # Stratified split
    train_ratio = config["preprocessing"]["train_ratio"]
    valid_ratio = config["preprocessing"]["valid_ratio"]
    splits = split_data(X, y, train_ratio, valid_ratio, seed)

    # Save
    save_splits(splits, output_dir, args.suffix)

    logger.info(f"Done. Pickle files saved to {output_dir}")


if __name__ == "__main__":
    main()
