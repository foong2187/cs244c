#!/usr/bin/env python3
"""Convert DF pickle datasets to defended pickle datasets.

Takes the original DF NoDef pickle files, reconstructs trace files with
synthetic timestamps, applies each defense simulator, and produces new
pickle files for each defense.

This is needed because the original DF dataset is direction-only (no
timestamps), but defense simulators need (timestamp, direction) pairs.
We assign synthetic timestamps at a constant inter-packet interval.

Usage:
    python scripts/pickle_to_defended_datasets.py \
        --input-dir data/raw/ClosedWorld/NoDef \
        --output-base data/raw/ClosedWorld \
        --defenses bro regulator tamaraw buflo

    # All defenses:
    python scripts/pickle_to_defended_datasets.py \
        --input-dir data/raw/ClosedWorld/NoDef --all
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from defenses import bro, regulator, tamaraw, buflo

logger = logging.getLogger(__name__)

# Mean inter-packet time for Tor traffic (~10ms per cell)
SYNTHETIC_IPT = 0.01

DEFENSE_MAP = {
    "bro": ("BRO", bro),
    "regulator": ("RegulaTor", regulator),
    "tamaraw": ("Tamaraw", tamaraw),
    "buflo": ("BuFLO", buflo),
}


def load_pkl(path: Path) -> np.ndarray:
    """Load pickle file (Python 2 or 3 compatible)."""
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    return np.array(data)


def directions_to_trace(directions: np.ndarray) -> list[list[float]]:
    """Convert a direction-only sequence to (timestamp, direction) trace.

    Assigns synthetic timestamps at a constant interval.
    Stops at the last non-zero direction (ignores padding).
    """
    # Find last non-zero element
    nonzero = np.nonzero(directions)[0]
    if len(nonzero) == 0:
        return []

    last_idx = nonzero[-1]
    trace = []
    for i in range(last_idx + 1):
        if directions[i] != 0:
            trace.append([i * SYNTHETIC_IPT, float(directions[i])])
    return trace


def apply_defense_to_sample(directions: np.ndarray, defense_name: str,
                            seed: int, seq_length: int = 5000) -> np.ndarray:
    """Apply a defense to a single direction sequence and return defended directions."""
    trace = directions_to_trace(directions)
    if not trace:
        return np.zeros(seq_length, dtype=np.float32)

    rng = np.random.RandomState(seed)

    if defense_name == "bro":
        defended = bro.simulate(trace, config="b1", rng=rng)
    elif defense_name == "regulator":
        defended = regulator.simulate(trace, rng=rng)
    elif defense_name == "tamaraw":
        defended = tamaraw.simulate(trace)
    elif defense_name == "buflo":
        defended = buflo.simulate(trace)
    else:
        raise ValueError(f"Unknown defense: {defense_name}")

    # Extract directions from defended trace
    defended_dirs = np.array([d for _, d in defended], dtype=np.float32)

    # Pad or truncate to seq_length
    if len(defended_dirs) >= seq_length:
        return defended_dirs[:seq_length]
    else:
        padded = np.zeros(seq_length, dtype=np.float32)
        padded[:len(defended_dirs)] = defended_dirs
        return padded


def process_split(X: np.ndarray, defense_name: str,
                  base_seed: int, seq_length: int = 5000) -> np.ndarray:
    """Apply defense to all samples in a split."""
    X_defended = np.zeros_like(X)
    for i in tqdm(range(len(X)), desc=f"    Defending", leave=False):
        X_defended[i] = apply_defense_to_sample(
            X[i], defense_name, seed=base_seed + i, seq_length=seq_length
        )
    return X_defended


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing NoDef pickle files",
    )
    parser.add_argument(
        "--output-base", type=str, default=None,
        help="Base output directory. Each defense gets a subdirectory. "
             "Default: same parent as input-dir",
    )
    parser.add_argument(
        "--defenses", nargs="+",
        choices=["bro", "regulator", "tamaraw", "buflo"],
        help="Which defenses to apply",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Apply all defenses",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--input-suffix", type=str, default="NoDef",
        help="Suffix of input pickle files (default: NoDef)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_dir = Path(args.input_dir).resolve()
    output_base = Path(args.output_base).resolve() if args.output_base else input_dir.parent

    if args.all:
        defenses = list(DEFENSE_MAP.keys())
    elif args.defenses:
        defenses = args.defenses
    else:
        parser.error("Specify --defenses or --all")

    in_suffix = args.input_suffix

    # Load all splits
    splits = {}
    for split_name in ["train", "valid", "test"]:
        x_path = input_dir / f"X_{split_name}_{in_suffix}.pkl"
        y_path = input_dir / f"y_{split_name}_{in_suffix}.pkl"

        if not x_path.exists():
            logger.warning(f"Missing {x_path}, skipping {split_name} split")
            continue

        X = load_pkl(x_path)
        y = load_pkl(y_path)
        splits[split_name] = (X, y)
        logger.info(f"Loaded {split_name}: X={X.shape}, y={y.shape}")

    if not splits:
        logger.error("No splits loaded")
        sys.exit(1)

    seq_length = splits[next(iter(splits))][0].shape[1]

    # Apply each defense
    for defense_name in defenses:
        suffix, _ = DEFENSE_MAP[defense_name]
        output_dir = output_base / suffix
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n=== Applying {defense_name} -> {output_dir} ===")

        seed_offset = 0
        for split_name, (X, y) in splits.items():
            logger.info(f"  Processing {split_name} ({len(X)} samples)...")

            X_defended = process_split(
                X, defense_name,
                base_seed=args.seed + seed_offset,
                seq_length=seq_length,
            )
            seed_offset += len(X)

            # Save
            x_path = output_dir / f"X_{split_name}_{suffix}.pkl"
            y_path = output_dir / f"y_{split_name}_{suffix}.pkl"

            with open(x_path, "wb") as f:
                pickle.dump(X_defended, f, protocol=4)
            with open(y_path, "wb") as f:
                pickle.dump(y, f, protocol=4)

            logger.info(f"  Saved {split_name}: X={X_defended.shape} -> {x_path.name}")

        logger.info(f"  Done: {output_dir}")

    logger.info("\n=== All defenses complete ===")


if __name__ == "__main__":
    main()
