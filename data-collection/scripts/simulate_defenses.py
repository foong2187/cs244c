#!/usr/bin/env python3
"""Apply WF defense simulators to raw traces and produce defended datasets.

Supports: BRO, RegulaTor, Tamaraw, BuFLO, WTF-PAD.
Each defense produces a separate directory of defended traces,
then converts them to DF-compatible pickle format.

Usage:
    python scripts/simulate_defenses.py --traces-dir data/collected/traces \
                                        --output-base data/collected \
                                        --defenses bro regulator tamaraw buflo

    # Run all defenses:
    python scripts/simulate_defenses.py --traces-dir data/collected/traces --all

    # Just one defense with custom params:
    python scripts/simulate_defenses.py --traces-dir data/collected/traces \
                                        --defenses bro --bro-config b2
"""

import argparse
import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from defenses import bro, regulator, tamaraw, buflo

logger = logging.getLogger(__name__)


def traces_to_pickle(traces_dir: Path, output_dir: Path, suffix: str,
                     seq_length: int = 5000, seed: int = 42,
                     min_packets: int = 50) -> None:
    """Convert defended trace files to DF-compatible pickle format.

    Inline version of traces_to_pickle.py to avoid config file dependency.
    """
    import pickle
    from collections import Counter

    output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(
        f for f in traces_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )

    all_traces = []
    all_labels = []
    for trace_file in trace_files:
        try:
            label = int(trace_file.name.split("-")[0])
        except (ValueError, IndexError):
            continue

        trace = load_trace(trace_file)
        if not trace:
            continue

        directions = np.array([d for _, d in trace], dtype=np.float32)
        # Pad or truncate to seq_length
        if len(directions) >= seq_length:
            padded = directions[:seq_length]
        else:
            padded = np.zeros(seq_length, dtype=np.float32)
            padded[:len(directions)] = directions

        all_traces.append(padded)
        all_labels.append(label)

    if not all_traces:
        raise ValueError("No valid traces loaded")

    X = np.array(all_traces, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Filter short traces
    non_zero = np.array([(t != 0).sum() for t in X])
    keep = non_zero >= min_packets
    X, y = X[keep], y[keep]

    if len(X) == 0:
        raise ValueError("No traces remaining after filtering")

    # Stratified split
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(y)
    train_idx, valid_idx, test_idx = [], [], []

    for label in unique_labels:
        idxs = np.where(y == label)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        t_end = int(0.8 * n)
        v_end = t_end + int(0.1 * n)
        train_idx.extend(idxs[:t_end])
        valid_idx.extend(idxs[t_end:v_end])
        test_idx.extend(idxs[v_end:])

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)

    for split_name, idxs in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        if not idxs:
            continue
        idxs = np.array(idxs)
        x_path = output_dir / f"X_{split_name}_{suffix}.pkl"
        y_path = output_dir / f"y_{split_name}_{suffix}.pkl"
        with open(x_path, "wb") as f:
            pickle.dump(X[idxs], f, protocol=4)
        with open(y_path, "wb") as f:
            pickle.dump(y[idxs], f, protocol=4)
        logger.info(f"    {split_name}: X={X[idxs].shape}, y={y[idxs].shape}")


DEFENSE_SUFFIXES = {
    "bro": "BRO",
    "regulator": "RegulaTor",
    "tamaraw": "Tamaraw",
    "buflo": "BuFLO",
    "wtfpad": "WTFPAD",
}


def load_trace(path: Path) -> list[list[float]]:
    """Load a trace file as list of [timestamp, direction] pairs."""
    trace = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            try:
                trace.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return trace


def write_trace(trace: list[list[float]], path: Path) -> None:
    """Write a defended trace to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ts, direction in trace:
            d = 1 if direction > 0 else -1
            f.write(f"{ts:.6f}\t{d}\n")


def _apply_defense_worker(args: tuple) -> tuple[str, bool]:
    """Worker function for parallel defense simulation.

    Returns (filename, success) tuple.
    """
    trace_path, output_path, defense_name, defense_kwargs, seed = args
    try:
        trace = load_trace(Path(trace_path))
        if not trace:
            return (str(trace_path), False)

        rng = np.random.RandomState(seed)

        if defense_name == "bro":
            defended = bro.simulate(trace, rng=rng, **defense_kwargs)
        elif defense_name == "regulator":
            defended = regulator.simulate(trace, rng=rng, **defense_kwargs)
        elif defense_name == "tamaraw":
            defended = tamaraw.simulate(trace, **defense_kwargs)
        elif defense_name == "buflo":
            defended = buflo.simulate(trace, **defense_kwargs)
        else:
            return (str(trace_path), False)

        write_trace(defended, Path(output_path))
        return (Path(trace_path).name, True)
    except Exception as e:
        logger.warning(f"Failed to defend {trace_path}: {e}")
        return (str(trace_path), False)


def apply_defense(defense_name: str, traces_dir: Path, output_dir: Path,
                  defense_kwargs: dict, seed: int = 42,
                  n_workers: int = 4) -> int:
    """Apply a defense to all traces in a directory.

    Args:
        defense_name: Name of the defense.
        traces_dir: Directory containing raw trace files.
        output_dir: Output directory for defended traces.
        defense_kwargs: Extra kwargs passed to the defense simulate() fn.
        seed: Base random seed. Each trace gets seed + trace_index.
        n_workers: Number of parallel workers.

    Returns:
        Number of successfully defended traces.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(
        f for f in traces_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )

    if not trace_files:
        logger.error(f"No trace files found in {traces_dir}")
        return 0

    # Build work items
    work_items = []
    for i, trace_file in enumerate(trace_files):
        output_path = output_dir / trace_file.name
        work_items.append((
            str(trace_file),
            str(output_path),
            defense_name,
            defense_kwargs,
            seed + i,
        ))

    # Process in parallel
    success_count = 0
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_apply_defense_worker, work_items),
            total=len(work_items),
            desc=f"  {defense_name}",
        ))

    for fname, ok in results:
        if ok:
            success_count += 1
        else:
            logger.warning(f"  Failed: {fname}")

    return success_count


def apply_wtfpad(traces_dir: Path, output_dir: Path,
                 config: str = "normal_rcv") -> int:
    """Apply WTF-PAD defense (subprocess-based)."""
    from defenses import wtfpad

    output_dir.mkdir(parents=True, exist_ok=True)

    ok = wtfpad.simulate_directory(traces_dir, output_dir, config=config)
    if ok:
        return len(list(output_dir.iterdir()))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--traces-dir", type=str, required=True,
        help="Directory containing raw (undefended) trace files",
    )
    parser.add_argument(
        "--output-base", type=str, default=None,
        help="Base output directory. Each defense gets a subdirectory. "
             "Default: same parent as traces-dir",
    )
    parser.add_argument(
        "--defenses", nargs="+",
        choices=["bro", "regulator", "tamaraw", "buflo", "wtfpad"],
        help="Which defenses to apply",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Apply all defenses (bro, regulator, tamaraw, buflo, wtfpad)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    # BRO-specific
    parser.add_argument(
        "--bro-config", type=str, default="b1", choices=["b1", "b2"],
        help="BRO configuration (default: b1)",
    )
    # RegulaTor-specific
    parser.add_argument(
        "--regulator-budget", type=int, default=None,
        help="RegulaTor max padding budget (default: 3550)",
    )
    # WTF-PAD-specific
    parser.add_argument(
        "--wtfpad-config", type=str, default="normal_rcv",
        help="WTF-PAD configuration (default: normal_rcv)",
    )
    # Pickle conversion
    parser.add_argument(
        "--pickle", action="store_true",
        help="Also convert defended traces to pickle format",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=5000,
        help="Sequence length for pickle conversion (default: 5000)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    traces_dir = Path(args.traces_dir).resolve()
    if not traces_dir.exists():
        logger.error(f"Traces directory not found: {traces_dir}")
        sys.exit(1)

    output_base = Path(args.output_base).resolve() if args.output_base else traces_dir.parent
    n_workers = args.workers or min(cpu_count(), 8)

    if args.all:
        defenses = ["bro", "regulator", "tamaraw", "buflo", "wtfpad"]
    elif args.defenses:
        defenses = args.defenses
    else:
        parser.error("Specify --defenses or --all")

    logger.info(f"Traces directory: {traces_dir}")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Defenses: {defenses}")
    logger.info(f"Workers: {n_workers}")

    results = {}

    for defense in defenses:
        suffix = DEFENSE_SUFFIXES[defense]
        defended_traces_dir = output_base / f"traces_{suffix}"

        logger.info(f"\n=== Applying {defense} -> {defended_traces_dir} ===")

        if defense == "wtfpad":
            count = apply_wtfpad(
                traces_dir, defended_traces_dir,
                config=args.wtfpad_config,
            )
        else:
            kwargs = {}
            if defense == "bro":
                kwargs["config"] = args.bro_config
            elif defense == "regulator" and args.regulator_budget:
                kwargs["params"] = dict(regulator.DEFAULT_PARAMS)
                kwargs["params"]["budget"] = args.regulator_budget

            count = apply_defense(
                defense, traces_dir, defended_traces_dir,
                defense_kwargs=kwargs,
                seed=args.seed,
                n_workers=n_workers,
            )

        results[defense] = count
        logger.info(f"  {defense}: {count} traces defended")

        # Optionally convert to pickle
        if args.pickle and count > 0:
            pickle_dir = output_base / f"pickle_{suffix}"
            logger.info(f"  Converting to pickle -> {pickle_dir}")
            try:
                traces_to_pickle(
                    defended_traces_dir, pickle_dir, suffix,
                    seq_length=args.sequence_length, seed=args.seed,
                )
                logger.info(f"  Pickle files saved to {pickle_dir}")
            except Exception as e:
                logger.error(f"  Pickle conversion failed: {e}")

    # Summary
    logger.info("\n=== Summary ===")
    for defense, count in results.items():
        logger.info(f"  {defense}: {count} traces")


if __name__ == "__main__":
    main()
