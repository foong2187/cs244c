#!/usr/bin/env python3
"""Generate open-world datasets for all defenses.

Open-world setting (per DF paper):
- 95 monitored sites (labels 0-94) + 1 unmonitored class (label 95) = 96 classes
- Training: monitored (80%) + unmonitored (80%), labels 0-95
- Validation: monitored (10%) + unmonitored (10%), labels 0-95
- Test monitored: X_test_Mon, y_test_Mon (remaining 10% monitored)
- Test unmonitored: X_test_Unmon, y_test_Unmon (remaining 10% unmonitored)

Usage:
    python scripts/generate_openworld.py \
        --mon-traces data/raw/ClosedWorld/NoDef/traces \
        --unmon-traces data/raw/unmonitored_traces \
        --output-base data/raw/OpenWorld \
        --defenses bro regulator tamaraw buflo nodef
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

N_MON_SITES = 95  # DF paper uses 95 monitored sites
UNMON_LABEL = 95  # unmonitored class label
SEQ_LENGTH = 5000
MIN_PACKETS = 50


def load_trace(path: Path) -> list[list[float]]:
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


def trace_to_directions(trace: list[list[float]]) -> np.ndarray:
    dirs = np.array([d for _, d in trace], dtype=np.float32)
    if len(dirs) >= SEQ_LENGTH:
        return dirs[:SEQ_LENGTH]
    padded = np.zeros(SEQ_LENGTH, dtype=np.float32)
    padded[:len(dirs)] = dirs
    return padded


def apply_defense(trace: list[list[float]], defense_name: str,
                  seed: int) -> list[list[float]]:
    rng = np.random.RandomState(seed)
    if defense_name == "nodef":
        return trace
    elif defense_name == "bro":
        return bro.simulate(trace, config="b1", rng=rng)
    elif defense_name == "regulator":
        return regulator.simulate(trace, rng=rng)
    elif defense_name == "tamaraw":
        return tamaraw.simulate(trace)
    elif defense_name == "buflo":
        return buflo.simulate(trace)
    raise ValueError(f"Unknown defense: {defense_name}")


SUFFIX_MAP = {
    "nodef": "NoDef",
    "bro": "BRO",
    "regulator": "RegulaTor",
    "tamaraw": "Tamaraw",
    "buflo": "BuFLO",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mon-traces", type=str, required=True,
                        help="Dir with monitored traces ({label}-{batch} files)")
    parser.add_argument("--unmon-traces", type=str, required=True,
                        help="Dir with unmonitored traces ({index} files)")
    parser.add_argument("--output-base", type=str, required=True,
                        help="Base output dir (each defense gets a subdir)")
    parser.add_argument("--defenses", nargs="+", required=True,
                        choices=["nodef", "bro", "regulator", "tamaraw", "buflo"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mon_dir = Path(args.mon_traces)
    unmon_dir = Path(args.unmon_traces)
    output_base = Path(args.output_base)

    # ---- Load monitored traces (only sites 0-94) ----
    logger.info("Loading monitored traces (sites 0-94)...")
    mon_files = sorted(
        f for f in mon_dir.iterdir()
        if f.is_file() and not f.name.startswith(".") and "-" in f.name
    )

    mon_by_site = {}
    for f in mon_files:
        try:
            label = int(f.name.split("-")[0])
        except (ValueError, IndexError):
            continue
        if label >= N_MON_SITES:
            continue
        if label not in mon_by_site:
            mon_by_site[label] = []
        mon_by_site[label].append(f)

    logger.info(f"  {len(mon_by_site)} monitored sites, "
                f"{sum(len(v) for v in mon_by_site.values())} total traces")

    # ---- Load unmonitored traces ----
    logger.info("Loading unmonitored traces...")
    unmon_files = sorted(
        f for f in unmon_dir.iterdir()
        if f.is_file() and not f.name.startswith(".") and "-" not in f.name
    )
    logger.info(f"  {len(unmon_files)} unmonitored traces")

    # ---- Split monitored: 80/10/10 per site ----
    rng = np.random.RandomState(args.seed)

    mon_train_files, mon_valid_files, mon_test_files = [], [], []
    mon_train_labels, mon_valid_labels, mon_test_labels = [], [], []

    for label in sorted(mon_by_site.keys()):
        files = mon_by_site[label]
        indices = np.arange(len(files))
        rng.shuffle(indices)
        n = len(indices)
        t_end = int(0.8 * n)
        v_end = t_end + int(0.1 * n)

        for i in indices[:t_end]:
            mon_train_files.append(files[i])
            mon_train_labels.append(label)
        for i in indices[t_end:v_end]:
            mon_valid_files.append(files[i])
            mon_valid_labels.append(label)
        for i in indices[v_end:]:
            mon_test_files.append(files[i])
            mon_test_labels.append(label)

    # ---- Split unmonitored: 80/10/10 ----
    unmon_indices = np.arange(len(unmon_files))
    rng.shuffle(unmon_indices)
    n_unmon = len(unmon_indices)
    ut_end = int(0.8 * n_unmon)
    uv_end = ut_end + int(0.1 * n_unmon)

    unmon_train_files = [unmon_files[i] for i in unmon_indices[:ut_end]]
    unmon_valid_files = [unmon_files[i] for i in unmon_indices[ut_end:uv_end]]
    unmon_test_files = [unmon_files[i] for i in unmon_indices[uv_end:]]

    logger.info(f"  Monitored split:   train={len(mon_train_files)} "
                f"valid={len(mon_valid_files)} test={len(mon_test_files)}")
    logger.info(f"  Unmonitored split: train={len(unmon_train_files)} "
                f"valid={len(unmon_valid_files)} test={len(unmon_test_files)}")

    # ---- Process each defense ----
    for defense_name in args.defenses:
        suffix = SUFFIX_MAP[defense_name]
        out_dir = output_base / suffix
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n=== {defense_name} -> {out_dir} ===")

        seed_offset = 0

        def process_files(files, labels, desc):
            nonlocal seed_offset
            X_list, y_list = [], []
            for f, lab in tqdm(zip(files, labels), total=len(files),
                               desc=f"  {desc}", leave=False):
                trace = load_trace(f)
                if not trace:
                    continue
                defended = apply_defense(trace, defense_name, args.seed + seed_offset)
                seed_offset += 1
                dirs = trace_to_directions(defended)
                if (dirs != 0).sum() < MIN_PACKETS:
                    continue
                X_list.append(dirs)
                y_list.append(lab)
            return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

        def process_unmon(files, desc):
            nonlocal seed_offset
            X_list, y_list = [], []
            for f in tqdm(files, desc=f"  {desc}", leave=False):
                trace = load_trace(f)
                if not trace:
                    continue
                defended = apply_defense(trace, defense_name, args.seed + seed_offset)
                seed_offset += 1
                dirs = trace_to_directions(defended)
                if (dirs != 0).sum() < MIN_PACKETS:
                    continue
                X_list.append(dirs)
                y_list.append(UNMON_LABEL)
            return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

        # Training: monitored + unmonitored combined
        X_mon_train, y_mon_train = process_files(
            mon_train_files, mon_train_labels, "train-mon")
        X_unmon_train, y_unmon_train = process_unmon(
            unmon_train_files, "train-unmon")
        X_train = np.concatenate([X_mon_train, X_unmon_train])
        y_train = np.concatenate([y_mon_train, y_unmon_train])
        # Shuffle
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        # Validation: monitored + unmonitored combined
        X_mon_valid, y_mon_valid = process_files(
            mon_valid_files, mon_valid_labels, "valid-mon")
        X_unmon_valid, y_unmon_valid = process_unmon(
            unmon_valid_files, "valid-unmon")
        X_valid = np.concatenate([X_mon_valid, X_unmon_valid])
        y_valid = np.concatenate([y_mon_valid, y_unmon_valid])
        idx = rng.permutation(len(X_valid))
        X_valid, y_valid = X_valid[idx], y_valid[idx]

        # Test: separate monitored and unmonitored
        X_test_Mon, y_test_Mon = process_files(
            mon_test_files, mon_test_labels, "test-mon")
        X_test_Unmon, y_test_Unmon = process_unmon(
            unmon_test_files, "test-unmon")

        # Save
        def save(name, arr):
            with open(out_dir / f"{name}_{suffix}.pkl", "wb") as f:
                pickle.dump(arr, f, protocol=4)

        save("X_train", X_train)
        save("y_train", y_train)
        save("X_valid", X_valid)
        save("y_valid", y_valid)
        save("X_test_Mon", X_test_Mon)
        save("y_test_Mon", y_test_Mon)
        save("X_test_Unmon", X_test_Unmon)
        save("y_test_Unmon", y_test_Unmon)

        logger.info(f"  train:      X={X_train.shape} y={y_train.shape} "
                    f"(mon={len(X_mon_train)}, unmon={len(X_unmon_train)})")
        logger.info(f"  valid:      X={X_valid.shape} y={y_valid.shape}")
        logger.info(f"  test_Mon:   X={X_test_Mon.shape}")
        logger.info(f"  test_Unmon: X={X_test_Unmon.shape}")

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()
