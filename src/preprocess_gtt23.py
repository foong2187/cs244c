"""
Preprocess the GTT23 HDF5 dataset into pickle files for DF training.

GTT23 stores Tor circuit metadata measured at exit relays. This script:
  1. Reads the HDF5 file and selects labels with sufficient samples.
  2. Subsamples each label to --max_samples (default 1000) to stay
     within memory limits -- the paper uses ~1000 per class.
  3. Extracts cell direction sequences (length 5000) from each circuit.
  4. Flips direction signs to match the paper's convention
     (GTT23: +1=toward client; paper: +1=outgoing from client).
  5. Splits data into train/valid/test sets.
  6. Saves pickle files in the directory layout expected by train_*.py.

Memory usage: ~2 GB for 95 classes x 1000 samples (the default).

Usage:
    python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5
    python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5 --open_world
    python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5 --max_samples 500
"""

import argparse
import gc
import os
import pickle
import sys

import numpy as np

try:
    import h5py
    import hdf5plugin  # noqa: F401 – registers HDF5 compression filters
except ImportError:
    sys.exit(
        "h5py and hdf5plugin are required. Install with:\n"
        "  pip install h5py hdf5plugin\n"
        "You may also need system HDF5 libs:\n"
        "  apt install libhdf5-mpi-dev h5utils hdf5-helpers hdf5-tools"
    )

INPUT_LENGTH = 5000
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')


def parse_args():
    p = argparse.ArgumentParser(
        description='Preprocess GTT23 HDF5 into pickle files for DF training')
    p.add_argument('--hdf5', type=str, required=True,
                   help='Path to the GTT23.hdf5 file')
    p.add_argument('--num_classes', type=int, default=95,
                   help='Number of monitored site classes (default: 95)')
    p.add_argument('--max_samples', type=int, default=1000,
                   help='Max samples per monitored class (default: 1000). '
                        'The paper uses ~1000. Controls memory usage.')
    p.add_argument('--min_samples', type=int, default=100,
                   help='Minimum circuits per label to be eligible '
                        '(default: 100)')
    p.add_argument('--split_ratio', type=float, nargs=3,
                   default=[0.8, 0.1, 0.1],
                   help='Train/valid/test split ratios (default: 0.8 0.1 0.1)')
    p.add_argument('--open_world', action='store_true',
                   help='Also generate open-world splits')
    p.add_argument('--ow_unmon_train', type=int, default=9000,
                   help='Unmonitored training traces for open-world '
                        '(default: 9000)')
    p.add_argument('--ow_unmon_test', type=int, default=9000,
                   help='Unmonitored test traces for open-world '
                        '(default: 9000)')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory (default: ../dataset/)')
    p.add_argument('--defense', type=str, default='NoDef',
                   help='Defense label for output subdirectory (default: NoDef)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--port', type=int, default=None,
                   help='Filter circuits by port (e.g. 443 for HTTPS)')
    p.add_argument('--min_len', type=int, default=1,
                   help='Min non-padding cells per circuit (default: 1)')
    return p.parse_args()


def load_label_counts(f):
    """Return a dict mapping label -> (count, index_row_position).

    Only stores counts and row positions to avoid holding millions of
    circuit indices in memory at once.
    """
    i_label = f['index']['label']
    label_info = {}
    for row_pos in range(len(i_label)):
        row = i_label[row_pos]
        label = row[0]
        if isinstance(label, bytes):
            label = label.decode('ascii', errors='replace')
        count = len(row[1])
        label_info[label] = (count, row_pos)
    return label_info


def load_indices_for_label(f, row_pos):
    """Load circuit indices for a single label by its row position."""
    return np.array(f['index']['label'][row_pos][1])


def extract_directions(circuits, circuit_indices, flip_sign=True,
                       length=INPUT_LENGTH, batch_size=200):
    """Extract direction sequences from circuit records.

    Reads in small batches to limit memory.
    """
    n = len(circuit_indices)
    X = np.zeros((n, length), dtype='float32')

    sorted_order = np.argsort(circuit_indices)
    sorted_indices = circuit_indices[sorted_order]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = sorted_indices[start:end]

        batch_records = circuits[batch_idx.tolist()]
        cells = batch_records['cells']

        for j, cell_seq in enumerate(cells):
            dirs = cell_seq['direction'].astype('float32')
            if flip_sign:
                dirs = -dirs
            out_idx = sorted_order[start + j]
            X[out_idx, :len(dirs)] = dirs[:length]

    return X


def select_top_labels(label_info, num_classes, min_samples):
    """Select labels with the most circuits.

    Returns list of (label, count, row_pos) sorted by count descending.
    """
    eligible = [(lbl, count, row_pos)
                for lbl, (count, row_pos) in label_info.items()
                if count >= min_samples]
    eligible.sort(key=lambda x: x[1], reverse=True)

    if len(eligible) < num_classes:
        print(f"WARNING: Only {len(eligible)} labels have >= {min_samples} "
              f"samples (requested {num_classes}). Using all {len(eligible)}.")
        num_classes = len(eligible)

    selected = eligible[:num_classes]
    print(f"Selected {len(selected)} labels. Sample counts range from "
          f"{selected[-1][1]} to {selected[0][1]}")
    return selected


def split_data(X, y, split_ratio, rng):
    """Shuffle and split into train/valid/test."""
    n = len(X)
    indices = rng.permutation(n)
    X, y = X[indices], y[indices]

    n_train = int(n * split_ratio[0])
    n_valid = int(n * split_ratio[1])

    return {
        'X_train': X[:n_train],
        'y_train': y[:n_train],
        'X_valid': X[n_train:n_train + n_valid],
        'y_valid': y[n_train:n_train + n_valid],
        'X_test': X[n_train + n_valid:],
        'y_test': y[n_train + n_valid:],
    }


def save_pickles(data_dict, output_dir, defense):
    """Save arrays as pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in data_dict.items():
        fname = f'{key}_{defense}.pkl'
        path = os.path.join(output_dir, fname)
        with open(path, 'wb') as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved {path}  shape={arr.shape}  dtype={arr.dtype}")


def make_closed_world(f, label_info, args, rng):
    """Build closed-world dataset, capping each class at max_samples."""
    circuits = f['circuits']
    selected = select_top_labels(label_info, args.num_classes, args.min_samples)

    total_samples = min(args.max_samples, selected[-1][1]) * len(selected)
    print(f"Will use up to {args.max_samples} samples/class "
          f"(~{total_samples} total, ~{total_samples * 5000 * 4 / 1e9:.1f} GB)")

    all_X = []
    all_y = []
    for class_id, (label, count, row_pos) in enumerate(selected):
        indices = load_indices_for_label(f, row_pos)

        use_n = min(len(indices), args.max_samples)
        if use_n < len(indices):
            choice = rng.choice(len(indices), size=use_n, replace=False)
            indices = indices[choice]

        print(f"  Class {class_id:3d}: {label[:20]}...  "
              f"using {use_n}/{count}")

        X_class = extract_directions(circuits, indices)
        all_X.append(X_class)
        all_y.append(np.full(use_n, class_id, dtype='int32'))

        del X_class, indices
        gc.collect()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    del all_X, all_y
    gc.collect()

    print(f"Closed-world total: {X.shape[0]} samples, "
          f"{len(selected)} classes")

    splits = split_data(X, y, args.split_ratio, rng)
    del X, y
    gc.collect()

    output_dir = os.path.join(
        args.output_dir or BASE_OUTPUT_DIR, 'ClosedWorld', args.defense)
    print(f"\nSaving closed-world pickles to {output_dir}")
    save_pickles(splits, output_dir, args.defense)

    del splits
    gc.collect()
    return selected


def make_open_world(f, label_info, selected_labels, args, rng):
    """Build open-world dataset with capped sample counts.

    Monitored sites = the labels chosen for closed-world (classes 0..N-1).
    Unmonitored sites = other labels pooled into one class (class N).
    """
    circuits = f['circuits']
    num_monitored = len(selected_labels)
    unmon_class = num_monitored

    monitored_label_set = {lbl for lbl, _, _ in selected_labels}

    # --- Monitored data (capped) ---
    print("Extracting monitored class data...")
    mon_X, mon_y = [], []
    for class_id, (label, count, row_pos) in enumerate(selected_labels):
        indices = load_indices_for_label(f, row_pos)
        use_n = min(len(indices), args.max_samples)
        if use_n < len(indices):
            choice = rng.choice(len(indices), size=use_n, replace=False)
            indices = indices[choice]

        X_class = extract_directions(circuits, indices)
        mon_X.append(X_class)
        mon_y.append(np.full(use_n, class_id, dtype='int32'))
        del X_class, indices
        gc.collect()

    mon_X = np.concatenate(mon_X)
    mon_y = np.concatenate(mon_y)

    # --- Unmonitored data (collect indices first, then cap) ---
    print("Collecting unmonitored circuit indices...")
    unmon_total_needed = args.ow_unmon_train + args.ow_unmon_test
    unmon_idx_pool = []
    collected = 0

    unmon_labels = [(lbl, info) for lbl, info in label_info.items()
                    if lbl not in monitored_label_set]
    rng.shuffle(unmon_labels)

    for lbl, (count, row_pos) in unmon_labels:
        if collected >= unmon_total_needed:
            break
        indices = load_indices_for_label(f, row_pos)
        take = min(len(indices), unmon_total_needed - collected)
        if take < len(indices):
            choice = rng.choice(len(indices), size=take, replace=False)
            indices = indices[choice]
        unmon_idx_pool.append(indices)
        collected += take

    if not unmon_idx_pool:
        print("WARNING: No unmonitored labels found. Skipping open-world.")
        return

    all_unmon_idx = np.concatenate(unmon_idx_pool)
    del unmon_idx_pool
    rng.shuffle(all_unmon_idx)
    all_unmon_idx = all_unmon_idx[:unmon_total_needed]

    print(f"Unmonitored pool: {len(all_unmon_idx)} circuits "
          f"(target: {unmon_total_needed})")

    unmon_X = extract_directions(circuits, all_unmon_idx)
    unmon_y = np.full(len(all_unmon_idx), unmon_class, dtype='int32')
    del all_unmon_idx
    gc.collect()

    # Split monitored: train+valid (90%) and test (10%)
    n_mon = len(mon_X)
    mon_perm = rng.permutation(n_mon)
    mon_X, mon_y = mon_X[mon_perm], mon_y[mon_perm]
    n_mon_test = max(1, int(n_mon * 0.1))

    X_test_Mon = mon_X[-n_mon_test:]
    y_test_Mon = mon_y[-n_mon_test:]
    X_mon_trainval = mon_X[:-n_mon_test]
    y_mon_trainval = mon_y[:-n_mon_test]
    del mon_X, mon_y
    gc.collect()

    # Split unmonitored
    n_unmon_train = min(args.ow_unmon_train, len(unmon_X) - 1)
    X_unmon_train = unmon_X[:n_unmon_train]
    y_unmon_train = unmon_y[:n_unmon_train]
    X_test_Unmon = unmon_X[n_unmon_train:]
    y_test_Unmon = unmon_y[n_unmon_train:]
    del unmon_X, unmon_y
    gc.collect()

    # Combine for training, split train/valid
    X_trainval = np.concatenate([X_mon_trainval, X_unmon_train])
    y_trainval = np.concatenate([y_mon_trainval, y_unmon_train])
    del X_mon_trainval, y_mon_trainval, X_unmon_train, y_unmon_train
    gc.collect()

    perm = rng.permutation(len(X_trainval))
    X_trainval, y_trainval = X_trainval[perm], y_trainval[perm]

    n_valid = max(1, int(len(X_trainval) * 0.1))
    X_train = X_trainval[:-n_valid]
    y_train = y_trainval[:-n_valid]
    X_valid = X_trainval[-n_valid:]
    y_valid = y_trainval[-n_valid:]
    del X_trainval, y_trainval
    gc.collect()

    print(f"Open-world splits:")
    print(f"  Train: {X_train.shape}  Valid: {X_valid.shape}")
    print(f"  Test Mon: {X_test_Mon.shape}  Test Unmon: {X_test_Unmon.shape}")

    output_dir = os.path.join(
        args.output_dir or BASE_OUTPUT_DIR, 'OpenWorld', args.defense)
    print(f"\nSaving open-world pickles to {output_dir}")

    save_pickles({
        'X_train': X_train, 'y_train': y_train,
        'X_valid': X_valid, 'y_valid': y_valid,
    }, output_dir, args.defense)

    save_pickles({
        'X_test_Mon': X_test_Mon, 'y_test_Mon': y_test_Mon,
        'X_test_Unmon': X_test_Unmon, 'y_test_Unmon': y_test_Unmon,
    }, output_dir, args.defense)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    print(f"Opening {args.hdf5} ...")
    with h5py.File(args.hdf5, 'r') as f:
        print(f"HDF5 keys: {list(f.keys())}")
        print(f"Circuits dataset shape: {f['circuits'].shape}")

        print("Loading label index (counts only)...")
        label_info = load_label_counts(f)
        print(f"Found {len(label_info)} unique labels")

        total_circuits = sum(c for c, _ in label_info.values())
        print(f"Total indexed circuits: {total_circuits}")

        # --- Closed-world ---
        print("\n=== Closed-World Preprocessing ===")
        selected = make_closed_world(f, label_info, args, rng)

        # --- Open-world ---
        if args.open_world:
            print("\n=== Open-World Preprocessing ===")
            make_open_world(f, label_info, selected, args, rng)

    print("\nDone.")


if __name__ == '__main__':
    main()
