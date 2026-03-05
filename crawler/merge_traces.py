"""
Merge per-trace pickles from the crawler into DF-format datasets.

Reads all site_*_visit_*.pkl from --input_dir (each has "label" and "sequence"),
splits into train/valid/test (e.g. 80/10/10), and writes X_train_Yousef.pkl etc.
for use with train_yousef_closed_world.py (point --data_dir at the output dir).

Usage:
  python -m crawler.merge_traces --input_dir data/crawler-traces --output_dir data/yousef-data/pickle
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from .config import SEQUENCE_LENGTH


def parse_args():
    p = argparse.ArgumentParser(description="Merge crawler traces into DF train/valid/test pickles")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Directory containing site_*_visit_*.pkl files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for X_train_*.pkl, y_train_*.pkl, etc.")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   help="Train / valid / test ratio (default: 0.8 0.1 0.1)")
    p.add_argument("--suffix", type=str, default="Yousef",
                   help="Suffix for filenames (default: Yousef -> X_train_Yousef.pkl)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for split")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all traces
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".pkl")]
    if not files:
        print("No .pkl files in", args.input_dir)
        return 1
    traces = []
    for f in files:
        path = os.path.join(args.input_dir, f)
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        label = data.get("label", 0)
        seq = data.get("sequence")
        if seq is None:
            continue
        seq = np.asarray(seq, dtype=np.float32)
        if len(seq) != SEQUENCE_LENGTH:
            pad = np.zeros(SEQUENCE_LENGTH - len(seq), dtype=np.float32)
            seq = np.concatenate([seq, pad]) if len(seq) < SEQUENCE_LENGTH else seq[:SEQUENCE_LENGTH]
        traces.append((seq, label))
    if not traces:
        print("No valid traces.")
        return 1

    random.seed(args.seed)
    random.shuffle(traces)
    X = np.stack([t[0] for t in traces])
    y = np.array([t[1] for t in traces], dtype=np.int64)
    n = len(X)
    a, b = args.split[0], args.split[0] + args.split[1]
    n_train = int(n * a)
    n_valid = int(n * b) - n_train
    n_test = n - n_train - n_valid
    X_train, y_train = X[:n_train], y[:n_train]
    X_valid, y_valid = X[n_train : n_train + n_valid], y[n_train : n_train + n_valid]
    X_test, y_test = X[n_train + n_valid :], y[n_train + n_valid :]

    def save(name, x, yy):
        path_x = os.path.join(args.output_dir, f"X_{name}_{args.suffix}.pkl")
        path_y = os.path.join(args.output_dir, f"y_{name}_{args.suffix}.pkl")
        with open(path_x, "wb") as fp:
            pickle.dump(x, fp)
        with open(path_y, "wb") as fp:
            pickle.dump(yy, fp)
        print(f"  {path_x} {x.shape}, {path_y} {yy.shape}")

    print("Writing DF-format pickles:")
    save("train", X_train, y_train)
    save("valid", X_valid, y_valid)
    save("test", X_test, y_test)
    print("Done. Use train_yousef_closed_world.py with --data_dir", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
