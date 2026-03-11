#!/usr/bin/env python3
"""
Filter + resplit a DF-format closed-world dataset (X/y pickles).

Motivation:
Our crawler pipeline already drops extremely short traces (default min_packets=50),
but many remaining traces can still be very short/noisy compared to curated
benchmarks. This script lets you:
  - filter by non-pad length (count of non-zero entries in X row)
  - optionally cap per-class samples to balance
  - create a fresh randomized train/valid/test split

Input format:
  X_{train,valid,test}_Combined.pkl and y_{train,valid,test}_Combined.pkl
  (or another suffix via --suffix)
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def _load_pickle(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.array(pickle.load(f))


def _save_pickle(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    p = argparse.ArgumentParser(description="Filter + resplit Combined DF dataset")
    p.add_argument("--data_dir", default="/mnt/d/cs244c-combined")
    p.add_argument("--suffix", default="Combined")
    p.add_argument("--out_dir", default="/mnt/d/cs244c-combined-filtered")
    p.add_argument("--min_nonpad", type=int, default=1500,
                   help="Keep traces with >= this many non-zero packets")
    p.add_argument("--cap_per_class", type=int, default=None,
                   help="If set, cap each class to at most this many samples")
    p.add_argument("--min_per_class", type=int, default=30,
                   help="Drop classes with fewer than this many samples after filtering/capping")
    p.add_argument("--keep_top_k_classes", type=int, default=None,
                   help="If set, keep only the K classes with the most samples (after filtering/capping)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.80)
    p.add_argument("--valid_frac", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate splits
    X_parts, y_parts = [], []
    for split in ("train", "valid", "test"):
        X_parts.append(_load_pickle(in_dir / f"X_{split}_{args.suffix}.pkl").astype(np.float32))
        y_parts.append(_load_pickle(in_dir / f"y_{split}_{args.suffix}.pkl").astype(np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    n0 = len(X)
    classes = int(y.max() + 1)
    print(f"Loaded: {n0} traces, classes={classes}")

    # Filter by non-pad length
    nonpad = np.count_nonzero(X != 0, axis=1)
    keep = nonpad >= args.min_nonpad
    X, y, nonpad = X[keep], y[keep], nonpad[keep]
    print(f"After min_nonpad>={args.min_nonpad}: {len(X)} kept ({len(X)/n0:.1%})")
    print(f"Nonpad kept: min={nonpad.min()} p25={np.percentile(nonpad,25):.0f} "
          f"med={np.median(nonpad):.0f} p75={np.percentile(nonpad,75):.0f} max={nonpad.max()}")

    # Optional per-class cap to balance (applied before min_per_class drop)
    rng = np.random.default_rng(args.seed)
    idx_keep = []
    for c in range(classes):
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        if args.cap_per_class is not None and len(idx) > args.cap_per_class:
            idx = rng.choice(idx, size=args.cap_per_class, replace=False)
        idx_keep.append(idx)
    if not idx_keep:
        raise SystemExit("No classes left after filtering.")
    idx_keep = np.concatenate(idx_keep)
    rng.shuffle(idx_keep)
    X, y = X[idx_keep], y[idx_keep]

    # Drop classes with too few samples to support a stratified split
    counts = np.bincount(y, minlength=classes)
    keep_classes = np.flatnonzero(counts >= args.min_per_class)
    drop_classes = np.flatnonzero((counts > 0) & (counts < args.min_per_class))
    if len(drop_classes) > 0:
        mask = np.isin(y, keep_classes)
        X, y = X[mask], y[mask]
        old_to_new = {int(old): int(new) for new, old in enumerate(sorted(map(int, keep_classes)))}
        y = np.array([old_to_new[int(lbl)] for lbl in y], dtype=np.int64)
        classes = int(y.max() + 1) if len(y) else 0
        print(f"Dropped {len(drop_classes)} classes with <{args.min_per_class} samples; now classes={classes}")
    counts = np.bincount(y, minlength=classes)

    # Optional: keep only top-K classes by remaining count (max-accuracy mode)
    if args.keep_top_k_classes is not None:
        k = int(args.keep_top_k_classes)
        if k <= 0:
            raise SystemExit("--keep_top_k_classes must be positive")
        order = np.argsort(counts)[::-1]
        top = order[: min(k, len(order))]
        mask = np.isin(y, top)
        X, y = X[mask], y[mask]
        # remap labels to 0..k-1
        top_sorted = sorted(map(int, top))
        old_to_new = {old: new for new, old in enumerate(top_sorted)}
        y = np.array([old_to_new[int(lbl)] for lbl in y], dtype=np.int64)
        classes = int(y.max() + 1) if len(y) else 0
        counts = np.bincount(y, minlength=classes)
        print(f"Kept top-{k} classes by count; now classes={classes}, traces={len(X)}")
    if args.cap_per_class is not None:
        print(f"After cap_per_class={args.cap_per_class}: {len(X)} traces")
    print(f"Class counts: min={counts.min()} p10={int(np.percentile(counts,10))} "
          f"median={int(np.median(counts))} max={counts.max()} mean={counts.mean():.1f}")

    # Stratified resplit (so every class appears in every split)
    if len(X) == 0 or classes == 0:
        raise SystemExit("No traces left after filtering.")

    train_idx, valid_idx, test_idx = [], [], []
    for c in range(classes):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_c = len(idx)
        n_train_c = int(round(n_c * args.train_frac))
        n_valid_c = int(round(n_c * args.valid_frac))
        # ensure at least 1 example per split when possible
        if n_c >= 3:
            n_train_c = max(1, min(n_train_c, n_c - 2))
            n_valid_c = max(1, min(n_valid_c, n_c - n_train_c - 1))
        n_test_c = n_c - n_train_c - n_valid_c
        if n_test_c <= 0:
            # fall back to leaving at least 1 for test
            n_test_c = 1
            if n_valid_c > 1:
                n_valid_c -= 1
            else:
                n_train_c = max(1, n_train_c - 1)
        train_idx.append(idx[:n_train_c])
        valid_idx.append(idx[n_train_c:n_train_c + n_valid_c])
        test_idx.append(idx[n_train_c + n_valid_c:])

    train_idx = np.concatenate(train_idx)
    valid_idx = np.concatenate(valid_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)

    splits = {
        "train": (X[train_idx], y[train_idx]),
        "valid": (X[valid_idx], y[valid_idx]),
        "test": (X[test_idx], y[test_idx]),
    }

    for name, (Xs, ys) in splits.items():
        _save_pickle(out_dir / f"X_{name}_{args.suffix}.pkl", Xs)
        _save_pickle(out_dir / f"y_{name}_{args.suffix}.pkl", ys)
        print(f"Wrote {name:5s}: {len(Xs):6d}")

    print(f"\nOutput: {out_dir}")
    print("Next: python src/train_combined.py --data_dir <out_dir> --epochs 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

