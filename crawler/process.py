"""
Convert pcap files to direction sequences and split into train/val/test pickles.

Reads pcaps from data/crawler-pcap/ and guard IPs from data/crawler-traces/progress.csv,
produces X_train/valid/test.pkl and y_train/valid/test.pkl ready for DF training.

Uses the per-pcap guard IP from progress.csv for accurate direction filtering,
even when the guard rotated during collection.

Usage:
  python -m crawler.process
  python -m crawler.process --min_packets 100 --output_dir data/processed
"""

import argparse
import csv
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crawler.config import DEFAULT_OUTPUT_DIR, DEFAULT_PCAP_DIR, SEQUENCE_LENGTH
from crawler.capture import pcap_to_sequence

SPLIT_RATIO = (0.80, 0.10, 0.10)
RANDOM_SEED = 42


def load_guard_map(progress_path):
    """Build (site_idx, instance) -> guard_ip map from progress CSV."""
    guard_map = {}
    if not os.path.exists(progress_path):
        return guard_map
    with open(progress_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            ip = row.get("guard_ip", "").strip()
            if not ip:
                continue
            try:
                key = (int(row["site_idx"]), int(row["instance"]))
                guard_map[key] = ip
            except (KeyError, ValueError):
                continue
    return guard_map


def get_fallback_guard(progress_path):
    """Most common guard IP from progress CSV."""
    if not os.path.exists(progress_path):
        return None
    ips = []
    with open(progress_path, newline="") as f:
        for row in csv.DictReader(f):
            ip = row.get("guard_ip", "").strip()
            if ip:
                ips.append(ip)
    if ips:
        return Counter(ips).most_common(1)[0][0]
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Process pcaps into DF training data")
    p.add_argument("--pcap_dir", type=str, default=DEFAULT_PCAP_DIR)
    p.add_argument("--progress", type=str,
                   default=os.path.join(DEFAULT_OUTPUT_DIR, "progress.csv"))
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(REPO_ROOT, "dataset", "ClosedWorld", "NoDef"))
    p.add_argument("--min_packets", type=int, default=50,
                   help="Discard traces with fewer non-pad packets")
    p.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find pcap files (named {site_idx:03d}-{instance:03d}.pcap)
    pcap_files = sorted(f for f in os.listdir(args.pcap_dir) if f.endswith(".pcap"))
    if not pcap_files:
        print(f"No pcap files in {args.pcap_dir}. Run the crawler first.")
        return 1
    print(f"Found {len(pcap_files)} pcap files")

    guard_map = load_guard_map(args.progress)
    fallback_guard = get_fallback_guard(args.progress)
    if guard_map:
        unique_guards = len(set(guard_map.values()))
        print(f"Per-pcap guard IPs for {len(guard_map)} traces ({unique_guards} unique guards)")
    elif fallback_guard:
        print(f"No per-pcap guard map; using fallback: {fallback_guard}")
    else:
        print("WARNING: no guard IP info. Will infer from pcap content.")

    # Group by site
    by_site = defaultdict(list)
    for fname in pcap_files:
        try:
            parts = fname.replace(".pcap", "").split("-")
            site_idx = int(parts[0])
            by_site[site_idx].append(fname)
        except (ValueError, IndexError):
            print(f"  Skipping unexpected filename: {fname}")

    print(f"Sites with pcaps: {len(by_site)}")

    X_all, y_all = [], []
    skipped = 0

    for site_idx in sorted(by_site):
        site_ok = 0
        for fname in by_site[site_idx]:
            try:
                instance = int(fname.replace(".pcap", "").split("-")[1])
            except (IndexError, ValueError):
                instance = -1

            guard_ip = guard_map.get((site_idx, instance), fallback_guard)
            pcap_path = os.path.join(args.pcap_dir, fname)
            seq = pcap_to_sequence(pcap_path, guard_ip, max_len=args.seq_len)
            nonpad = int((seq != 0).sum())

            if nonpad < args.min_packets:
                skipped += 1
                continue

            X_all.append(seq)
            y_all.append(site_idx)
            site_ok += 1

        if site_ok > 0:
            print(f"  site {site_idx:3d}: {site_ok} valid traces")

    print(f"\nValid: {len(X_all)}  Skipped: {skipped}")

    if not X_all:
        print("No valid traces. Check pcaps and guard IPs.")
        return 1

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    # Remap labels to contiguous 0..N-1
    unique_labels = sorted(set(y_all))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y_all], dtype=np.int64)
    num_classes = len(unique_labels)
    print(f"Classes: {num_classes}")

    # Shuffle and split
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "valid": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
    }

    for name, (X_s, y_s) in splits.items():
        xp = os.path.join(args.output_dir, f"X_{name}_NoDef.pkl")
        yp = os.path.join(args.output_dir, f"y_{name}_NoDef.pkl")
        with open(xp, "wb") as f:
            pickle.dump(X_s, f)
        with open(yp, "wb") as f:
            pickle.dump(y_s, f)
        print(f"  {name:5s}: {len(X_s):6d} traces -> {xp}")

    print(f"\n=== Dataset ready ===")
    print(f"  Classes  : {num_classes}")
    print(f"  Seq len  : {args.seq_len}")
    print(f"  Train    : {n_train}")
    print(f"  Valid    : {n_val}")
    print(f"  Test     : {n - n_train - n_val}")
    print(f"  Output   : {args.output_dir}")
    print(f"\nNext: python -m crawler.analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
