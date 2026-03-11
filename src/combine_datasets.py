#!/usr/bin/env python3
"""
Combine crawler pcap data from local + GCP machines into a single DF-format dataset.

Data sources (all use the same 100-site list, site_idx 0-99):
  1. Local crawler pcaps:  /mnt/d/cs244c-data/crawler-pcap/       (instances 0-349)
  2. GCP crawler pcaps:    /mnt/d/cs244c-data-gcp/data/crawler-pcap/ (instances 350+)
  3. (optional) Devin/Fresh2026 pickles
  4. (optional) Yousef pickles

Output:  /mnt/d/cs244c-combined/
  X_train_Combined.pkl, y_train_Combined.pkl, ...
  label_map.pkl   (new_label -> original site_idx)
  sources.pkl     (per-sample source tag for provenance)

Usage:
  cd /home/mswisher/cs244c && source .venv/bin/activate

  # Crawler data only (local + GCP):
  python src/combine_datasets.py

  # Include teammate data:
  python src/combine_datasets.py --include-devin --include-yousef

  # More workers for faster pcap processing:
  python src/combine_datasets.py --workers 8
"""

import argparse
import csv
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SPLIT_RATIO = (0.80, 0.10, 0.10)
RANDOM_SEED = 42
MIN_PACKETS = 50
SEQ_LEN = 5000


# ---------------------------------------------------------------------------
# Guard-IP helpers (same logic as crawler.process)
# ---------------------------------------------------------------------------

def load_guard_map(progress_path):
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


def get_fallback_guard(guard_map):
    if not guard_map:
        return None
    return Counter(guard_map.values()).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Pcap processing (runs in worker processes)
# ---------------------------------------------------------------------------

def _init_worker():
    """Import heavy modules once per worker process."""
    global _pcap_to_sequence
    from crawler.capture import pcap_to_sequence as _pts
    _pcap_to_sequence = _pts


def _process_one(args):
    """Process a single pcap file. Returns (site_idx, seq) or None."""
    pcap_path, guard_ip, seq_len, min_packets = args
    try:
        seq = _pcap_to_sequence(pcap_path, guard_ip, max_len=seq_len)
        nonpad = int((seq != 0).sum())
        if nonpad < min_packets:
            return None
        site_idx = int(os.path.basename(pcap_path).split("-")[0])
        return (site_idx, seq)
    except Exception:
        return None


def discover_pcaps(pcap_dir, guard_map, fallback_guard, seq_len, min_packets):
    tasks = []
    for fname in sorted(os.listdir(pcap_dir)):
        if not fname.endswith(".pcap"):
            continue
        try:
            parts = fname.replace(".pcap", "").split("-")
            site_idx, instance = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue
        guard_ip = guard_map.get((site_idx, instance), fallback_guard)
        tasks.append((os.path.join(pcap_dir, fname), guard_ip, seq_len, min_packets))
    return tasks


# ---------------------------------------------------------------------------
# Pickle loading for teammate datasets
# ---------------------------------------------------------------------------

def load_pickle_dataset(base_dir, suffix):
    """Load all splits from pickle dir and concatenate into one (X, y) pair."""
    def _lp(name):
        path = os.path.join(base_dir, name)
        with open(path, "rb") as f:
            return np.array(pickle.load(f))

    parts_X, parts_y = [], []
    for split in ("train", "valid", "test"):
        parts_X.append(_lp(f"X_{split}_{suffix}.pkl"))
        parts_y.append(_lp(f"y_{split}_{suffix}.pkl"))

    X = np.concatenate(parts_X, axis=0).astype(np.float32)
    y = np.concatenate(parts_y, axis=0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Combine crawler + teammate datasets into DF-format pickles"
    )
    p.add_argument("--local-pcap", default="/mnt/d/cs244c-data/crawler-pcap")
    p.add_argument("--gcp-pcap", default="/mnt/d/cs244c-data-gcp/data/crawler-pcap")
    p.add_argument(
        "--progress", default="/mnt/d/cs244c-data/crawler-traces/progress.csv",
        help="progress.csv with guard IPs for local crawls"
    )
    p.add_argument(
        "--gcp-progress", default=None,
        help="progress.csv for GCP crawls (instance 350+). If set, use for GCP pcaps; else infer guard from pcap."
    )
    p.add_argument("--devin-dir",
                    default=os.path.join(REPO_ROOT, "data", "devin-data", "data", "pickle"))
    p.add_argument("--yousef-dir",
                    default=os.path.join(REPO_ROOT, "data", "yousef-data", "pickle"))
    p.add_argument("--output", default="/mnt/d/cs244c-combined")

    p.add_argument("--skip-local", action="store_true",
                    help="Skip local crawler pcaps")
    p.add_argument("--skip-gcp", action="store_true",
                    help="Skip GCP crawler pcaps")
    p.add_argument("--include-devin", action="store_true",
                    help="Include Devin/Fresh2026 pickle data")
    p.add_argument("--include-yousef", action="store_true",
                    help="Include Yousef pickle data")

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--min-packets", type=int, default=MIN_PACKETS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ----- Guard IP maps -----
    print("Loading guard IP map …")
    guard_map = load_guard_map(args.progress)
    fallback_guard = get_fallback_guard(guard_map)
    print(f"  {len(guard_map)} entries, fallback guard: {fallback_guard}")

    gcp_guard_map = None
    if args.gcp_progress and os.path.exists(args.gcp_progress):
        gcp_guard_map = load_guard_map(args.gcp_progress)
        print(f"  GCP progress: {len(gcp_guard_map)} entries (for instance 350+)")

    # ----- Discover pcap files -----
    all_tasks = []

    if not args.skip_local and os.path.isdir(args.local_pcap):
        local_tasks = discover_pcaps(
            args.local_pcap, guard_map, fallback_guard,
            args.seq_len, args.min_packets
        )
        print(f"Local pcaps:  {len(local_tasks):,} files")
        all_tasks.extend(local_tasks)

    if not args.skip_gcp and os.path.isdir(args.gcp_pcap):
        gcp_map = gcp_guard_map if gcp_guard_map is not None else guard_map
        gcp_fallback = get_fallback_guard(gcp_map) if gcp_map else None
        gcp_tasks = discover_pcaps(
            args.gcp_pcap, gcp_map, gcp_fallback,
            args.seq_len, args.min_packets
        )
        print(f"GCP pcaps:    {len(gcp_tasks):,} files")
        all_tasks.extend(gcp_tasks)

    # ----- Process pcaps -----
    sources = []
    all_X, all_y = [], []

    if all_tasks:
        print(f"\nProcessing {len(all_tasks):,} pcaps with {args.workers} workers …")
        t0 = time.time()
        valid, skipped = 0, 0
        X_pcap, y_pcap = [], []

        with Pool(processes=args.workers, initializer=_init_worker) as pool:
            for result in pool.imap_unordered(_process_one, all_tasks, chunksize=64):
                if result is not None:
                    site_idx, seq = result
                    X_pcap.append(seq)
                    y_pcap.append(site_idx)
                    valid += 1
                else:
                    skipped += 1
                total = valid + skipped
                if total % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    eta = (len(all_tasks) - total) / max(rate, 1)
                    print(f"  {total:,}/{len(all_tasks):,}  "
                          f"({rate:.0f}/s, ETA {eta:.0f}s)  "
                          f"valid={valid:,}  skipped={skipped:,}")

        elapsed = time.time() - t0
        print(f"  Pcap processing done: {valid:,} valid, {skipped:,} skipped "
              f"in {elapsed:.1f}s ({valid/max(elapsed,1):.0f}/s)")

        if X_pcap:
            X_pcap = np.array(X_pcap, dtype=np.float32)
            y_pcap = np.array(y_pcap, dtype=np.int64)
            sources.append(("crawler", len(X_pcap)))
            all_X.append(X_pcap)
            all_y.append(y_pcap)

    # ----- Optional: teammate data -----
    if args.include_devin and os.path.isdir(args.devin_dir):
        print(f"\nLoading Devin/Fresh2026 from {args.devin_dir} …")
        Xd, yd = load_pickle_dataset(args.devin_dir, "Fresh2026")
        if Xd.shape[1] != args.seq_len:
            tmp = np.zeros((len(Xd), args.seq_len), dtype=np.float32)
            c = min(Xd.shape[1], args.seq_len)
            tmp[:, :c] = Xd[:, :c]
            Xd = tmp
        n_cls = len(np.unique(yd))
        print(f"  {len(Xd):,} samples, {n_cls} classes (labels {yd.min()}–{yd.max()})")
        if n_cls < 100:
            print(f"  ⚠  Only {n_cls} classes — labels may not align with site_idx. "
                  "Proceed with caution.")
        sources.append(("devin", len(Xd)))
        all_X.append(Xd)
        all_y.append(yd)

    if args.include_yousef and os.path.isdir(args.yousef_dir):
        print(f"\nLoading Yousef from {args.yousef_dir} …")
        Xy, yy = load_pickle_dataset(args.yousef_dir, "Fresh2026")
        if Xy.shape[1] != args.seq_len:
            tmp = np.zeros((len(Xy), args.seq_len), dtype=np.float32)
            c = min(Xy.shape[1], args.seq_len)
            tmp[:, :c] = Xy[:, :c]
            Xy = tmp
        n_cls = len(np.unique(yy))
        print(f"  {len(Xy):,} samples, {n_cls} classes (labels {yy.min()}–{yy.max()})")
        sources.append(("yousef", len(Xy)))
        all_X.append(Xy)
        all_y.append(yy)

    if not all_X:
        print("No data collected. Nothing to do.")
        return 1

    # ----- Combine -----
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\n{'='*55}")
    print(f"Combined total: {len(X):,} samples")
    for name, cnt in sources:
        print(f"  {name:10s}: {cnt:>7,}  ({100*cnt/len(X):5.1f}%)")

    # Remap labels to contiguous 0..N-1
    unique_labels = sorted(set(y.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y], dtype=np.int64)
    num_classes = len(unique_labels)
    print(f"Classes: {num_classes} (remapped to 0..{num_classes - 1})")

    # Save label mapping (new_label -> original site_idx)
    map_path = os.path.join(args.output, "label_map.pkl")
    with open(map_path, "wb") as f:
        pickle.dump({v: k for k, v in label_map.items()}, f)
    print(f"Label map: {map_path}")

    # Save per-sample source tags
    src_tags = []
    for name, cnt in sources:
        src_tags.extend([name] * cnt)
    src_path = os.path.join(args.output, "sources.pkl")
    with open(src_path, "wb") as f:
        pickle.dump(src_tags, f)

    # ----- Shuffle & split -----
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "valid": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test":  (X[n_train + n_val:], y[n_train + n_val:]),
    }

    print(f"\nWriting to {args.output}/")
    for name, (X_s, y_s) in splits.items():
        xp = os.path.join(args.output, f"X_{name}_Combined.pkl")
        yp = os.path.join(args.output, f"y_{name}_Combined.pkl")
        with open(xp, "wb") as f:
            pickle.dump(X_s, f)
        with open(yp, "wb") as f:
            pickle.dump(y_s, f)
        print(f"  {name:5s}: {len(X_s):>7,} samples")

    print(f"\n{'='*55}")
    print(f"Done!  {num_classes} classes, {args.seq_len}-length sequences")
    print(f"  Train : {n_train:>7,}")
    print(f"  Valid : {n_val:>7,}")
    print(f"  Test  : {n - n_train - n_val:>7,}")
    print(f"  Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
