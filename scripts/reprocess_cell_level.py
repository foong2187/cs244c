#!/usr/bin/env python3
"""
Reprocess all crawler pcaps using CELL-LEVEL extraction instead of raw packets.

The DF paper uses Tor cell-level direction sequences: each 512-byte Tor cell
counts as one event. Our original pipeline counted raw TCP packets (including
ACKs), which gave ~50/50 out/in ratio instead of the paper's ~15/85.

This script:
  1. Reads all pcaps (local + GCP)
  2. For each TCP packet with payload, emits ceil(payload / 512) direction events
  3. Filters, splits, and saves DF-format pickles

Usage:
  cd ~/cs244c
  .venv/bin/python scripts/reprocess_cell_level.py --workers 6
  .venv/bin/python scripts/reprocess_cell_level.py --workers 6 --min-cells 50
"""

import argparse
import csv
import math
import os
import pickle
import socket
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

CELL_SIZE = 512
SEQ_LEN = 5000
MIN_CELLS = 50
SPLIT_RATIO = (0.80, 0.10, 0.10)
RANDOM_SEED = 42


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


def _is_local_ip(ip_str):
    if not ip_str or ip_str == "0.0.0.0":
        return True
    parts = ip_str.split(".")
    if len(parts) != 4:
        return True
    try:
        a, b = int(parts[0]), int(parts[1])
        if a == 127 or a == 10:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
    except ValueError:
        return True
    return False


def infer_guard_from_pcap(pcap_path):
    """Infer guard IP as the non-local IP with the most packets."""
    import dpkt
    from dpkt import sll, sll2
    counts = {}
    try:
        with open(pcap_path, "rb") as f:
            reader = dpkt.pcap.Reader(f)
            dl = reader.datalink()
            for _, buf in reader:
                try:
                    if dl == 1:
                        ip = dpkt.ethernet.Ethernet(buf).data
                    elif dl == 113:
                        ip = sll.SLL(buf).data
                    elif dl == 276:
                        ip = sll2.SLL2(buf).data
                    else:
                        continue
                    if not isinstance(ip, dpkt.ip.IP):
                        continue
                except Exception:
                    continue
                for addr_bytes in (ip.src, ip.dst):
                    addr = socket.inet_ntoa(addr_bytes)
                    if not _is_local_ip(addr):
                        counts[addr] = counts.get(addr, 0) + 1
    except Exception:
        return None
    if not counts:
        return None
    return max(counts, key=counts.get)


def pcap_to_cell_sequence(pcap_path, guard_ip, max_len=SEQ_LEN):
    """
    Extract cell-level direction sequence from pcap.
    
    For each TCP packet with payload to/from guard_ip, emit
    ceil(payload_bytes / 512) direction events (+1 or -1).
    Skip pure ACKs (payload == 0).
    """
    import dpkt
    from dpkt import sll, sll2

    def collect(gip):
        events = []
        with open(pcap_path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
                dl = reader.datalink()
            except dpkt.dpkt.UnpackError:
                return []
            for ts, buf in reader:
                try:
                    if dl == 1:
                        ip = dpkt.ethernet.Ethernet(buf).data
                    elif dl == 113:
                        ip = sll.SLL(buf).data
                    elif dl == 276:
                        ip = sll2.SLL2(buf).data
                    else:
                        continue
                    if not isinstance(ip, dpkt.ip.IP):
                        continue
                except Exception:
                    continue

                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)
                if src != gip and dst != gip:
                    continue

                if not isinstance(ip.data, dpkt.tcp.TCP):
                    continue
                payload_len = len(ip.data.data)
                if payload_len == 0:
                    continue

                direction = 1.0 if dst == gip else -1.0
                n_cells = max(1, math.ceil(payload_len / CELL_SIZE))
                for _ in range(n_cells):
                    events.append((ts, direction))
        return events

    events = collect(guard_ip or "0.0.0.0")
    if not events and (not guard_ip or guard_ip == "0.0.0.0"):
        inferred = infer_guard_from_pcap(pcap_path)
        if inferred:
            events = collect(inferred)

    if not events:
        return np.zeros(max_len, dtype=np.float32)

    events.sort(key=lambda x: x[0])
    dirs = np.array([d for _, d in events], dtype=np.float32)
    if len(dirs) > max_len:
        dirs = dirs[:max_len]
    elif len(dirs) < max_len:
        pad = np.zeros(max_len - len(dirs), dtype=np.float32)
        dirs = np.concatenate([dirs, pad])
    return dirs


def _process_one(args):
    pcap_path, guard_ip, seq_len, min_cells = args
    try:
        seq = pcap_to_cell_sequence(pcap_path, guard_ip, max_len=seq_len)
        nonpad = int((seq != 0).sum())
        if nonpad < min_cells:
            return None
        site_idx = int(os.path.basename(pcap_path).split("-")[0])
        return (site_idx, seq)
    except Exception:
        return None


def discover_pcaps(pcap_dir, guard_map, fallback_guard, seq_len, min_cells):
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
        tasks.append((os.path.join(pcap_dir, fname), guard_ip, seq_len, min_cells))
    return tasks


def parse_args():
    p = argparse.ArgumentParser(description="Reprocess pcaps at cell level")
    p.add_argument("--local-pcap", default="/mnt/d/cs244c-data/crawler-pcap")
    p.add_argument("--gcp-pcap", default="/mnt/d/cs244c-data-gcp/data/crawler-pcap")
    p.add_argument("--progress", default="/mnt/d/cs244c-data/crawler-traces/progress.csv")
    p.add_argument("--output", default="/mnt/d/cs244c-cell-level")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--min-cells", type=int, default=MIN_CELLS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--skip-gcp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("CELL-LEVEL REPROCESSING")
    print(f"  Cell size: {CELL_SIZE} bytes")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Min cells: {args.min_cells}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    # Guard IPs
    guard_map = load_guard_map(args.progress)
    fallback_guard = Counter(guard_map.values()).most_common(1)[0][0] if guard_map else None
    print(f"\nGuard map: {len(guard_map)} entries, fallback: {fallback_guard}")

    # Discover pcaps
    all_tasks = []
    if not args.skip_local and os.path.isdir(args.local_pcap):
        local_tasks = discover_pcaps(args.local_pcap, guard_map, fallback_guard,
                                     args.seq_len, args.min_cells)
        print(f"Local pcaps: {len(local_tasks):,}")
        all_tasks.extend(local_tasks)

    if not args.skip_gcp and os.path.isdir(args.gcp_pcap):
        gcp_tasks = discover_pcaps(args.gcp_pcap, {}, None,
                                   args.seq_len, args.min_cells)
        print(f"GCP pcaps:   {len(gcp_tasks):,}")
        all_tasks.extend(gcp_tasks)

    if not all_tasks:
        print("No pcaps found!")
        return 1

    # Process
    print(f"\nProcessing {len(all_tasks):,} pcaps ...")
    t0 = time.time()
    valid, skipped = 0, 0
    X_all, y_all = [], []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_one, all_tasks, chunksize=8):
            if result is not None:
                site_idx, seq = result
                X_all.append(seq)
                y_all.append(site_idx)
                valid += 1
            else:
                skipped += 1
            total = valid + skipped
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed
                eta = (len(all_tasks) - total) / max(rate, 1)
                print(f"  {total:,}/{len(all_tasks):,}  "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)  "
                      f"valid={valid:,} skipped={skipped:,}",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {valid:,} valid, {skipped:,} skipped in {elapsed:.1f}s")

    if not X_all:
        print("No valid traces!")
        return 1

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    # Stats
    unique_labels = sorted(set(y.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y], dtype=np.int64)
    num_classes = len(unique_labels)
    print(f"\nClasses: {num_classes}")
    print(f"Total samples: {len(X):,}")

    # Quick sanity: direction ratio
    sample = X[:min(500, len(X))]
    out_fracs = []
    lens = []
    for x in sample:
        nz = x[x != 0]
        if len(nz) > 0:
            out_fracs.append((nz == 1).sum() / len(nz))
            lens.append(len(nz))
    print(f"\nSanity check (first {len(out_fracs)} traces):")
    print(f"  Outgoing ratio: {np.mean(out_fracs):.4f}  (benchmark: 0.155)")
    print(f"  Incoming ratio: {1-np.mean(out_fracs):.4f}  (benchmark: 0.845)")
    print(f"  Median non-zero length: {np.median(lens):.0f}  (benchmark: 4022)")

    # Save label map
    with open(os.path.join(args.output, "label_map.pkl"), "wb") as f:
        pickle.dump({v: k for k, v in label_map.items()}, f)

    # Shuffle & split
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

    print(f"\n{'='*60}")
    print(f"DONE — cell-level dataset ready at {args.output}")
    print(f"  Classes: {num_classes}")
    print(f"  Train: {n_train:,} / Valid: {n_val:,} / Test: {n - n_train - n_val:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
