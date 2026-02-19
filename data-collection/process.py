#!/usr/bin/env python3
# process.py - convert pcaps to direction sequences and split into train/val/test pickles

import csv
import logging
import pickle
import socket
import sys
from collections import defaultdict
from pathlib import Path

import dpkt
import numpy as np
from tqdm import tqdm

PCAP_DIR      = Path(__file__).parent / "data" / "pcap"
PROGRESS_FILE = Path(__file__).parent / "data" / "progress.csv"
PICKLE_DIR    = Path(__file__).parent / "data" / "pickle"
# Miro's training code expects files here:
MIRO_DIR      = Path(__file__).parent.parent / "dataset" / "ClosedWorld" / "NoDef"
SEQ_LEN       = 5000
MIN_PACKETS   = 50     # discard traces shorter than this (failed page loads)
SPLIT         = (0.80, 0.10, 0.10)  # train / val / test
RANDOM_SEED   = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)



def pcap_to_sequence(pcap_path: Path, guard_ip: str) -> list[int] | None:
    """
    Parse a pcap and return a direction-only sequence.

    Filters to packets involving guard_ip (the Tor entry relay).
    Returns None if the trace is too short (likely a failed page load).

    +1 = outgoing (client → guard)
    -1 = incoming (guard → client)
    """
    directions = []
    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for _ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    src = socket.inet_ntoa(ip.src)
                    dst = socket.inet_ntoa(ip.dst)

                    if dst == guard_ip:
                        directions.append(1)    # outgoing
                    elif src == guard_ip:
                        directions.append(-1)   # incoming
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"  Could not parse {pcap_path.name}: {e}")
        return None

    if len(directions) < MIN_PACKETS:
        return None

    if len(directions) >= SEQ_LEN:
        return directions[:SEQ_LEN]
    return directions + [0] * (SEQ_LEN - len(directions))


def load_guard_ip_map() -> dict[tuple[int, int], str]:
    """
    Build a (site_idx, instance) -> guard_ip map from progress.csv.
    Each pcap was captured on a specific guard; using the correct one
    ensures accurate direction filtering even when the guard rotated.
    """
    guard_map = {}
    if not PROGRESS_FILE.exists():
        return guard_map
    with open(PROGRESS_FILE, newline="") as f:
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


def get_fallback_guard_ip() -> str | None:
    """Return the most common guard IP from progress.csv as a fallback."""
    if not PROGRESS_FILE.exists():
        return None
    from collections import Counter
    ips = []
    with open(PROGRESS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            ip = row.get("guard_ip", "").strip()
            if ip:
                ips.append(ip)
    if ips:
        return Counter(ips).most_common(1)[0][0]
    return None



def main():
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)

    pcaps = sorted(PCAP_DIR.glob("*.pcap"))
    if not pcaps:
        log.error(f"No pcap files found in {PCAP_DIR}. Run collect.py first.")
        sys.exit(1)
    log.info(f"Found {len(pcaps)} pcap files")

    guard_map = load_guard_ip_map()
    fallback_guard_ip = get_fallback_guard_ip()
    if not guard_map and not fallback_guard_ip:
        fallback_guard_ip = input(
            "Enter Tor guard IP (check collection.log for 'guard IP: ...'): "
        ).strip()
    if guard_map:
        log.info(f"Loaded per-pcap guard IPs for {len(guard_map)} traces "
                 f"({len(set(guard_map.values()))} unique guards)")
    else:
        log.info(f"No per-pcap guard map; using fallback IP: {fallback_guard_ip}")

    # Filename format: {site_idx:03d}-{instance:03d}.pcap
    by_site: dict[int, list[Path]] = defaultdict(list)
    for p in pcaps:
        try:
            parts = p.stem.split("-")
            site_idx = int(parts[0])
            by_site[site_idx].append(p)
        except ValueError:
            log.warning(f"Skipping unexpected filename: {p.name}")

    log.info(f"Sites with data: {len(by_site)}")

    X_all, y_all = [], []
    skipped = 0

    for site_idx in tqdm(sorted(by_site), desc="Processing sites"):
        for pcap_path in by_site[site_idx]:
            try:
                instance = int(pcap_path.stem.split("-")[1])
            except (IndexError, ValueError):
                instance = -1
            guard_ip = guard_map.get((site_idx, instance), fallback_guard_ip)
            if not guard_ip:
                log.warning(f"  No guard IP for {pcap_path.name}, skipping")
                skipped += 1
                continue
            seq = pcap_to_sequence(pcap_path, guard_ip)
            if seq is None:
                skipped += 1
                log.debug(f"  Skipped short/failed trace: {pcap_path.name}")
                continue
            X_all.append(seq)
            y_all.append(site_idx)

    log.info(f"Valid traces: {len(X_all)}  |  Skipped (too short): {skipped}")

    if not X_all:
        log.error("No valid traces to process.")
        sys.exit(1)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    # Remap labels to 0..N-1 (in case some sites have zero valid traces)
    unique_labels = sorted(set(y_all))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y], dtype=np.int64)
    num_classes = len(unique_labels)
    log.info(f"Number of classes (sites with ≥1 valid trace): {num_classes}")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])

    splits = {
        "train": (X[:n_train],          y[:n_train]),
        "valid": (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        "test":  (X[n_train+n_val:],    y[n_train+n_val:]),
    }

    # Save pickles in two places:
    #   1. data/pickle/X_{split}_Fresh2026.pkl  — our own naming (concept drift)
    #   2. dataset/ClosedWorld/NoDef/X_{split}_NoDef.pkl — Miro's expected path
    MIRO_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, (X_split, y_split) in splits.items():
        # Our copy
        x_path = PICKLE_DIR / f"X_{split_name}_Fresh2026.pkl"
        y_path = PICKLE_DIR / f"y_{split_name}_Fresh2026.pkl"
        with open(x_path, "wb") as f:
            pickle.dump(X_split, f)
        with open(y_path, "wb") as f:
            pickle.dump(y_split, f)

        # Miro's copy
        mx_path = MIRO_DIR / f"X_{split_name}_NoDef.pkl"
        my_path = MIRO_DIR / f"y_{split_name}_NoDef.pkl"
        with open(mx_path, "wb") as f:
            pickle.dump(X_split, f)
        with open(my_path, "wb") as f:
            pickle.dump(y_split, f)

        log.info(f"  {split_name:5s}: {len(X_split):6d} traces  → {x_path.name} + {mx_path}")

    log.info("")
    log.info("=== Dataset summary ===")
    log.info(f"  Total valid traces : {n}")
    log.info(f"  Classes            : {num_classes}")
    log.info(f"  Sequence length    : {SEQ_LEN}")
    log.info(f"  Train / Val / Test : {n_train} / {n_val} / {n - n_train - n_val}")
    log.info(f"  Saved to           : {PICKLE_DIR}")
    log.info("")
    log.info("Next step: copy data/pickle/ to the src/ directory and run:")
    log.info("  python src/train_closed_world.py --defense NoDef")


if __name__ == "__main__":
    main()
