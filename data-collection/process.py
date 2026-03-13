#!/usr/bin/env python3
# process.py - convert pcaps to direction sequences and split into train/val/test pickles
#
# In the parallel-collection pipeline, each collect pod writes its output to
# GCS under  gs://<bucket>/<run-prefix>/<user-id>/pcap/  and
#            gs://<bucket>/<run-prefix>/<user-id>/logs/
#
# This script (running as a single pod after all collect pods finish):
#   1. Pulls every user's pcap/ and logs/ from GCS into the local data dir.
#   2. Merges all progress.csv files to build the guard-IP map.
#   3. Processes all pcaps (same site_idx from different users = same label).
#   4. Writes pickles locally and then uploads them back to GCS.

import argparse
import csv
import logging
import pickle
import socket
import sys
from collections import defaultdict
from pathlib import Path

import dpkt
import numpy as np
from google.cloud import storage as gcs
from tqdm import tqdm

PICKLE_DIR    = Path(__file__).parent / "data" / "pickle"
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None,
                   help="Local base data directory (default: <script_dir>/data).")
    p.add_argument("--gcs-bucket", type=str, default=None,
                   help="GCS bucket name (no gs:// prefix). When set, pcaps are "
                        "pulled from GCS before processing and pickles are pushed "
                        "back to GCS afterwards.")
    p.add_argument("--gcs-prefix", type=str, default="",
                   help="Run prefix inside the GCS bucket, e.g. "
                        "'runs/wf-data-collection-pipeline-abc12'.")
    return p.parse_args()


def gcs_push(local_dir: Path, bucket_name: str, gcs_prefix: str):
    """Upload all files under local_dir recursively to gs://<bucket_name>/<gcs_prefix>/."""
    if not local_dir.exists() or not any(local_dir.rglob("*")):
        log.warning(f"  Nothing to upload from {local_dir}")
        return
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    uploaded = 0
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_dir)
        blob_name = f"{gcs_prefix.strip('/')}/{relative}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_file))
        log.info(f"  Uploaded {local_file.name} → gs://{bucket_name}/{blob_name}")
        uploaded += 1
    log.info(f"Upload complete: {uploaded} file(s) → gs://{bucket_name}/{gcs_prefix.strip('/')}")



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


def load_guard_ip_map(progress_files: list[Path]) -> dict[tuple[int, int, str], str]:
    """
    Build a (site_idx, instance, user_id) -> guard_ip map from all progress.csv files.

    The user_id is the name of the parent directory of the progress.csv file
    (e.g. data/user-003/progress.csv → user_id = "user-003").
    Including user_id in the key lets multiple users collect the same
    (site_idx, instance) pair without collision.
    """
    guard_map = {}
    for progress_file in progress_files:
        if not progress_file.exists():
            continue
        user_id = progress_file.parent.name  # e.g. "user-003"
        with open(progress_file, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                ip = row.get("guard_ip", "").strip()
                if not ip:
                    continue
                try:
                    key = (int(row["site_idx"]), int(row["instance"]), user_id)
                    guard_map[key] = ip
                except (KeyError, ValueError):
                    continue
    return guard_map


def get_fallback_guard_ip(progress_files: list[Path]) -> str | None:
    """Return the most common guard IP across all progress.csv files."""
    from collections import Counter
    ips = []
    for progress_file in progress_files:
        if not progress_file.exists():
            continue
        with open(progress_file, newline="") as f:
            for row in csv.DictReader(f):
                ip = row.get("guard_ip", "").strip()
                if ip:
                    ips.append(ip)
    if ips:
        return Counter(ips).most_common(1)[0][0]
    return None



def main():
    args = parse_args()

    base_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"

    # ── discover user directories ────────────────────────────────────────────
    # Collect pods wrote to <base_dir>/user-NNN/pcap/ on the shared node disk.
    # Data is already here — no GCS pull needed for raw pcaps.
    user_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("user-")
    )
    if not user_dirs:
        # Fallback: legacy single-user layout (base_dir/pcap/*.pcap)
        user_dirs_pcap = list(base_dir.glob("pcap/*.pcap"))
        if user_dirs_pcap:
            log.info("No user-NNN subdirectories found; using legacy single-user layout")
            user_dirs = [base_dir]
        else:
            log.error(f"No user directories or pcap files found under {base_dir}.")
            sys.exit(1)

    log.info(f"Found {len(user_dirs)} user director(ies): "
             f"{[d.name for d in user_dirs]}")

    # ── collect all pcaps and build guard-IP map ─────────────────────────────
    progress_files = []
    for ud in user_dirs:
        pf = ud / "logs" / "progress.csv"
        if not pf.exists():
            pf = ud / "progress.csv"   # legacy path
        progress_files.append(pf)

    guard_map = load_guard_ip_map(progress_files)
    fallback_guard_ip = get_fallback_guard_ip(progress_files)
    if not guard_map and not fallback_guard_ip:
        fallback_guard_ip = input(
            "Enter Tor guard IP (check collection.log for 'guard IP: ...'): "
        ).strip()
    if guard_map:
        log.info(f"Loaded per-pcap guard IPs for {len(guard_map)} traces "
                 f"({len(set(guard_map.values()))} unique guards)")
    else:
        log.info(f"No per-pcap guard map; using fallback IP: {fallback_guard_ip}")

    # ── gather all pcaps grouped by site_idx ─────────────────────────────────
    # Key insight: pcaps from different users with the same site_idx
    # are treated as independent instances of the SAME class.
    # Filename: {site_idx:03d}-{instance:03d}.pcap  (unchanged)
    by_site: dict[int, list[tuple[Path, str]]] = defaultdict(list)

    for ud in user_dirs:
        user_id = ud.name
        pcap_dir = ud / "pcap"
        if not pcap_dir.exists():
            log.warning(f"  No pcap/ dir under {ud}, skipping")
            continue
        for p in sorted(pcap_dir.glob("*.pcap")):
            try:
                parts = p.stem.split("-")
                site_idx = int(parts[0])
                by_site[site_idx].append((p, user_id))
            except ValueError:
                log.warning(f"Skipping unexpected filename: {p.name}")

    total_pcaps = sum(len(v) for v in by_site.values())
    if total_pcaps == 0:
        log.error("No pcap files found. Run collect.py first.")
        sys.exit(1)
    log.info(f"Found {total_pcaps} pcap files across {len(by_site)} site(s)")

    # ── process ──────────────────────────────────────────────────────────────
    global PICKLE_DIR
    PICKLE_DIR = base_dir / "pickle"
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = [], []
    skipped = 0

    for site_idx in tqdm(sorted(by_site), desc="Processing sites"):
        for (pcap_path, user_id) in by_site[site_idx]:
            try:
                instance = int(pcap_path.stem.split("-")[1])
            except (IndexError, ValueError):
                instance = -1
            guard_ip = guard_map.get((site_idx, instance, user_id), fallback_guard_ip)
            if not guard_ip:
                log.warning(f"  No guard IP for {pcap_path.name} (user={user_id}), skipping")
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
        "train": (X[:n_train],               y[:n_train]),
        "valid": (X[n_train:n_train+n_val],  y[n_train:n_train+n_val]),
        "test":  (X[n_train+n_val:],         y[n_train+n_val:]),
    }

    for split_name, (X_split, y_split) in splits.items():
        x_path = PICKLE_DIR / f"X_{split_name}_Fresh2026.pkl"
        y_path = PICKLE_DIR / f"y_{split_name}_Fresh2026.pkl"
        with open(x_path, "wb") as f:
            pickle.dump(X_split, f)
        with open(y_path, "wb") as f:
            pickle.dump(y_split, f)
        log.info(f"  {split_name:5s}: {len(X_split):6d} traces  → {x_path}")

    log.info("")
    log.info("=== Dataset summary ===")
    log.info(f"  Total valid traces : {n}")
    log.info(f"  Classes            : {num_classes}")
    log.info(f"  Sequence length    : {SEQ_LEN}")
    log.info(f"  Train / Val / Test : {n_train} / {n_val} / {n - n_train - n_val}")
    log.info(f"  Saved to           : {PICKLE_DIR}")

    # ── push pickles back to GCS ─────────────────────────────────────────────
    if args.gcs_bucket:
        prefix = args.gcs_prefix.strip("/")
        gcs_pickle_path = f"{prefix}/pickle" if prefix else "pickle"
        gcs_push(PICKLE_DIR, args.gcs_bucket, gcs_pickle_path)
        log.info("GCS upload complete.")
    else:
        log.info("")
        log.info("Next step: run  .venv/bin/python analyze.py")


if __name__ == "__main__":
    main()
