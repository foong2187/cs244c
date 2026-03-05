"""
Quick dataset analysis: completeness, trace quality, direction ratios, outliers.

Loads processed pickle files and prints stats. Optionally saves plots if
matplotlib is available.

Usage:
  python -m crawler.analyze
  python -m crawler.analyze --pickle_dir dataset/ClosedWorld/NoDef
"""

import argparse
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SEQ_LEN = 5000


def actual_length(seq):
    """Number of non-padding elements."""
    nz = np.nonzero(seq)[0]
    return int(nz[-1]) + 1 if len(nz) else 0


def load_split(pickle_dir, split):
    xp = os.path.join(pickle_dir, f"X_{split}_NoDef.pkl")
    yp = os.path.join(pickle_dir, f"y_{split}_NoDef.pkl")
    if not os.path.exists(xp):
        return None, None
    with open(xp, "rb") as f:
        X = np.array(pickle.load(f))
    with open(yp, "rb") as f:
        y = np.array(pickle.load(f))
    return X, y


def parse_args():
    p = argparse.ArgumentParser(description="Analyze processed WF dataset")
    p.add_argument("--pickle_dir", type=str,
                   default=os.path.join(REPO_ROOT, "dataset", "ClosedWorld", "NoDef"))
    p.add_argument("--site_list", type=str, default=None,
                   help="Optional site list for labeling")
    return p.parse_args()


def main():
    args = parse_args()

    sites = None
    if args.site_list and os.path.exists(args.site_list):
        with open(args.site_list) as f:
            sites = [l.strip().split("\t")[-1] for l in f
                     if l.strip() and not l.startswith("#")]

    print("Loading pickles...")
    splits = {}
    for s in ("train", "valid", "test"):
        X, y = load_split(args.pickle_dir, s)
        if X is not None:
            splits[s] = (X, y)

    if not splits:
        print(f"No pickle files found in {args.pickle_dir}. Run process.py first.")
        return 1

    X_all = np.concatenate([v[0] for v in splits.values()])
    y_all = np.concatenate([v[1] for v in splits.values()])
    num_classes = int(y_all.max()) + 1

    # 1. Completeness
    print("\n" + "=" * 60)
    print("1. COMPLETENESS")
    print("=" * 60)
    for name, (X, y) in splits.items():
        print(f"  {name:5s}: {len(X):6d} traces  shape={X.shape}")
    print(f"\n  Total traces : {len(X_all)}")
    print(f"  Classes      : {num_classes}")

    counts = np.bincount(y_all, minlength=num_classes)
    print(f"\n  Traces/site  : min={counts.min()}  max={counts.max()}  "
          f"mean={counts.mean():.1f}  median={np.median(counts):.0f}")

    low = np.where(counts < 50)[0]
    if len(low):
        print(f"\n  Sites with <50 traces ({len(low)}):")
        for idx in low:
            name = sites[idx] if sites and idx < len(sites) else f"site_{idx}"
            print(f"    [{idx:3d}] {name}  ({counts[idx]})")
    else:
        print("  All sites have >=50 traces")

    # 2. Trace length
    print("\n" + "=" * 60)
    print("2. TRACE LENGTH")
    print("=" * 60)

    sample_size = min(len(X_all), 3000)
    sample_idx = np.random.default_rng(42).choice(len(X_all), sample_size, replace=False)
    lengths = np.array([actual_length(X_all[i]) for i in sample_idx])

    print(f"  (sampled {sample_size} traces)")
    print(f"  Min    : {lengths.min()}")
    print(f"  Max    : {lengths.max()}")
    print(f"  Mean   : {lengths.mean():.1f}")
    print(f"  Median : {np.median(lengths):.0f}")
    print(f"  Std    : {lengths.std():.1f}")

    capped = (lengths == SEQ_LEN).sum()
    print(f"\n  Hit 5000 cap : {capped}/{sample_size} ({100 * capped / sample_size:.1f}%)")

    # 3. Direction ratio
    print("\n" + "=" * 60)
    print("3. DIRECTION RATIO")
    print("=" * 60)

    out_ratios = []
    for i in sample_idx:
        seq = X_all[i]
        nz = seq[seq != 0]
        out_ratios.append((nz == 1).sum() / len(nz) if len(nz) else 0.5)
    out_ratios = np.array(out_ratios)

    print(f"  Mean outgoing : {out_ratios.mean() * 100:.1f}%")
    print(f"  Std           : {out_ratios.std() * 100:.1f}%")

    # 4. Per-site outliers
    print("\n" + "=" * 60)
    print("4. PER-SITE OUTLIERS")
    print("=" * 60)

    site_mean_len = np.zeros(num_classes)
    for cls in range(num_classes):
        idx = np.where(y_all == cls)[0]
        if len(idx) == 0:
            continue
        sel = idx[:30]
        site_mean_len[cls] = np.mean([actual_length(X_all[i]) for i in sel])

    mean_l, std_l = site_mean_len.mean(), site_mean_len.std()
    short = np.where(site_mean_len < mean_l - 2 * std_l)[0]
    long_ = np.where(site_mean_len > mean_l + 2 * std_l)[0]

    def sname(idx):
        return sites[idx] if sites and idx < len(sites) else f"site_{idx}"

    if len(short):
        print(f"  Unusually SHORT traces ({len(short)} sites):")
        for i in short:
            print(f"    [{i:3d}] {sname(i):<40s} mean_len={site_mean_len[i]:.0f}")
    if len(long_):
        print(f"  Unusually LONG traces ({len(long_)} sites):")
        for i in long_:
            print(f"    [{i:3d}] {sname(i):<40s} mean_len={site_mean_len[i]:.0f}")
    if not len(short) and not len(long_):
        print("  No outliers detected")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total traces : {len(X_all)}")
    print(f"  Classes      : {num_classes}")
    print(f"  Splits       : " +
          " / ".join(f"{k}={len(v[0])}" for k, v in splits.items()))
    print(f"  Avg length   : {site_mean_len.mean():.0f} packets")
    print(f"  Avg outgoing : {out_ratios.mean() * 100:.1f}%")
    print()

    # Try to save plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = os.path.join(os.path.dirname(args.pickle_dir), "analysis")
        os.makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].bar(range(num_classes), counts, color="steelblue", width=1.0)
        axes[0].set_xlabel("Site index")
        axes[0].set_ylabel("Traces")
        axes[0].set_title("Traces per site")

        axes[1].hist(lengths, bins=50, color="steelblue")
        axes[1].axvline(SEQ_LEN, color="red", ls="--", label="5000 cap")
        axes[1].set_xlabel("Packet count")
        axes[1].set_title("Trace length distribution")
        axes[1].legend()

        axes[2].hist(out_ratios * 100, bins=40, color="darkorange")
        axes[2].axvline(50, color="black", ls="--")
        axes[2].set_xlabel("% outgoing")
        axes[2].set_title("Direction ratio")

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "dataset_analysis.png")
        fig.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")
    except ImportError:
        print("  (matplotlib not available, skipping plots)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
