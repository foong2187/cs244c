#!/usr/bin/env python3
"""
analyze.py - Quick EDA on the collected WF dataset.

Loads the processed pickle files and checks:
  1. Completeness  — sites, traces per site, coverage heatmap
  2. Trace quality — length distribution, padding rate, short traces
  3. Direction     — outgoing/incoming ratio per site
  4. Outliers      — unusually short, long, or biased sites

Run from data-collection/:
    .venv/bin/python analyze.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — saves PNGs instead of showing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ────────────────────────────────────────────────────────────────────
PICKLE_DIR  = Path(__file__).parent / "data" / "pickle"
SITES_FILE  = Path(__file__).parent / "sites.txt"
OUT_DIR     = Path(__file__).parent / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 5000


# ── helpers ──────────────────────────────────────────────────────────────────
def load(split):
    xp = PICKLE_DIR / f"X_{split}_Fresh2026.pkl"
    yp = PICKLE_DIR / f"y_{split}_Fresh2026.pkl"
    if not xp.exists():
        print(f"  [warn] {xp.name} not found — skipping {split}")
        return None, None
    with open(xp, "rb") as f:
        X = np.array(pickle.load(f))
    with open(yp, "rb") as f:
        y = np.array(pickle.load(f))
    return X, y


def actual_length(seq):
    """Return the number of non-padding elements (first run of trailing zeros)."""
    arr = np.asarray(seq)
    nz = np.nonzero(arr)[0]
    return int(nz[-1]) + 1 if len(nz) else 0


# ── load all splits ───────────────────────────────────────────────────────────
print("Loading pickles...")
splits = {}
for s in ("train", "valid", "test"):
    X, y = load(s)
    if X is not None:
        splits[s] = (X, y)

if not splits:
    print("No pickle files found. Run process.py first.")
    sys.exit(1)

X_all = np.concatenate([v[0] for v in splits.values()])
y_all = np.concatenate([v[1] for v in splits.values()])

sites = [s.strip() for s in SITES_FILE.read_text().splitlines()
         if s.strip() and not s.startswith("#")]

num_classes = int(y_all.max()) + 1


# ════════════════════════════════════════════════════════════════════════════
# 1. COMPLETENESS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("1. COMPLETENESS")
print("═" * 60)

for name, (X, y) in splits.items():
    print(f"  {name:5s}: {len(X):6d} traces  shape={X.shape}")

print(f"\n  Total traces  : {len(X_all)}")
print(f"  Classes       : {num_classes}")

counts = np.bincount(y_all, minlength=num_classes)
print(f"\n  Traces / site : min={counts.min()}  max={counts.max()}  "
      f"mean={counts.mean():.1f}  median={np.median(counts):.0f}")

# sites with fewer than 70 traces (less than ~78% of target 90)
low = np.where(counts < 70)[0]
if len(low):
    print(f"\n  ⚠  Sites with <70 traces ({len(low)}):")
    for idx in low:
        name = sites[idx] if idx < len(sites) else f"site_{idx}"
        print(f"      [{idx:3d}] {name}  ({counts[idx]} traces)")
else:
    print("  ✓  All sites have ≥70 traces")

# coverage heatmap
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(num_classes), counts, color="steelblue", width=1.0, edgecolor="none")
ax.axhline(90, color="red", linestyle="--", linewidth=1, label="target (90)")
ax.set_xlabel("Site index")
ax.set_ylabel("Traces collected")
ax.set_title("Traces per site")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "1_traces_per_site.png", dpi=150)
plt.close()
print(f"\n  Saved → data/analysis/1_traces_per_site.png")


# ════════════════════════════════════════════════════════════════════════════
# 2. TRACE LENGTH / QUALITY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("2. TRACE LENGTH & QUALITY")
print("═" * 60)

# subsample for speed if large
sample_idx = np.random.default_rng(42).choice(len(X_all),
                                               min(len(X_all), 3000),
                                               replace=False)
lengths = np.array([actual_length(X_all[i]) for i in sample_idx])

print(f"  (length stats from {len(sample_idx)} sampled traces)")
print(f"  Min    : {lengths.min()}")
print(f"  Max    : {lengths.max()}")
print(f"  Mean   : {lengths.mean():.1f}")
print(f"  Median : {np.median(lengths):.0f}")
print(f"  Std    : {lengths.std():.1f}")

capped = (lengths == SEQ_LEN).sum()
print(f"\n  Traces hitting 5000-cap : {capped}/{len(lengths)} "
      f"({100*capped/len(lengths):.1f}%)")

padded = (lengths < SEQ_LEN).sum()
print(f"  Traces padded (< 5000)  : {padded}/{len(lengths)} "
      f"({100*padded/len(lengths):.1f}%)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(lengths, bins=60, color="steelblue", edgecolor="none")
ax.axvline(SEQ_LEN, color="red", linestyle="--", linewidth=1, label="5000 cap")
ax.set_xlabel("Actual packet count (non-padded)")
ax.set_ylabel("Frequency")
ax.set_title("Trace length distribution")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "2_trace_lengths.png", dpi=150)
plt.close()
print(f"\n  Saved → data/analysis/2_trace_lengths.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. DIRECTION RATIO
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("3. DIRECTION RATIO (outgoing %)")
print("═" * 60)

# % of non-padding packets that are outgoing (+1)
out_ratios = []
for i in sample_idx:
    seq = X_all[i]
    nonzero = seq[seq != 0]
    if len(nonzero) == 0:
        out_ratios.append(0.5)
    else:
        out_ratios.append((nonzero == 1).sum() / len(nonzero))
out_ratios = np.array(out_ratios)

print(f"  Mean outgoing % : {out_ratios.mean()*100:.1f}%")
print(f"  Std             : {out_ratios.std()*100:.1f}%")
print(f"  Min             : {out_ratios.min()*100:.1f}%")
print(f"  Max             : {out_ratios.max()*100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(out_ratios * 100, bins=50, color="darkorange", edgecolor="none")
ax.axvline(50, color="black", linestyle="--", linewidth=1, label="50% (balanced)")
ax.set_xlabel("% outgoing packets")
ax.set_ylabel("Frequency")
ax.set_title("Outgoing packet ratio per trace")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "3_direction_ratio.png", dpi=150)
plt.close()
print(f"\n  Saved → data/analysis/3_direction_ratio.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. PER-SITE STATS & OUTLIERS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("4. PER-SITE STATS & OUTLIERS")
print("═" * 60)

site_mean_len   = np.zeros(num_classes)
site_out_ratio  = np.zeros(num_classes)

for cls in range(num_classes):
    idx = np.where(y_all == cls)[0]
    if len(idx) == 0:
        continue
    # sample up to 30 per site for speed
    sel = idx[:30]
    lens = np.array([actual_length(X_all[i]) for i in sel])
    site_mean_len[cls] = lens.mean()

    ratios = []
    for i in sel:
        seq = X_all[i]
        nz  = seq[seq != 0]
        ratios.append((nz == 1).mean() if len(nz) else 0.5)
    site_out_ratio[cls] = np.mean(ratios)

# outlier thresholds
len_mean, len_std   = site_mean_len.mean(), site_mean_len.std()
rat_mean, rat_std   = site_out_ratio.mean(), site_out_ratio.std()

short_sites  = np.where(site_mean_len < len_mean - 2 * len_std)[0]
long_sites   = np.where(site_mean_len > len_mean + 2 * len_std)[0]
upload_heavy = np.where(site_out_ratio > rat_mean + 2 * rat_std)[0]
download_heavy = np.where(site_out_ratio < rat_mean - 2 * rat_std)[0]

def site_name(idx):
    return sites[idx] if idx < len(sites) else f"site_{idx}"

print(f"\n  Sites with unusually SHORT traces (>2σ below mean):")
if len(short_sites):
    for i in short_sites:
        print(f"    [{i:3d}] {site_name(i):<40s} mean_len={site_mean_len[i]:.0f}")
else:
    print("    none")

print(f"\n  Sites with unusually LONG traces (>2σ above mean):")
if len(long_sites):
    for i in long_sites:
        print(f"    [{i:3d}] {site_name(i):<40s} mean_len={site_mean_len[i]:.0f}")
else:
    print("    none")

print(f"\n  Upload-heavy sites (mostly outgoing, >2σ above mean):")
if len(upload_heavy):
    for i in upload_heavy:
        print(f"    [{i:3d}] {site_name(i):<40s} out%={site_out_ratio[i]*100:.1f}%")
else:
    print("    none")

print(f"\n  Download-heavy sites (mostly incoming, >2σ below mean):")
if len(download_heavy):
    for i in download_heavy:
        print(f"    [{i:3d}] {site_name(i):<40s} out%={site_out_ratio[i]*100:.1f}%")
else:
    print("    none")

# scatter: mean length vs direction ratio, colored by trace count
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(site_mean_len, site_out_ratio * 100,
                c=counts, cmap="viridis", alpha=0.8, s=40)
plt.colorbar(sc, ax=ax, label="Trace count")
ax.set_xlabel("Mean trace length (packets)")
ax.set_ylabel("Outgoing %")
ax.set_title("Per-site: trace length vs direction ratio")
plt.tight_layout()
fig.savefig(OUT_DIR / "4_per_site_scatter.png", dpi=150)
plt.close()
print(f"\n  Saved → data/analysis/4_per_site_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SUMMARY")
print("═" * 60)
print(f"  Total traces collected : {len(X_all)}")
print(f"  Sites with data        : {num_classes}")
print(f"  Train / Val / Test     : "
      + " / ".join(str(len(v[0])) for v in splits.values()))
print(f"  Avg trace length       : {site_mean_len.mean():.0f} packets")
print(f"  Avg outgoing ratio     : {site_out_ratio.mean()*100:.1f}%")
print(f"\n  Plots saved to: {OUT_DIR}")
print()
