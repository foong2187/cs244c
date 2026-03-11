#!/usr/bin/env python3
"""
Parse Keras epoch logs and plot training/validation accuracy curves.

Works with logs like those produced by `src/train_combined.py` (verbose=2),
where each epoch emits a line containing:
  - accuracy: <float> ... val_accuracy: <float>

Example:
  python scripts/plot_training_curve_from_log.py \
    --log /home/.../terminals/853657.txt \
    --out experiments/fig_training_curve_maxacc.png \
    --title "DFNet convergence (max-accuracy subset)"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s*/\s*(\d+)\s*$")
METRIC_RE = re.compile(
    r"accuracy:\s*([0-9]*\.[0-9]+).*?val_accuracy:\s*([0-9]*\.[0-9]+)"
)


def parse_args():
    p = argparse.ArgumentParser(description="Plot train/val accuracy from log file")
    p.add_argument("--log", required=True, help="Path to training log (text)")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="Training convergence", help="Figure title")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = []
    acc = []
    val = []
    current_epoch = None

    text = log_path.read_text(errors="replace").splitlines()
    for line in text:
        m1 = EPOCH_RE.match(line.strip())
        if m1:
            current_epoch = int(m1.group(1))
            continue
        m2 = METRIC_RE.search(line)
        if m2 and current_epoch is not None:
            epochs.append(current_epoch)
            acc.append(float(m2.group(1)))
            val.append(float(m2.group(2)))
            current_epoch = None  # consume epoch

    if not epochs:
        raise SystemExit(f"No epoch accuracy lines found in {log_path}")

    # Plot
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(8.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(epochs, acc, label="train accuracy", color="#2E86AB", linewidth=2)
    ax.plot(epochs, val, label="val accuracy", color="#F18F01", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(args.title)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # annotate best val
    best_i = max(range(len(val)), key=lambda i: val[i])
    ax.scatter([epochs[best_i]], [val[best_i]], color="#F18F01", zorder=3)
    ax.text(epochs[best_i], val[best_i] + 0.03,
            f"best val={val[best_i]:.3f} (ep {epochs[best_i]})",
            ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(epochs)} epochs parsed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

