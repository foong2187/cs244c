#!/usr/bin/env python3
"""
Plot train/val accuracy or loss curves from a Keras verbose=2 training log.

Supports logs emitted by `src/train_combined.py` and similar scripts.

Examples:
  # accuracy
  python scripts/plot_training_metrics_from_log.py \
    --log /path/to/log.txt \
    --metric accuracy \
    --out experiments/fig_training_curve.png \
    --title "Convergence"

  # loss
  python scripts/plot_training_metrics_from_log.py \
    --log /path/to/log.txt \
    --metric loss \
    --out experiments/fig_loss_curve.png \
    --title "Loss"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s*/\s*(\d+)\s*$")
ACC_RE = re.compile(r"accuracy:\s*([0-9]*\.[0-9]+).*?val_accuracy:\s*([0-9]*\.[0-9]+)")
LOSS_RE = re.compile(r"loss:\s*([0-9]*\.[0-9]+).*?val_loss:\s*([0-9]*\.[0-9]+)")


def parse_args():
    p = argparse.ArgumentParser(description="Plot train/val curves from training log")
    p.add_argument("--log", required=True)
    p.add_argument("--metric", choices=["accuracy", "loss"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs: list[int] = []
    train: list[float] = []
    val: list[float] = []
    current_epoch: int | None = None

    metric_re = ACC_RE if args.metric == "accuracy" else LOSS_RE
    lines = log_path.read_text(errors="replace").splitlines()

    for line in lines:
        m1 = EPOCH_RE.match(line.strip())
        if m1:
            current_epoch = int(m1.group(1))
            continue
        m2 = metric_re.search(line)
        if m2 and current_epoch is not None:
            epochs.append(current_epoch)
            train.append(float(m2.group(1)))
            val.append(float(m2.group(2)))
            current_epoch = None

    if not epochs:
        raise SystemExit(f"No {args.metric} lines found in {log_path}")

    title = args.title
    if title is None:
        title = "Training convergence" if args.metric == "accuracy" else "Training loss"

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(8.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    if args.metric == "accuracy":
        ax.plot(epochs, train, label="train accuracy", color="#2E86AB", linewidth=2)
        ax.plot(epochs, val, label="val accuracy", color="#F18F01", linewidth=2)
        ax.set_ylim(0, 1.0)
        best_i = max(range(len(val)), key=lambda i: val[i])
        ax.scatter([epochs[best_i]], [val[best_i]], color="#F18F01", zorder=3)
        ax.text(epochs[best_i], val[best_i] + 0.03,
                f"best val={val[best_i]:.3f} (ep {epochs[best_i]})",
                ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Accuracy")
    else:
        ax.plot(epochs, train, label="train loss", color="#2E86AB", linewidth=2)
        ax.plot(epochs, val, label="val loss", color="#F18F01", linewidth=2)
        best_i = min(range(len(val)), key=lambda i: val[i])
        ax.scatter([epochs[best_i]], [val[best_i]], color="#F18F01", zorder=3)
        ax.text(epochs[best_i], val[best_i] - 0.05,
                f"best val={val[best_i]:.3f} (ep {epochs[best_i]})",
                ha="center", va="top", fontsize=10)
        ax.set_ylabel("Loss")

    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(epochs)} epochs parsed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

