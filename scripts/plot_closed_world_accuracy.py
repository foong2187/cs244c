#!/usr/bin/env python3
"""
Create a paper-style closed-world accuracy comparison figure.

Outputs:
  experiments/fig_closed_world_accuracy_ours_vs_modern.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]


def load_modern_defenses_closed_world(summary_csv: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(summary_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["scenario"] != "closed":
                continue
            if row["metric"] != "test_top1_acc":
                continue
            out[row["defense"]] = float(row["value"])
    return out


def main() -> int:
    modern_csv = REPO / "results" / "modern_defenses_run_20260309_234617" / "summary.csv"
    modern = load_modern_defenses_closed_world(modern_csv)

    # Our crawled experiments (representative point estimates from logs)
    # These are intentionally included as separate bars (dataset variants), not "defenses".
    ours = {
        "Crawled 95c\nnp600 cap300\n(test)": 0.388737,
        "Crawled max-acc\n30c np1500\n(test)": 0.598817,
    }

    # Modern defenses (closed-world top-1)
    modern_order = ["NoDef", "RegulaTor", "BRO", "BuFLO", "Tamaraw", "WalkieTalkie"]
    modern_labels = [d for d in modern_order if d in modern]
    modern_vals = [modern[d] for d in modern_labels]

    # Plot
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(11, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Panel A: modern defenses
    bars = ax1.bar(range(len(modern_labels)), modern_vals, color="#2E86AB")
    ax1.set_xticks(range(len(modern_labels)))
    ax1.set_xticklabels(modern_labels, rotation=25, ha="right")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Closed-world test accuracy (top-1)")
    ax1.set_title("A) Curated benchmark (modern_defenses)")
    ax1.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, modern_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)

    # Panel B: ours
    our_labels = list(ours.keys())
    our_vals = list(ours.values())
    bars2 = ax2.bar(range(len(our_labels)), our_vals, color="#F18F01")
    ax2.set_xticks(range(len(our_labels)))
    ax2.set_xticklabels(our_labels, rotation=0, ha="center")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("B) Our crawled data (selected variants)")
    ax2.grid(axis="y", alpha=0.25)
    for b, v in zip(bars2, our_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)

    fig.suptitle("Closed-world DF accuracy: curated benchmark vs. our crawled data", y=1.02)
    fig.tight_layout()

    out = REPO / "experiments" / "fig_closed_world_accuracy_ours_vs_modern.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

