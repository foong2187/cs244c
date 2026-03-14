#!/usr/bin/env python3
"""
Train DFNet on benchmark defenses and generate publication-quality
training curves (loss + validation accuracy) in side-by-side panels.

Reproduces the style: faint lines for training, bold for validation,
clean light background, color-coded per defense.

Usage:
    cd ~/cs244c
    .venv/bin/python scripts/plot_defense_curves.py
    # Skip retraining if CSVs already exist:
    .venv/bin/python scripts/plot_defense_curves.py --plot-only
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

HIST_DIR = REPO / "experiments" / "defense_histories"
OUT_PATH = REPO / "experiments" / "fig_defense_training_curves.png"

DEFENSES = ["NoDef", "RegulaTor", "BRO", "BuFLO", "Tamaraw"]
EPOCHS = 30

COLORS = {
    "NoDef":     "#4285F4",  # Google blue
    "RegulaTor": "#34A853",  # Google green
    "BRO":       "#F9AB00",  # amber/gold
    "BuFLO":     "#EA8600",  # dark orange
    "Tamaraw":   "#EA4335",  # Google red
}


def train_defense(defense: str) -> dict:
    """Train DFNet on a benchmark defense and return Keras history dict."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adamax
    from model import DFNet
    from data_utils import load_closed_world_data, preprocess_data

    tf.random.set_seed(0)
    np.random.seed(0)

    print(f"\n{'='*60}")
    print(f"  Training: {defense} ({EPOCHS} epochs)")
    print(f"{'='*60}")

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        load_closed_world_data(defense=defense)
    num_classes = int(np.max(y_train) + 1)

    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        num_classes=num_classes)

    model = DFNet.build(input_shape=(5000, 1), classes=num_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
        metrics=["accuracy"],
    )

    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    del X_train, y_train, X_valid, y_valid

    train_ds = train_ds.shuffle(50000, seed=0).batch(128).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(128).prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_ds, epochs=EPOCHS, verbose=2, validation_data=valid_ds)

    with tf.device("/cpu:0"):
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)
    score = model.evaluate(test_ds, verbose=0)
    print(f"  Test accuracy: {score[1]:.4f}")

    return history.history


def save_history(defense: str, history: dict):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    path = HIST_DIR / f"{defense}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for i in range(len(history["loss"])):
            w.writerow([
                i + 1,
                f"{history['loss'][i]:.6f}",
                f"{history['accuracy'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_accuracy'][i]:.6f}",
            ])
    print(f"  Saved history → {path}")


def load_history(defense: str) -> dict:
    path = HIST_DIR / f"{defense}.csv"
    data = {"epoch": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["loss"].append(float(row["loss"]))
            data["accuracy"].append(float(row["accuracy"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_accuracy"].append(float(row["val_accuracy"]))
    return data


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    rcParams["axes.linewidth"] = 0.6
    rcParams["xtick.major.width"] = 0.6
    rcParams["ytick.major.width"] = 0.6

    histories = {}
    for d in DEFENSES:
        histories[d] = load_history(d)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.16, wspace=0.28)

    for defense in DEFENSES:
        h = histories[defense]
        epochs = h["epoch"]
        color = COLORS[defense]

        # --- Loss panel ---
        ax_loss.plot(epochs, h["loss"], color=color, alpha=0.25, linewidth=1.4)
        ax_loss.plot(epochs, h["val_loss"], color=color, alpha=0.9, linewidth=2.0,
                     label=defense)

        # --- Accuracy panel ---
        train_acc_pct = [a * 100 for a in h["accuracy"]]
        val_acc_pct = [a * 100 for a in h["val_accuracy"]]

        ax_acc.plot(epochs, train_acc_pct, color=color, alpha=0.25, linewidth=1.4)
        ax_acc.plot(epochs, val_acc_pct, color=color, alpha=0.9, linewidth=2.0,
                    label=defense)

    # --- Style: Loss ---
    ax_loss.set_title("Training Loss", fontsize=12, fontweight="medium", pad=8)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax_loss.set_ylim(0, 2.0)
    ax_loss.set_xlim(1, EPOCHS)
    ax_loss.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#cccccc",
                   framealpha=0.95, loc="upper right")
    ax_loss.grid(True, alpha=0.15, linewidth=0.5)
    ax_loss.tick_params(labelsize=9)

    # --- Style: Accuracy ---
    ax_acc.set_title("Validation Accuracy", fontsize=12, fontweight="medium", pad=8)
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_ylim(0, 105)
    ax_acc.set_xlim(1, EPOCHS)
    ax_acc.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#cccccc",
                  framealpha=0.95, loc="lower right")
    ax_acc.grid(True, alpha=0.15, linewidth=0.5)
    ax_acc.tick_params(labelsize=9)

    fig.text(0.5, 0.03, "Faint = training     Bold = validation",
             ha="center", fontsize=8.5, color="#888888")

    fig.savefig(str(OUT_PATH), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved → {OUT_PATH}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, just plot from existing CSVs")
    args = parser.parse_args()

    if not args.plot_only:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        for defense in DEFENSES:
            history = train_defense(defense)
            save_history(defense, history)

    missing = [d for d in DEFENSES if not (HIST_DIR / f"{d}.csv").exists()]
    if missing:
        print(f"ERROR: Missing history CSVs for: {missing}")
        print("Run without --plot-only first.")
        sys.exit(1)

    make_plots()


if __name__ == "__main__":
    main()
