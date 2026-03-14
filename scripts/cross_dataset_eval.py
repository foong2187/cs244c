#!/usr/bin/env python3
"""
Cross-dataset experiment: train on curated benchmark NoDef, evaluate on our crawled traces.

Steps:
  1. Load benchmark NoDef (dataset/ClosedWorld/NoDef), restrict to 95 overlapping classes (0-94)
  2. Train DFNet on benchmark train/valid
  3. Evaluate on benchmark test (sanity check — should be ~96%)
  4. Evaluate on our crawled test set (cross-dataset generalization)
  5. Save full training history + results to experiments/

Usage:
  cd ~/cs244c
  .venv/bin/python scripts/cross_dataset_eval.py
  .venv/bin/python scripts/cross_dataset_eval.py --crawled_dir /mnt/d/cs244c-combined-np600-min50-cap300
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import DFNet


def _lp(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f, encoding="latin1"))


def load_benchmark(base, suffix="NoDef", keep_classes=None):
    X_train = _lp(base / f"X_train_{suffix}.pkl").astype(np.float32)
    y_train = _lp(base / f"y_train_{suffix}.pkl").astype(np.int64)
    X_valid = _lp(base / f"X_valid_{suffix}.pkl").astype(np.float32)
    y_valid = _lp(base / f"y_valid_{suffix}.pkl").astype(np.int64)
    X_test = _lp(base / f"X_test_{suffix}.pkl").astype(np.float32)
    y_test = _lp(base / f"y_test_{suffix}.pkl").astype(np.int64)

    if keep_classes is not None:
        keep = set(keep_classes)
        for name in ["train", "valid", "test"]:
            X, y = locals()[f"X_{name}"], locals()[f"y_{name}"]
            mask = np.isin(y, list(keep))
            locals()[f"X_{name}"] = X[mask]
            locals()[f"y_{name}"] = y[mask]
        X_train = X_train[np.isin(y_train, list(keep))]
        y_train = y_train[np.isin(y_train, list(keep))]
        X_valid = X_valid[np.isin(y_valid, list(keep))]
        y_valid = y_valid[np.isin(y_valid, list(keep))]
        X_test = X_test[np.isin(y_test, list(keep))]
        y_test = y_test[np.isin(y_test, list(keep))]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_crawled_test(crawled_dir, suffix="Combined"):
    X_test = _lp(crawled_dir / f"X_test_{suffix}.pkl").astype(np.float32)
    y_test = _lp(crawled_dir / f"y_test_{suffix}.pkl").astype(np.int64)
    return X_test, y_test


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset: benchmark train → crawled eval")
    p.add_argument("--benchmark_dir", type=str,
                   default=str(REPO / "dataset" / "ClosedWorld" / "NoDef"))
    p.add_argument("--crawled_dir", type=str,
                   default="/mnt/d/cs244c-combined-np600-min50-cap300")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    benchmark_dir = Path(args.benchmark_dir)
    crawled_dir = Path(args.crawled_dir)
    num_classes = 95
    keep_classes = list(range(num_classes))

    print("=== Cross-dataset experiment ===")
    print(f"  Train on:  {benchmark_dir}")
    print(f"  Eval on:   {crawled_dir}")
    print(f"  Classes:   {num_classes} (sites 0-94)")
    print()

    # --- Load benchmark (restrict to 95 overlapping classes) ---
    X_train, y_train, X_valid, y_valid, X_test_bench, y_test_bench = load_benchmark(
        benchmark_dir, suffix="NoDef", keep_classes=keep_classes
    )
    print(f"Benchmark train: {X_train.shape}, valid: {X_valid.shape}, test: {X_test_bench.shape}")

    # --- Load crawled test ---
    X_test_crawled, y_test_crawled = load_crawled_test(crawled_dir)
    print(f"Crawled test:    {X_test_crawled.shape}")

    # --- Preprocess ---
    X_train = X_train[:, :, np.newaxis]
    X_valid = X_valid[:, :, np.newaxis]
    X_test_bench = X_test_bench[:, :, np.newaxis]
    X_test_crawled = X_test_crawled[:, :, np.newaxis]

    y_train_oh = to_categorical(y_train, num_classes)
    y_valid_oh = to_categorical(y_valid, num_classes)
    y_test_bench_oh = to_categorical(y_test_bench, num_classes)
    y_test_crawled_oh = to_categorical(y_test_crawled, num_classes)

    INPUT_SHAPE = (5000, 1)

    # --- Build model ---
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)
    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    print("Model compiled")

    # --- tf.data ---
    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid_oh))
    del X_train, y_train_oh, X_valid, y_valid_oh

    train_ds = train_ds.shuffle(50000, seed=0).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # --- Train ---
    print(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    history = model.fit(
        train_ds, epochs=args.epochs, verbose=2, validation_data=valid_ds
    )

    # --- Evaluate on benchmark test (sanity check) ---
    with tf.device("/cpu:0"):
        bench_ds = tf.data.Dataset.from_tensor_slices((X_test_bench, y_test_bench_oh))
    bench_ds = bench_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    score_bench = model.evaluate(bench_ds, verbose=2)
    print(f"\nBenchmark test accuracy (sanity): {score_bench[1]:.6f}")

    # --- Evaluate on crawled test (cross-dataset) ---
    with tf.device("/cpu:0"):
        crawled_ds = tf.data.Dataset.from_tensor_slices((X_test_crawled, y_test_crawled_oh))
    crawled_ds = crawled_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    score_crawled = model.evaluate(crawled_ds, verbose=2)
    print(f"Crawled test accuracy (cross):    {score_crawled[1]:.6f}")

    # --- Per-class accuracy on crawled ---
    y_pred_probs = model.predict(crawled_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    for true, pred in zip(y_test_crawled, y_pred):
        per_class_total[true] += 1
        if true == pred:
            per_class_correct[true] += 1
    per_class_acc = np.divide(per_class_correct, per_class_total,
                              out=np.zeros(num_classes), where=per_class_total > 0)
    active = per_class_total > 0
    print(f"\nPer-class accuracy (crawled): mean={per_class_acc[active].mean():.4f}, "
          f"min={per_class_acc[active].min():.4f}, max={per_class_acc[active].max():.4f}")
    worst5 = np.argsort(per_class_acc)[:5]
    for c in worst5:
        print(f"  class {c:3d}: {per_class_acc[c]:.4f} ({int(per_class_correct[c])}/{int(per_class_total[c])})")

    # --- Save results ---
    out_dir = REPO / "experiments"
    out_dir.mkdir(exist_ok=True)

    # Training history CSV
    hist_path = out_dir / "cross_dataset_training_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        for i in range(len(history.history["accuracy"])):
            w.writerow([
                i + 1,
                f"{history.history['accuracy'][i]:.6f}",
                f"{history.history['loss'][i]:.6f}",
                f"{history.history['val_accuracy'][i]:.6f}",
                f"{history.history['val_loss'][i]:.6f}",
            ])
    print(f"\nTraining history saved to {hist_path}")

    # Summary
    summary_path = out_dir / "cross_dataset_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Cross-Dataset Experiment: Benchmark Train → Crawled Eval\n\n")
        f.write(f"Run: {datetime.now().isoformat()}\n\n")
        f.write("## Setup\n")
        f.write(f"- Train/valid source: `{benchmark_dir}` (NoDef, classes 0-94)\n")
        f.write(f"- Benchmark test: same source (sanity check)\n")
        f.write(f"- Crawled test: `{crawled_dir}`\n")
        f.write(f"- Classes: {num_classes}\n")
        f.write(f"- Epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}\n\n")
        f.write("## Results\n")
        f.write(f"- **Benchmark test accuracy** (in-distribution): **{score_bench[1]:.4f}**\n")
        f.write(f"- **Crawled test accuracy** (cross-dataset): **{score_crawled[1]:.4f}**\n\n")
        f.write("## Interpretation\n")
        f.write("The gap between in-distribution and cross-dataset accuracy quantifies\n")
        f.write("how much of the benchmark's high accuracy is due to distribution-specific\n")
        f.write("patterns vs. genuinely generalizable website fingerprints.\n")
    print(f"Summary saved to {summary_path}")

    print("\n=== Done ===")
    print(f"  Benchmark test acc: {score_bench[1]:.4f}")
    print(f"  Crawled test acc:   {score_crawled[1]:.4f}")
    print(f"  Gap:                {score_bench[1] - score_crawled[1]:.4f}")


if __name__ == "__main__":
    main()
