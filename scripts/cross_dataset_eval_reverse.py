#!/usr/bin/env python3
"""
Reverse cross-dataset experiment: train on OUR crawled traces, evaluate on THEIR benchmark test.

Usage:
  cd ~/cs244c
  .venv/bin/python scripts/cross_dataset_eval_reverse.py
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


def parse_args():
    p = argparse.ArgumentParser()
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

    print("=== Reverse cross-dataset experiment ===")
    print(f"  Train on:  {crawled_dir}  (OUR crawled data)")
    print(f"  Eval on:   {benchmark_dir} (THEIR benchmark)")
    print(f"  Classes:   {num_classes} (sites 0-94)")
    print()

    # --- Load crawled train/valid ---
    X_train = _lp(crawled_dir / "X_train_Combined.pkl").astype(np.float32)
    y_train = _lp(crawled_dir / "y_train_Combined.pkl").astype(np.int64)
    X_valid = _lp(crawled_dir / "X_valid_Combined.pkl").astype(np.float32)
    y_valid = _lp(crawled_dir / "y_valid_Combined.pkl").astype(np.int64)
    X_test_crawled = _lp(crawled_dir / "X_test_Combined.pkl").astype(np.float32)
    y_test_crawled = _lp(crawled_dir / "y_test_Combined.pkl").astype(np.int64)
    print(f"Crawled train: {X_train.shape}, valid: {X_valid.shape}, test: {X_test_crawled.shape}")

    # --- Load benchmark test (restrict to classes 0-94) ---
    X_test_bench = _lp(benchmark_dir / "X_test_NoDef.pkl").astype(np.float32)
    y_test_bench = _lp(benchmark_dir / "y_test_NoDef.pkl").astype(np.int64)
    keep = set(range(num_classes))
    mask = np.isin(y_test_bench, list(keep))
    X_test_bench = X_test_bench[mask]
    y_test_bench = y_test_bench[mask]
    print(f"Benchmark test (0-94 only): {X_test_bench.shape}")

    # --- Preprocess ---
    X_train = X_train[:, :, np.newaxis]
    X_valid = X_valid[:, :, np.newaxis]
    X_test_crawled = X_test_crawled[:, :, np.newaxis]
    X_test_bench = X_test_bench[:, :, np.newaxis]

    y_train_oh = to_categorical(y_train, num_classes)
    y_valid_oh = to_categorical(y_valid, num_classes)
    y_test_crawled_oh = to_categorical(y_test_crawled, num_classes)
    y_test_bench_oh = to_categorical(y_test_bench, num_classes)

    INPUT_SHAPE = (5000, 1)

    # --- Build model ---
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)
    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    print("Model compiled\n")

    # --- tf.data ---
    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid_oh))
    del X_train, y_train_oh, X_valid, y_valid_oh

    train_ds = train_ds.shuffle(50000, seed=0).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # --- Train ---
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    history = model.fit(
        train_ds, epochs=args.epochs, verbose=2, validation_data=valid_ds
    )

    # --- Evaluate on crawled test (in-distribution sanity) ---
    with tf.device("/cpu:0"):
        crawled_ds = tf.data.Dataset.from_tensor_slices((X_test_crawled, y_test_crawled_oh))
    crawled_ds = crawled_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    score_crawled = model.evaluate(crawled_ds, verbose=2)
    print(f"\nCrawled test accuracy (in-distribution): {score_crawled[1]:.6f}")

    # --- Evaluate on benchmark test (cross-dataset) ---
    with tf.device("/cpu:0"):
        bench_ds = tf.data.Dataset.from_tensor_slices((X_test_bench, y_test_bench_oh))
    bench_ds = bench_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    score_bench = model.evaluate(bench_ds, verbose=2)
    print(f"Benchmark test accuracy (cross-dataset): {score_bench[1]:.6f}")

    # --- Per-class accuracy on benchmark ---
    y_pred_probs = model.predict(bench_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    for true, pred in zip(y_test_bench, y_pred):
        per_class_total[true] += 1
        if true == pred:
            per_class_correct[true] += 1
    per_class_acc = np.divide(per_class_correct, per_class_total,
                              out=np.zeros(num_classes), where=per_class_total > 0)
    active = per_class_total > 0
    print(f"\nPer-class accuracy (benchmark): mean={per_class_acc[active].mean():.4f}, "
          f"min={per_class_acc[active].min():.4f}, max={per_class_acc[active].max():.4f}")
    best5 = np.argsort(per_class_acc[active])[-5:][::-1]
    active_classes = np.where(active)[0]
    print("Top 5 classes:")
    for idx in best5:
        c = active_classes[idx]
        print(f"  class {c:3d}: {per_class_acc[c]:.4f} ({int(per_class_correct[c])}/{int(per_class_total[c])})")

    # --- Save results ---
    out_dir = REPO / "experiments"
    out_dir.mkdir(exist_ok=True)

    hist_path = out_dir / "cross_dataset_reverse_training_history.csv"
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

    summary_path = out_dir / "cross_dataset_summary.md"
    existing = ""
    if summary_path.exists():
        existing = open(summary_path).read()

    with open(summary_path, "w") as f:
        f.write(existing)
        f.write("\n\n---\n\n")
        f.write("# Reverse Cross-Dataset: Crawled Train → Benchmark Eval\n\n")
        f.write(f"Run: {datetime.now().isoformat()}\n\n")
        f.write("## Setup\n")
        f.write(f"- Train/valid source: `{crawled_dir}` (our crawled traces)\n")
        f.write(f"- Crawled test: same source (in-distribution sanity)\n")
        f.write(f"- Benchmark test: `{benchmark_dir}` (classes 0-94)\n")
        f.write(f"- Classes: {num_classes}\n")
        f.write(f"- Epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}\n\n")
        f.write("## Results\n")
        f.write(f"- **Crawled test accuracy** (in-distribution): **{score_crawled[1]:.4f}**\n")
        f.write(f"- **Benchmark test accuracy** (cross-dataset): **{score_bench[1]:.4f}**\n\n")
        f.write("## Per-class accuracy on benchmark (top 5)\n")
        for idx in best5:
            c = active_classes[idx]
            f.write(f"- class {c}: {per_class_acc[c]:.4f} ({int(per_class_correct[c])}/{int(per_class_total[c])})\n")

    print(f"Summary appended to {summary_path}")

    print("\n=== Done ===")
    print(f"  Crawled test acc (in-dist):    {score_crawled[1]:.4f}")
    print(f"  Benchmark test acc (cross):    {score_bench[1]:.4f}")
    print(f"  Gap:                           {score_crawled[1] - score_bench[1]:.4f}")


if __name__ == "__main__":
    main()
