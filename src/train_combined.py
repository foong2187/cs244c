#!/usr/bin/env python3
"""
Train and evaluate the DF model on the combined crawler dataset.

Loads pickles from /mnt/d/cs244c-combined/ (or --data_dir):
    X_train_Combined.pkl, y_train_Combined.pkl, ...

Usage:
  cd /home/mswisher/cs244c && source .venv/bin/activate

  python src/train_combined.py
  python src/train_combined.py --epochs 60 --batch_size 256
  python src/train_combined.py --data_dir /mnt/d/cs244c-combined --top_n 2
"""

import argparse
import os
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.utils import to_categorical

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import DFNet
from evaluate import compute_top_n_accuracy, compute_confusion_matrix

DEFAULT_DATA_DIR = "/mnt/d/cs244c-combined"
DEFAULT_SUFFIX = "Combined"


def load_combined(data_dir, suffix):
    """Load X/y train/valid/test pickles."""
    def _lp(name):
        path = os.path.join(data_dir, name)
        with open(path, "rb") as f:
            return np.array(pickle.load(f))

    X_train = _lp(f"X_train_{suffix}.pkl").astype(np.float32)
    y_train = _lp(f"y_train_{suffix}.pkl").astype(np.int64)
    X_valid = _lp(f"X_valid_{suffix}.pkl").astype(np.float32)
    y_valid = _lp(f"y_valid_{suffix}.pkl").astype(np.int64)
    X_test = _lp(f"X_test_{suffix}.pkl").astype(np.float32)
    y_test = _lp(f"y_test_{suffix}.pkl").astype(np.int64)

    print(f"Loaded from {data_dir}/ (*_{suffix}.pkl)")
    print(f"  Train: {X_train.shape}  Valid: {X_valid.shape}  Test: {X_test.shape}")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def preprocess(X, y, num_classes):
    """Reshape for Conv1D input and one-hot encode labels."""
    X = X[:, :, np.newaxis]  # (n, 5000) -> (n, 5000, 1)
    y = to_categorical(y, num_classes)
    return X, y


def parse_args():
    p = argparse.ArgumentParser(description="Train DF on combined crawler dataset")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--patience", type=int, default=7,
                    help="EarlyStopping patience on val_loss")
    p.add_argument("--input_length", type=int, default=5000)
    p.add_argument("--top_n", type=int, default=None,
                    help="Also compute top-N accuracy")
    p.add_argument("--no_save", action="store_true",
                    help="Don't save model checkpoint")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # ---- Load data ----
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_combined(
        args.data_dir, args.suffix
    )
    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max())) + 1
    print(f"Classes: {num_classes}")

    # ---- Preprocess ----
    INPUT_SHAPE = (args.input_length, 1)
    X_train, y_train_oh = preprocess(X_train, y_train, num_classes)
    X_valid, y_valid_oh = preprocess(X_valid, y_valid, num_classes)
    X_test, y_test_oh = preprocess(X_test, y_test, num_classes)

    print(f"{len(X_train)} train / {len(X_valid)} valid / {len(X_test)} test")

    # ---- Build model ----
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)
    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.summary()

    # ---- tf.data pipeline ----
    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid_oh))
    del X_train, y_train_oh, X_valid, y_valid_oh

    train_ds = (
        train_ds.shuffle(60000, seed=0)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = valid_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # ---- Callbacks ----
    save_dir = os.path.join(SRC_DIR, "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "DF_Combined_best.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-5, verbose=1,
        ),
    ]
    if not args.no_save:
        callbacks.append(
            ModelCheckpoint(
                best_model_path, monitor="val_accuracy",
                save_best_only=True, verbose=1,
            )
        )

    # ---- Train ----
    print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience}")

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=valid_ds,
        callbacks=callbacks,
    )

    # ---- Evaluate ----
    with tf.device("/cpu:0"):
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
    test_ds_batched = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    score = model.evaluate(test_ds_batched, verbose=args.verbose)
    test_loss, test_acc = score[0], score[1]
    print(f"\nTest accuracy: {test_acc:.6f}")
    print(f"Test loss:     {test_loss:.6f}")

    # Top-N accuracy
    top_n_acc = None
    if args.top_n and args.top_n > 1:
        y_pred_probs = model.predict(test_ds_batched, verbose=0)
        top_n_acc = compute_top_n_accuracy(y_test, y_pred_probs, n=args.top_n)
        print(f"Top-{args.top_n} accuracy: {top_n_acc:.6f}")

    # Per-class accuracy summary
    y_pred_probs = model.predict(test_ds_batched, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = compute_confusion_matrix(y_test, y_pred, num_classes)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print(f"\nPer-class accuracy: mean={per_class_acc.mean():.4f}, "
          f"min={per_class_acc.min():.4f}, max={per_class_acc.max():.4f}")
    worst_5 = np.argsort(per_class_acc)[:5]
    for c in worst_5:
        total = cm[c].sum()
        print(f"  class {c:3d}: {per_class_acc[c]:.4f} ({cm[c, c]}/{total})")

    # ---- Experiment summary ----
    exp_dir = os.path.join(SRC_DIR, "..", "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    summary_path = os.path.join(exp_dir, "combined_closed_world.md")

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    total_epochs = len(history.history["accuracy"])

    with open(summary_path, "w") as f:
        f.write("# Combined Crawler Dataset — Closed-World\n\n")
        f.write(f"Run: {datetime.now().isoformat()}\n\n")
        f.write("## Dataset\n")
        f.write(f"- Source: `{args.data_dir}` (suffix: {args.suffix})\n")
        f.write(f"- Classes: {num_classes}\n")
        f.write(f"- Samples: train={len(y_test)*8}, valid={len(y_test)}, "
                f"test={len(y_test)}\n")
        f.write(f"- Input length: {args.input_length}\n\n")
        f.write("## Hyperparameters\n")
        f.write(f"- Optimizer: Adamax (lr={args.lr})\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Max epochs: {args.epochs}, ran {total_epochs}\n")
        f.write(f"- EarlyStopping patience: {args.patience}\n")
        f.write(f"- ReduceLROnPlateau: factor=0.5, patience=3\n\n")
        f.write("## Results\n")
        f.write(f"- Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"- Test accuracy: {test_acc:.4f}\n")
        f.write(f"- Test loss: {test_loss:.4f}\n")
        if top_n_acc is not None:
            f.write(f"- Top-{args.top_n} accuracy: {top_n_acc:.4f}\n")
        f.write(f"- Per-class accuracy: mean={per_class_acc.mean():.4f}, "
                f"min={per_class_acc.min():.4f}\n")
        f.write(f"\n## Model\n")
        f.write(f"- Architecture: DFNet (Sirinam et al. CCS'18)\n")
        if not args.no_save:
            f.write(f"- Best checkpoint: `{best_model_path}`\n")

    print(f"\nExperiment summary: {summary_path}")
    if not args.no_save:
        print(f"Best model checkpoint: {best_model_path}")

    return history, model


if __name__ == "__main__":
    main()
