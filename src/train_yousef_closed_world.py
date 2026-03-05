"""
Train and evaluate the DF model on Yousef's dataset (closed-world).

Uses pickles from data/yousef-data/pickle/ (or --data_dir). Format matches
the DF paper: X (n, 5000) direction sequences, y (n,) integer labels.

Usage:
    python train_yousef_closed_world.py
    python train_yousef_closed_world.py --epochs 60 --save_model --top_n 2
    python train_yousef_closed_world.py --data_dir /path/to/yousef/pickle
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.optimizers import Adamax

from model import DFNet
from data_utils import preprocess_data
from yousef_data_utils import load_yousef_closed_world
from evaluate import compute_top_n_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DF model on Yousef's dataset (closed-world)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing Yousef pickle files (default: ../data/yousef-data/pickle)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.002, help="Adamax learning rate (default: 0.002)")
    parser.add_argument(
        "--input_length",
        type=int,
        default=5000,
        help="Input sequence length (default: 5000)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Also compute top-N accuracy (e.g., 2)",
    )
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Write experiment summary to this dir (default: ../experiments)",
    )
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity (0, 1, 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    print("Training DF model on Yousef dataset (closed-world)")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_yousef_closed_world(
        data_dir=args.data_dir
    )
    num_classes = int(np.max(y_train)) + 1
    print(f"Detected {num_classes} classes")

    # Preprocess for DFNet
    INPUT_SHAPE = (args.input_length, 1)
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes=num_classes
    )

    # Build model
    print("Building DFNet model")
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)
    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.summary()

    # Train with tf.data
    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    del X_train, y_train, X_valid, y_valid

    train_ds = (
        train_ds.shuffle(50000, seed=0)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = valid_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=valid_ds,
    )

    # Evaluate
    with tf.device("/cpu:0"):
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    score_test = model.evaluate(test_ds, verbose=args.verbose)
    test_acc = score_test[1]
    test_loss = score_test[0]
    print(f"Test accuracy: {test_acc:.6f}, Test loss: {test_loss:.6f}")

    top_n_acc = None
    if args.top_n is not None and args.top_n > 1:
        y_pred_probs = model.predict(X_test, verbose=0)
        y_true = np.argmax(y_test, axis=1)
        top_n_acc = compute_top_n_accuracy(y_true, y_pred_probs, n=args.top_n)
        print(f"Top-{args.top_n} accuracy: {top_n_acc:.6f}")

    # Save model
    if args.save_model:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "DF_ClosedWorld_Yousef.h5")
        model.save(save_path)
        print(f"Model saved to {save_path}")

    # Write experiment summary
    experiment_dir = args.experiment_dir or os.path.join(
        os.path.dirname(__file__), "..", "experiments"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    summary_path = os.path.join(experiment_dir, "03_yousef_closed_world.md")
    val_acc = history.history["val_accuracy"][-1] if history.history["val_accuracy"] else None
    with open(summary_path, "w") as f:
        f.write("# Yousef Closed-World Baseline\n\n")
        f.write(f"Run: {datetime.now().isoformat()}\n\n")
        f.write("## Dataset\n")
        f.write("- Source: Yousef pickles (data/yousef-data/pickle or --data_dir)\n")
        f.write(f"- Classes: {num_classes}\n")
        f.write("- Input length: 5000\n\n")
        f.write("## Hyperparameters\n")
        f.write(f"- Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}\n")
        f.write("- Optimizer: Adamax (paper default)\n\n")
        f.write("## Results\n")
        if val_acc is not None:
            f.write(f"- Validation accuracy (final): {val_acc:.4f}\n")
        f.write(f"- Test accuracy: {test_acc:.4f}\n")
        f.write(f"- Test loss: {test_loss:.4f}\n")
        if top_n_acc is not None:
            f.write(f"- Top-{args.top_n} accuracy: {top_n_acc:.4f}\n")
        f.write("\n## Comparison\n")
        f.write("- GTT23 baseline (01): ~53% val accuracy (exit-relay data).\n")
        f.write("- Fresh2026 (02): ~16% test accuracy (92 classes).\n")
        f.write("- CCS'18 paper: ~98% on client-side entry-guard data.\n")
    print(f"Experiment summary written to {summary_path}")

    return history, model


if __name__ == "__main__":
    main()
