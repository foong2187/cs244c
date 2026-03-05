"""
Train and evaluate the DF model on Yousef's dataset (open-world).

Uses the same pickles as closed-world; builds open-world splits by treating
the first num_monitored classes as monitored and pooling the rest as unmonitored.

Usage:
    python train_yousef_open_world.py --num_monitored 70
    python train_yousef_open_world.py --num_monitored 90 --epochs 30 --save_model
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax

from model import DFNet
from data_utils import preprocess_open_world_data, preprocess_evaluation_data
from yousef_data_utils import load_yousef_open_world
from evaluate import evaluate_open_world, save_open_world_results

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DF model on Yousef dataset (open-world)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing Yousef pickle files",
    )
    parser.add_argument(
        "--num_monitored",
        type=int,
        default=70,
        help="Number of monitored site classes (default: 70)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate (default: 0.002)")
    parser.add_argument("--input_length", type=int, default=5000, help="Input sequence length")
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Write experiment summary (default: ../experiments)",
    )
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    num_classes = args.num_monitored + 1
    INPUT_SHAPE = (args.input_length, 1)

    print("Open-world DF training on Yousef dataset")
    print(f"Monitored classes: {args.num_monitored}, total classes: {num_classes}")
    print(f"Epochs: {args.epochs}")

    (
        X_train, y_train, X_valid, y_valid,
        X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon,
    ) = load_yousef_open_world(data_dir=args.data_dir, num_monitored=args.num_monitored)

    X_train, y_train, X_valid, y_valid = preprocess_open_world_data(
        X_train, y_train, X_valid, y_valid, num_classes=num_classes
    )
    X_test_Mon, X_test_Unmon = preprocess_evaluation_data(X_test_Mon, X_test_Unmon)

    print("Building and training DF model (open-world)")
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)
    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=(X_valid, y_valid),
    )

    if args.save_model:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "DF_OpenWorld_Yousef.h5")
        model.save(save_path)
        print(f"Model saved to {save_path}")

    print("\nEvaluating open-world performance across thresholds...")
    pred_Mon = model.predict(X_test_Mon, verbose=0)
    pred_Unmon = model.predict(X_test_Unmon, verbose=0)

    results = evaluate_open_world(
        pred_Mon, y_test_Mon,
        pred_Unmon, y_test_Unmon,
        num_monitored=args.num_monitored,
    )

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "OpenWorld_Yousef.csv")
    save_open_world_results(results, results_path)
    print(f"Results saved to {results_path}")

    print("\n--- Open-World Results Summary ---")
    print(f"{'Threshold':>10} {'TPR':>8} {'FPR':>8} {'Prec':>8} {'Recall':>8}")
    for row in results:
        if row["threshold"] in [0.5, 0.7, 0.8, 0.9, 0.95]:
            print(
                f"{row['threshold']:>10.2f} {row['TPR']:>8.4f} {row['FPR']:>8.4f} "
                f"{row['Precision']:>8.4f} {row['Recall']:>8.4f}"
            )

    # Experiment log
    experiment_dir = args.experiment_dir or os.path.join(
        os.path.dirname(__file__), "..", "experiments"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    summary_path = os.path.join(experiment_dir, "04_yousef_open_world.md")
    with open(summary_path, "w") as f:
        f.write("# Yousef Open-World\n\n")
        f.write(f"Run: {datetime.now().isoformat()}\n\n")
        f.write("## Setup\n")
        f.write(f"- num_monitored: {args.num_monitored}, num_classes: {num_classes}\n")
        f.write(f"- Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}\n\n")
        f.write("## Results\n")
        f.write(f"- CSV: {results_path}\n")
        f.write("- Key thresholds:\n")
        for row in results:
            if row["threshold"] in [0.5, 0.7, 0.9]:
                f.write(
                    f"  - {row['threshold']:.2f}: TPR={row['TPR']:.4f}, "
                    f"FPR={row['FPR']:.4f}, Precision={row['Precision']:.4f}, "
                    f"Recall={row['Recall']:.4f}\n"
                )
    print(f"Experiment summary written to {summary_path}")

    return history, model


if __name__ == "__main__":
    main()
