"""
Open-world training and evaluation for the Deep Fingerprinting (DF) model.

In the open-world scenario (Section 5.7 of the paper), the adversary must
distinguish monitored sites from a much larger set of unmonitored sites.
The classifier is trained with monitored sites as individual classes and
all unmonitored sites as one additional class.

Evaluation uses threshold-based classification:
  - If the max output probability for any monitored class exceeds a threshold,
    the trace is classified as monitored (and attributed to that class).
  - Otherwise, it is classified as unmonitored.

Metrics: TP, FP, TN, FN, TPR, FPR, Precision, Recall across thresholds.

Usage:
    python train_open_world.py --defense NoDef
    python train_open_world.py --defense WTFPAD --epochs 40
    python train_open_world.py --synthetic
"""

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax

from model import DFNet
from data_utils import (
    load_open_world_training_data,
    load_open_world_evaluation_data,
    preprocess_open_world_data,
    preprocess_evaluation_data,
    generate_synthetic_open_world_data,
)
from evaluate import evaluate_open_world, save_open_world_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate DF model in open-world scenario')
    parser.add_argument('--defense', type=str, default='NoDef',
                        choices=['NoDef', 'WTFPAD', 'WalkieTalkie'],
                        help='Defense type (default: NoDef)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: 30 for NoDef/W-T, '
                             '40 for WTF-PAD)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate (default: 0.002)')
    parser.add_argument('--num_monitored', type=int, default=95,
                        help='Number of monitored site classes (default: 95)')
    parser.add_argument('--input_length', type=int, default=5000,
                        help='Input sequence length (default: 5000)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Training verbosity')
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    if args.epochs is None:
        args.epochs = 40 if args.defense == 'WTFPAD' else 30

    # Number of classes = monitored sites + 1 (unmonitored class)
    num_classes = args.num_monitored + 1
    INPUT_SHAPE = (args.input_length, 1)

    print(f"Open-world DF training: {args.defense} defense")
    print(f"Monitored classes: {args.num_monitored}, "
          f"Total classes (with unmonitored): {num_classes}")
    print(f"Epochs: {args.epochs}")

    # ---- Load Data ----
    if args.synthetic:
        print("Using synthetic data for architecture testing")
        data = generate_synthetic_open_world_data(
            num_monitored=args.num_monitored,
            mon_samples_per_class=100,
            num_unmonitored_train=5000,
            num_unmonitored_test=2000,
            mon_test_per_class=10,
            length=args.input_length)

        X_train, y_train = data['X_train'], data['y_train']
        X_valid, y_valid = data['X_valid'], data['y_valid']
        X_test_Mon, y_test_Mon = data['X_test_Mon'], data['y_test_Mon']
        X_test_Unmon, y_test_Unmon = data['X_test_Unmon'], data['y_test_Unmon']
    else:
        X_train, y_train, X_valid, y_valid = \
            load_open_world_training_data(defense=args.defense)
        X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon = \
            load_open_world_evaluation_data(defense=args.defense)

    # ---- Preprocess Training Data ----
    X_train, y_train, X_valid, y_valid = preprocess_open_world_data(
        X_train, y_train, X_valid, y_valid, num_classes=num_classes)

    # Preprocess evaluation data (reshape only, no one-hot)
    X_test_Mon, X_test_Unmon = preprocess_evaluation_data(
        X_test_Mon, X_test_Unmon)

    print(f"{X_train.shape[0]} train samples")
    print(f"{X_valid.shape[0]} validation samples")

    # ---- Build and Train Model ----
    print("Building and training DF model for open-world scenario")
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)

    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print("Model compiled")

    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=(X_valid, y_valid))

    # ---- Save Model ----
    if args.save_model:
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
                                 f'DF_OpenWorld_{args.defense}.h5')
        model.save(save_path)
        print(f"Model saved to {save_path}")

    # ---- Evaluate ----
    print("\nEvaluating open-world performance across thresholds...")
    pred_Mon = model.predict(X_test_Mon, verbose=0)
    pred_Unmon = model.predict(X_test_Unmon, verbose=0)

    results = evaluate_open_world(
        pred_Mon, y_test_Mon,
        pred_Unmon, y_test_Unmon,
        num_monitored=args.num_monitored)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir,
                                f'OpenWorld_{args.defense}.csv')
    save_open_world_results(results, results_path)
    print(f"Results saved to {results_path}")

    # Print summary at a few key thresholds
    print("\n--- Open-World Results Summary ---")
    print(f"{'Threshold':>10} {'TPR':>8} {'FPR':>8} {'Prec':>8} {'Recall':>8}")
    for row in results:
        if row['threshold'] in [0.5, 0.7, 0.8, 0.9, 0.95]:
            print(f"{row['threshold']:>10.2f} {row['TPR']:>8.4f} "
                  f"{row['FPR']:>8.4f} {row['Precision']:>8.4f} "
                  f"{row['Recall']:>8.4f}")

    return history, model


if __name__ == '__main__':
    main()
