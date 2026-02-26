"""
Closed-world training and evaluation for the Deep Fingerprinting (DF) model.

Supports three dataset types:
  - NoDef:         Non-defended Tor traffic (95 classes, 30 epochs)
  - WTFPAD:        WTF-PAD defended traffic  (95 classes, 40 epochs)
  - WalkieTalkie:  Walkie-Talkie defended traffic (100 classes, 30 epochs)

Usage:
    python train_closed_world.py --defense NoDef
    python train_closed_world.py --defense WTFPAD --epochs 40
    python train_closed_world.py --defense WalkieTalkie --top_n 2
    python train_closed_world.py --synthetic   # Use synthetic data for testing
"""

import argparse
import os
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax

from model import DFNet
from data_utils import (
    load_closed_world_data,
    preprocess_data,
    generate_synthetic_data,
)
from evaluate import compute_top_n_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate DF model in closed-world scenario')
    parser.add_argument('--defense', type=str, default='NoDef',
                        choices=['NoDef', 'WTFPAD', 'WalkieTalkie'],
                        help='Defense type (default: NoDef)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs '
                             '(default: 30 for NoDef/W-T, 40 for WTF-PAD)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate for Adamax (default: 0.002)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detected if not set)')
    parser.add_argument('--input_length', type=int, default=5000,
                        help='Input sequence length (default: 5000)')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Also compute top-N accuracy (e.g., 2 for W-T)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing architecture')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Training verbosity (0, 1, or 2)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Default epochs per defense
    if args.epochs is None:
        args.epochs = 40 if args.defense == 'WTFPAD' else 30

    print(f"Training and evaluating DF model for closed-world scenario "
          f"on {args.defense} dataset")
    print(f"Number of Epochs: {args.epochs}")

    # ---- Load Data ----
    if args.synthetic:
        num_classes = args.num_classes or 95
        print("Using synthetic data for architecture testing")
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            generate_synthetic_data(num_classes=num_classes,
                                    samples_per_class=100,
                                    length=args.input_length)
    else:
        print("Loading and preparing data for training and evaluation")
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            load_closed_world_data(defense=args.defense)
        num_classes = args.num_classes or int(np.max(y_train) + 1)

    # ---- Preprocess ----
    INPUT_SHAPE = (args.input_length, 1)
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        num_classes=num_classes)

    # ---- Build Model ----
    print("Building and training DF model")
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)

    optimizer = Adamax(learning_rate=args.lr, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print("Model compiled")

    # ---- Train ----
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=(X_valid, y_valid))

    # ---- Evaluate ----
    score_test = model.evaluate(X_test, y_test, verbose=args.verbose)
    print(f"Testing accuracy: {score_test[1]:.6f}")

    # Top-N accuracy (useful for Walkie-Talkie evaluation)
    if args.top_n is not None and args.top_n > 1:
        print(f"\nEvaluating Top-{args.top_n} Accuracy")
        y_pred_probs = model.predict(X_test, verbose=0)
        y_true = np.argmax(y_test, axis=1)
        top_n_acc = compute_top_n_accuracy(y_true, y_pred_probs, n=args.top_n)
        print(f"Top-{args.top_n} Accuracy: {top_n_acc:.6f}")

    # ---- Save Model ----
    if args.save_model:
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'DF_ClosedWorld_{args.defense}.h5')
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return history, model


if __name__ == '__main__':
    main()
