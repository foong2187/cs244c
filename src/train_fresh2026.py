"""
Train and evaluate the DF model on the Fresh2026 dataset provided by a teammate.

The Fresh2026 pickles live under:
    data/devin-data/data/pickle/
        X_train_Fresh2026.pkl
        y_train_Fresh2026.pkl
        X_valid_Fresh2026.pkl
        y_valid_Fresh2026.pkl
        X_test_Fresh2026.pkl
        y_test_Fresh2026.pkl

The feature format matches the DF model:
    - X: (n, 5000) direction sequences (+1, -1, 0)
    - y: (n,) integer labels
"""

import os
import pickle
import random

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.optimizers import Adamax

from model import DFNet
from data_utils import preprocess_data


def load_fresh2026():
    base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "devin-data",
        "data",
        "pickle",
    )

    def _lp(name):
        path = os.path.join(base, name)
        with open(path, "rb") as f:
            return np.array(pickle.load(f))

    X_train = _lp("X_train_Fresh2026.pkl")
    y_train = _lp("y_train_Fresh2026.pkl")
    X_valid = _lp("X_valid_Fresh2026.pkl")
    y_valid = _lp("y_valid_Fresh2026.pkl")
    X_test = _lp("X_test_Fresh2026.pkl")
    y_test = _lp("y_test_Fresh2026.pkl")

    print("Loaded Fresh2026 splits:")
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_valid {X_valid.shape}, y_valid {y_valid.shape}")
    print(f"  X_test  {X_test.shape}, y_test  {y_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main():
    # Hyperparameters aligned with DF closed-world NoDef
    epochs = 30
    batch_size = 64
    lr = 0.002
    input_length = 5000

    # Seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    print("Training DF model on Fresh2026 dataset")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fresh2026()
    num_classes = int(np.max(y_train) + 1)
    print(f"Detected {num_classes} classes")

    # Preprocess to match DFNet input
    INPUT_SHAPE = (input_length, 1)
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes=num_classes
    )

    # Build model
    print("Building DFNet model")
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=num_classes)
    optimizer = Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.summary()

    # Train (dataset is small enough to use in-memory arrays)
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(X_valid, y_valid),
    )

    # Evaluate
    score_test = model.evaluate(X_test, y_test, verbose=2)
    print(f"Fresh2026 test accuracy: {score_test[1]:.6f}")

    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "DF_ClosedWorld_Fresh2026.h5")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return history, model


if __name__ == "__main__":
    main()

