#!/usr/bin/env python3
"""
Apply Tamaraw defense simulation to our cell-level data, then train/eval DFNet.

Uses Yousef's Tamaraw simulator (from commit ea10ee2) to transform our
NoDef cell-level traces into Tamaraw-defended traces, then trains DFNet
to see how accuracy drops.

Usage:
  cd ~/cs244c
  .venv/bin/python scripts/defend_and_eval.py
"""

import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEQ_LEN = 5000
SYNTHETIC_IPT = 0.01


# === Tamaraw simulator (from data-collection/defenses/tamaraw.py) ===

TAMARAW_PARAMS = {
    "rho_out": 0.04,
    "rho_in": 0.012,
    "pad_l": 100,
}


def _anoa(trace, rho_out, rho_in):
    if not trace:
        return []
    outgoing = [t for t, d in trace if d > 0]
    incoming = [t for t, d in trace if d < 0]
    result = []
    if outgoing:
        current_time = outgoing[0]
        pkt_idx = 0
        while pkt_idx < len(outgoing):
            if outgoing[pkt_idx] <= current_time:
                result.append([current_time, 1])
                pkt_idx += 1
            else:
                result.append([current_time, 1])
            current_time += rho_out
    if incoming:
        current_time = incoming[0]
        pkt_idx = 0
        while pkt_idx < len(incoming):
            if incoming[pkt_idx] <= current_time:
                result.append([current_time, -1])
                pkt_idx += 1
            else:
                result.append([current_time, -1])
            current_time += rho_in
    return result


def _anoa_pad(trace, pad_l, rho_out, rho_in):
    if not trace:
        return []
    outgoing = [[t, d] for t, d in trace if d > 0]
    incoming = [[t, d] for t, d in trace if d < 0]
    n_out = len(outgoing)
    target_out = math.ceil(n_out / pad_l) * pad_l
    if outgoing:
        last_time = outgoing[-1][0]
        for _ in range(target_out - n_out):
            last_time += rho_out
            outgoing.append([last_time, 1])
    n_in = len(incoming)
    target_in = math.ceil(n_in / pad_l) * pad_l
    if incoming:
        last_time = incoming[-1][0]
        for _ in range(target_in - n_in):
            last_time += rho_in
            incoming.append([last_time, -1])
    result = outgoing + incoming
    result.sort(key=lambda x: x[0])
    return result


def tamaraw_simulate(trace):
    p = TAMARAW_PARAMS
    rate_controlled = _anoa(trace, p["rho_out"], p["rho_in"])
    return _anoa_pad(rate_controlled, p["pad_l"], p["rho_out"], p["rho_in"])


# === Apply defense to direction sequence ===

def directions_to_trace(directions):
    nonzero = np.nonzero(directions)[0]
    if len(nonzero) == 0:
        return []
    last_idx = nonzero[-1]
    trace = []
    for i in range(last_idx + 1):
        if directions[i] != 0:
            trace.append([i * SYNTHETIC_IPT, float(directions[i])])
    return trace


def defend_sample(directions):
    trace = directions_to_trace(directions)
    if not trace:
        return np.zeros(SEQ_LEN, dtype=np.float32)
    defended = tamaraw_simulate(trace)
    dirs = np.array([d for _, d in defended], dtype=np.float32)
    if len(dirs) >= SEQ_LEN:
        return dirs[:SEQ_LEN]
    padded = np.zeros(SEQ_LEN, dtype=np.float32)
    padded[:len(dirs)] = dirs
    return padded


def defend_split(X):
    X_def = np.zeros_like(X)
    for i in range(len(X)):
        X_def[i] = defend_sample(X[i])
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(X)}", flush=True)
    return X_def


def _lp(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f, encoding="latin1"))


def main():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras.optimizers import Adamax
    from tensorflow.keras.utils import to_categorical
    from model import DFNet

    data_dir = Path("/mnt/d/cs244c-cell-level")

    print("=== Tamaraw Defense Simulation on Self-Collected Data ===\n")

    # Load
    splits = {}
    for s in ["train", "valid", "test"]:
        X = _lp(data_dir / f"X_{s}_Combined.pkl").astype(np.float32)
        y = _lp(data_dir / f"y_{s}_Combined.pkl").astype(np.int64)
        splits[s] = (X, y)
        print(f"Loaded {s}: {X.shape}")

    # Apply Tamaraw
    print("\nApplying Tamaraw defense...")
    t0 = time.time()
    for s in ["train", "valid", "test"]:
        X, y = splits[s]
        print(f"  Defending {s} ({len(X)} samples)...")
        X_def = defend_split(X)
        splits[s] = (X_def, y)
    elapsed = time.time() - t0
    print(f"  Defense simulation done in {elapsed:.0f}s")

    # Quick sanity
    X_tr, y_tr = splits["train"]
    sample = X_tr[:500]
    out_fracs = []
    lens = []
    for x in sample:
        nz = x[x != 0]
        if len(nz) > 0:
            out_fracs.append((nz == 1).sum() / len(nz))
            lens.append(len(nz))
    print(f"\n  Tamaraw trace stats (500 sample):")
    print(f"    Outgoing ratio: {np.mean(out_fracs):.4f}")
    print(f"    Median length: {np.median(lens):.0f}")
    print(f"    Mean length: {np.mean(lens):.0f}")

    # Train
    num_classes = len(np.unique(y_tr))
    print(f"\nTraining DFNet: {num_classes} classes, 30 epochs")

    X_tr = X_tr[:, :, np.newaxis]
    X_val = splits["valid"][0][:, :, np.newaxis]
    X_te = splits["test"][0][:, :, np.newaxis]
    y_tr_oh = to_categorical(y_tr, num_classes)
    y_val_oh = to_categorical(splits["valid"][1], num_classes)
    y_te_oh = to_categorical(splits["test"][1], num_classes)

    model = DFNet.build(input_shape=(SEQ_LEN, 1), classes=num_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adamax(learning_rate=0.002),
        metrics=["accuracy"],
    )

    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr_oh))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
    del X_tr, y_tr_oh, X_val, y_val_oh

    train_ds = train_ds.shuffle(80000, seed=0).batch(128).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(128).prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_ds, epochs=30, verbose=2, validation_data=valid_ds)

    # Eval
    with tf.device("/cpu:0"):
        test_ds = tf.data.Dataset.from_tensor_slices((X_te, y_te_oh))
    test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)
    score = model.evaluate(test_ds, verbose=2)
    print(f"\n=== RESULTS ===")
    print(f"  Tamaraw-defended self-collected data")
    print(f"  Test accuracy: {score[1]:.4f}")
    print(f"  Test loss:     {score[0]:.4f}")
    print(f"\n  Compare: NoDef self-collected = 0.480")
    print(f"  Compare: Tamaraw benchmark    = 0.261")


if __name__ == "__main__":
    main()
