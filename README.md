# Deep Fingerprinting (DF) - Website Fingerprinting with Deep Learning

Reimplementation of the Deep Fingerprinting attack from:

> Sirinam, P., Imani, M., Juarez, M., & Wright, M. (2018).
> **Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.**
> In CCS '18. https://doi.org/10.1145/3243734.3243768

## Overview

The DF model is a 1D Convolutional Neural Network (CNN) designed for website fingerprinting attacks against Tor. It takes sequences of packet directions (+1 outgoing, -1 incoming) as input and classifies them into website classes.

Key architecture features:
- 4 convolutional blocks with increasing filter counts (32, 64, 128, 256)
- ELU activation in Block 1 (handles negative input values), ReLU in Blocks 2-4
- Batch Normalization and Dropout for regularization
- 2 fully-connected layers (512 units each) for classification
- Softmax output layer

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and TensorFlow 2.10+.

## Dataset

### GTT23 (recommended)

**[GTT23: A 2023 Dataset of Genuine Tor Traces](https://zenodo.org/records/10620520)** on Zenodo (DOI: [10.5281/zenodo.10620520](https://doi.org/10.5281/zenodo.10620520)) provides network metadata of encrypted traffic from Tor exit relays, suitable for website fingerprinting. Access may require signing in to Zenodo and accepting the record’s terms; download the `GTT23.hdf5` file and `README.md` from the record page.

#### Downloading

```bash
cd data
bash download.sh
```

The HDF5 file is ~44 GB and may require multiple retries (the script handles resume).

#### Preprocessing GTT23 into pickle files

Use the included preprocessing script to convert the HDF5 data into the pickle format expected by the training scripts:

```bash
cd src

# Closed-world only (95 classes, default)
python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5

# Closed-world + open-world splits
python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5 --open_world

# Filter to HTTPS traffic, require at least 200 samples per class
python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5 --port 443 --min_samples 200
```

The script selects labels with the most circuits, extracts cell direction sequences, flips the sign convention to match the paper (+1=outgoing, -1=incoming), and splits into train/valid/test sets.

System HDF5 dependencies: `apt install libhdf5-mpi-dev h5utils hdf5-helpers hdf5-tools`

### Original paper dataset (often unavailable)

The CCS’18 paper’s authors originally hosted closed-world and open-world data on Google Drive; those links are frequently broken. For reference, see the [deep-fingerprinting/df](https://github.com/deep-fingerprinting/df) repo. You can run and test **without** any real dataset using synthetic data (see “Testing Without Datasets” below).

### Expected layout and format (for our scripts)

After you have data in the right form, place the pickle files as follows:

```
dataset/
  ClosedWorld/
    NoDef/          # X_train_NoDef.pkl, y_train_NoDef.pkl, etc.
    WTFPAD/         # X_train_WTFPAD.pkl, y_train_WTFPAD.pkl, etc.
    WalkieTalkie/   # X_train_WalkieTalkie.pkl, etc.
  OpenWorld/
    NoDef/          # Training + Mon/Unmon test splits
    WTFPAD/
    WalkieTalkie/
```

Each pickle file should contain:
- `X_*.pkl`: Arrays of shape `(n, 5000)` with packet direction sequences
- `y_*.pkl`: Arrays of shape `(n,)` with website class labels

## Usage

### Closed-World Evaluation

Train and evaluate on non-defended Tor traffic (95 sites, 30 epochs):

```bash
cd src
python train_closed_world.py --defense NoDef
```

Train on WTF-PAD defended traffic (40 epochs):

```bash
python train_closed_world.py --defense WTFPAD --epochs 40
```

Train on Walkie-Talkie traffic with top-2 accuracy:

```bash
python train_closed_world.py --defense WalkieTalkie --top_n 2
```

### Open-World Evaluation

```bash
python train_open_world.py --defense NoDef
python train_open_world.py --defense WTFPAD --epochs 40
```

### Testing Without Datasets

Use synthetic data to verify the architecture works:

```bash
python train_closed_world.py --synthetic --epochs 5
python train_open_world.py --synthetic --epochs 5
```

### Model Summary

View the model architecture:

```bash
python model.py
```

## Hyperparameters (Table 1 from paper)

| Parameter | Value |
|-----------|-------|
| Input Dimension | 5000 |
| Optimizer | Adamax |
| Learning Rate | 0.002 |
| Epochs (NoDef/W-T) | 30 |
| Epochs (WTF-PAD) | 40 |
| Batch Size | 128 |
| Filter Sizes | [32, 64, 128, 256] |
| Kernel Size | 8 |
| Pool Size | 8 |
| Pool Stride | 4 |
| FC Hidden Units | [512, 512] |
| Dropout (pools) | 0.1 |
| Dropout (FC1, FC2) | 0.7, 0.5 |

## Project Structure

```
src/
  model.py               # DFNet CNN architecture
  data_utils.py           # Data loading, preprocessing, synthetic generation
  preprocess_gtt23.py     # Convert GTT23 HDF5 -> pickle files for training
  train_closed_world.py   # Closed-world training and evaluation
  train_open_world.py     # Open-world training and evaluation
  evaluate.py             # Evaluation metrics and plotting utilities
data/
  download.sh             # Download GTT23 from Zenodo
dataset/                  # Preprocessed pickle files (generated by preprocess_gtt23.py)
saved_models/             # Trained models saved here
results/                  # Evaluation results (CSV)
```

## Reference

```bibtex
@inproceedings{sirinam2018deep,
  title={Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning},
  author={Sirinam, Payap and Imani, Mohsen and Juarez, Marc and Wright, Matthew},
  booktitle={Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
  pages={1928--1943},
  year={2018},
  organization={ACM}
}
```
