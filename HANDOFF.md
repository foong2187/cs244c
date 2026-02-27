# Handoff: DF Model Training on 3080

## What this project is

Reimplementation of the Deep Fingerprinting (DF) website fingerprinting attack from:
> Sirinam et al., "Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning," CCS 2018.

The model is a 1D CNN (~4M params) that classifies Tor traffic traces into website classes based on packet direction sequences.

## What's been done

1. **Codebase is complete and tested.** All source files are in `src/`:
   - `model.py` - DF CNN architecture (verified matches paper exactly)
   - `data_utils.py` - data loading, preprocessing, synthetic data generation
   - `preprocess_gtt23.py` - converts GTT23 HDF5 dataset into pickle files
   - `train_closed_world.py` - closed-world training (95-class classification)
   - `train_open_world.py` - open-world training (monitored vs unmonitored)
   - `evaluate.py` - metrics and plotting utilities

2. **Data is preprocessed and ready.** Pickle files are in `dataset/`:
   - `dataset/ClosedWorld/NoDef/` - 76K train, 9.5K valid, 9.5K test (95 classes, 1000 samples/class)
   - `dataset/OpenWorld/NoDef/` - 85K train, 9.5K valid, 9.5K+9K test

3. **First training run completed on GCE VM (CPU-only).** Results in `experiments/01_closed_world_gtt23_baseline.md`:
   - 30 epochs, ~19 hours total
   - Plateaued at ~53% validation accuracy (paper reports 98%)
   - Gap is expected: we trained on GTT23 exit-relay data (passive, noisy, real-world) vs the paper's controlled client-side entry-guard data
   - Signs of overfitting after epoch 15

## What to do next

### 1. Install dependencies and verify GPU
```bash
cd ~/cs244c
pip install -r requirements.txt
# or if CUDA isn't detected:
pip install tensorflow[and-cuda] numpy scikit-learn matplotlib h5py hdf5plugin

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 2. Run closed-world training
```bash
cd src
python train_closed_world.py --defense NoDef
```
Should finish all 30 epochs in under an hour on the 3080.

### 3. Try improvements to beat the 53% baseline
- **More samples per class** - re-preprocess with more data (need GTT23.hdf5 for this, 41GB, still on GCE VM at `miroswisher@34.168.131.160:~/cs244c/data/GTT23.hdf5`):
  ```bash
  python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5 --max_samples 5000 --open_world
  ```
- **Cleaner data** - filter to HTTPS with substantial traces:
  ```bash
  python preprocess_gtt23.py --hdf5 ../data/GTT23.hdf5 --port 443 --min_len 50 --max_samples 5000
  ```
- **More epochs or learning rate tuning** - add `--epochs 60` or `--lr 0.001`
- **Run open-world** after closed-world looks good:
  ```bash
  python train_open_world.py --defense NoDef
  ```

### 4. Save trained model
```bash
python train_closed_world.py --defense NoDef --save_model
# Saves to saved_models/DF_ClosedWorld_NoDef.h5
```

## Key files
```
cs244c/
  src/                    # All source code
  dataset/                # Preprocessed pickle files (ready to train)
  experiments/            # Training results and notes
  data/                   # GTT23 download script (HDF5 not transferred, still on GCE)
  requirements.txt        # Python dependencies
  2018_deep.pdf           # The original paper
  HANDOFF.md              # This file
```

## Important context
- The paper used **client-side entry-guard** data (controlled lab). We used **GTT23 exit-relay** data (passive real-world). This is the main reason for the accuracy gap.
- Hyperparameters match the paper exactly: Adamax lr=0.002, batch 128, 30 epochs (NoDef), kernel size 8, filters [32,64,128,256], FC [512,512], dropout [0.1,0.7,0.5].
- GTT23 direction signs are flipped during preprocessing to match paper convention (+1=outgoing, -1=incoming).
