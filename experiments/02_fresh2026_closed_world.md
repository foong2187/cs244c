# Experiment 02 – DF on Fresh2026 (Closed World)

## Setup

- **Model**: DFNet (same architecture as paper; 4 conv blocks + 2 FC layers, ~4M params)
- **Data**: Fresh2026 teammate dataset
  - `X_train_Fresh2026.pkl`: (5808, 5000)
  - `X_valid_Fresh2026.pkl`: (726, 5000)
  - `X_test_Fresh2026.pkl`: (727, 5000)
  - Direction-only traces (+1 / -1 / 0), already length-5000
  - Detected **92 classes**
- **Preprocessing**: `preprocess_data` (float32, reshape to (n, 5000, 1), one-hot labels)
- **Hyperparameters**:
  - Epochs: 30
  - Batch size: 64
  - Optimizer: Adamax (lr=0.002, β1=0.9, β2=0.999)
  - Loss: categorical cross-entropy
  - Metrics: accuracy
- **Hardware**: RTX 3080 under WSL2 with `tensorflow[and-cuda]` 2.20.0

Command:

```bash
cd ~/cs244c/src
python train_fresh2026.py
```

## Results

- **Validation accuracy**:
  - Epoch 1: 0.0069
  - Epoch 10: 0.1116
  - Epoch 20: 0.1818
  - Epoch 25: 0.2314
  - Epoch 30: 0.1680 (some overfitting / instability)
- **Test accuracy**:
  - Final: **0.1637** (16.37%)

Training time:
- ~33s for epoch 1 (XLA autotuning / kernel compilation)
- ~1s per epoch for epochs 2–30 (small dataset)
- Total wall time ~1 minute.

## Discussion

1. **Performance is low but non-trivial.**  
   - Random guess baseline for 92-way classification ≈ 1.1% accuracy.  
   - DF reaches ~16% test accuracy, ~23% peak validation accuracy — clearly learning structure, but far from a practical classifier.

2. **Data regime is very different from GTT23 NoDef.**  
   - GTT23 closed-world (exit-relay data, 95 classes, 1000 samples/class) achieved ~52% test accuracy.  
   - Fresh2026 has only 5808 training traces over 92 classes (~63 samples/class on average), which is a **low-data regime** for a 4M-parameter CNN.
   - With so few examples per class, the model is likely undertrained and heavily regularized by dropout, limiting achievable accuracy.

3. **Label/feature mismatch may dominate.**  
   - The DF architecture and hyperparameters are tuned for Tor exit or entry data with specific collection and labeling conventions.  
   - If Fresh2026 labels group heterogeneous pages/services into single classes, intra-class variance may be high and inter-class separation weak, capping accuracy even with more training.

4. **Overfitting behavior suggests capacity is not the main bottleneck.**  
   - Training accuracy climbs steadily (to ~27%), while validation accuracy plateaus and then fluctuates, indicating the model is fitting the small training set but not generalizing strongly.
   - Reducing model size or increasing regularization would not necessarily fix the fundamental data scarcity and class ambiguity.

5. **Implications for your project.**  
   - DF on Fresh2026 is **not yet a strong classifier**, but it provides a concrete, reproducible baseline against the exact data shape your group is collecting (5000-length directions, 92 classes).  
   - Future improvements are more likely to come from:
     - Better-curated classes (cleaner labels, more homogeneous sites per class).  
     - More traces per class (data collection), or data augmentation.  
     - Incorporating richer features (timing, packet sizes, bursts) into the model, rather than just direction-only sequences.

