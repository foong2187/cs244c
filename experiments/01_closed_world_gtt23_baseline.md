# Experiment 1: Closed-World DF Training on GTT23 (Baseline)

**Date:** February 26, 2026
**Duration:** ~19 hours (30 epochs on CPU-only GCE VM)

## Objective

Reproduce the closed-world website fingerprinting results from Sirinam et al. (CCS 2018) using the Deep Fingerprinting (DF) model architecture, trained on the GTT23 dataset instead of the original paper's dataset.

## Model Architecture

Faithful reproduction of the DF architecture from the paper (Table 1, Appendix A):

- **Input:** 5000-length sequences of packet directions (+1 outgoing, -1 incoming, 0 padding)
- **Convolutional blocks:** 4 blocks with filter counts [32, 64, 128, 256], kernel size 8, pool size 8, pool stride 4
- **Activations:** ELU in Block 1 (to preserve negative direction values), ReLU in Blocks 2-4
- **Regularization:** Batch Normalization after each conv/FC layer; Dropout 0.1 after each pool, 0.7 and 0.5 on the two FC layers
- **Classification head:** Two FC layers (512 units each), softmax output over 95 classes
- **Total parameters:** ~3.98M

## Training Configuration

| Parameter        | Value                      |
|------------------|----------------------------|
| Optimizer        | Adamax (lr=0.002, β1=0.9, β2=0.999, ε=1e-8) |
| Loss             | Categorical cross-entropy  |
| Batch size       | 128                        |
| Epochs           | 30                         |
| Classes          | 95                         |
| Train samples    | 76,000 (800/class)         |
| Valid samples    | 9,500 (100/class)          |
| Test samples     | 9,500 (100/class)          |
| Hardware         | GCE VM, 4 vCPUs, 16 GB RAM, no GPU |
| Time per epoch   | ~38 minutes                |

All hyperparameters match the paper exactly.

## Dataset: GTT23 vs. Original Paper Data

### What we used: GTT23

The **GTT23 dataset** (Genuine Tor Traces 2023, DOI: [10.5281/zenodo.10620520](https://doi.org/10.5281/zenodo.10620520)) contains network metadata from 13.9 million Tor circuits across 1.14 million unique domains, collected passively from **Tor exit relays** over a 13-week period in 2023. Each circuit record includes a sequence of 5000 cell metadata tuples containing timestamps, direction, and cell/relay commands.

We preprocessed GTT23 as follows:
1. Selected the 95 domains (hashed labels) with the highest circuit counts.
2. Randomly sampled 1,000 circuits per domain.
3. Extracted the cell direction field from each circuit's 5000-cell sequence.
4. Flipped direction signs to match the paper's convention (GTT23: +1 = toward client; paper: +1 = outgoing from client).
5. Split 80/10/10 into train/validation/test sets.

### What the paper used

The original paper used datasets collected in a **controlled lab setting** where specific websites were repeatedly visited through Tor, and traffic was captured at the **client's entry guard** (the link between the Tor client and the first relay). Each website was visited hundreds of times under controlled conditions, producing clean, consistent traffic traces. The data was collected specifically for website fingerprinting research.

### Key differences

| Aspect | Paper's Data | GTT23 (Our Data) |
|--------|-------------|-------------------|
| **Observation point** | Entry guard (client side) | Exit relay (server side) |
| **Collection method** | Active, controlled visits | Passive, real-world measurement |
| **Traffic cleanliness** | Single website per circuit, clean traces | Real-world noise, possible multiplexing |
| **Label quality** | Known plaintext domains | Hashed domain identifiers |
| **Time period** | Pre-2018 | 2023 |
| **Tor version** | Older | Newer (potentially different cell scheduling) |

The most significant difference is the **observation point**. Website fingerprinting attacks are designed to be mounted by an adversary observing the encrypted link between a Tor client and its entry guard. The cell direction patterns at the entry guard directly reflect the client's webpage loading behavior (request bursts outgoing, response data incoming). At the exit relay, the relationship between cell patterns and individual webpage loads is less direct because:

1. **Different traffic perspective:** The exit relay sees decrypted traffic toward the destination server, not the encrypted cell patterns that an entry-guard observer would see.
2. **Circuit multiplexing:** Real-world Tor usage may multiplex multiple streams on a single circuit at the exit relay, diluting per-website fingerprint signals.
3. **Behavioral noise:** Passive measurement captures organic browsing behavior (partial loads, tabbed browsing, background requests) rather than clean single-page visits.

## Results

### Training Curve (All 29 Completed Epochs)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 7.03%     | 4.0093     | 11.62%  | 3.5966   |
| 2     | 20.49%    | 3.0224     | 31.37%  | 2.5143   |
| 3     | 31.17%    | 2.5070     | 38.59%  | 2.1759   |
| 4     | 37.64%    | 2.2272     | 43.84%  | 1.9618   |
| 5     | 41.30%    | 2.0587     | 45.48%  | 1.8769   |
| 6     | 44.07%    | 1.9461     | 46.66%  | 1.8151   |
| 7     | 45.87%    | 1.8619     | 48.26%  | 1.7777   |
| 8     | 47.59%    | 1.7985     | 49.15%  | 1.7554   |
| 9     | 48.51%    | 1.7528     | 48.80%  | 1.7481   |
| 10    | —         | —          | —       | —        |
| 11    | —         | —          | —       | —        |
| 12    | —         | —          | —       | —        |
| 13    | —         | —          | —       | —        |
| 14    | 52.26%    | 1.5866     | 51.82%  | 1.6197   |
| 15    | 52.70%    | 1.5670     | 51.48%  | 1.6435   |
| 16    | 53.39%    | 1.5449     | 51.55%  | 1.6271   |
| 17    | 53.84%    | 1.5257     | 52.15%  | 1.6299   |
| 18    | 54.35%    | 1.5072     | 52.63%  | 1.6076   |
| 19    | 54.48%    | 1.4985     | 52.56%  | 1.6059   |
| 20    | 54.76%    | 1.4788     | 52.94%  | 1.5792   |
| 21    | 55.16%    | 1.4646     | 53.02%  | 1.5799   |
| 22    | 55.58%    | 1.4533     | 52.31%  | 1.5974   |
| 23    | 55.84%    | 1.4372     | 52.66%  | 1.6334   |
| 24    | 56.28%    | 1.4237     | 52.29%  | 1.6189   |
| 25    | 56.51%    | 1.4158     | 52.99%  | 1.5740   |
| 26    | 56.73%    | 1.4036     | 52.17%  | 1.6182   |
| 27    | 56.98%    | 1.3958     | 52.99%  | 1.5978   |
| 28    | 57.33%    | 1.3838     | 53.15%  | 1.5948   |
| 29    | 57.35%    | 1.3790     | 53.32%  | 1.5759   |

*Note: Epochs 10-13 were not captured in terminal logs due to SSH reconnection. Epoch 30 was in progress at time of writing.*

### Summary

- **Best validation accuracy:** ~53.3% (epoch 29)
- **Paper's reported accuracy:** ~98% (closed-world, NoDef)
- **Gap:** ~45 percentage points

### Signs of overfitting

Starting around epoch 15, training accuracy continued climbing (52.7% → 57.4%) while validation accuracy plateaued around 52-53% and validation loss began oscillating. This train-val divergence indicates the model is beginning to overfit to the training data rather than learning more generalizable patterns.

## Analysis

The ~53% validation accuracy across 95 classes is well above the ~1.05% random baseline, demonstrating that the DF architecture does extract meaningful fingerprinting features from GTT23 exit-relay data. However, the large gap from the paper's 98% is expected and attributable to the fundamental dataset differences described above.

The model learned rapidly in the first few epochs (reaching ~49% by epoch 8), then entered a slow plateau phase where gains were marginal (~1% per 5 epochs). This saturation pattern, combined with early signs of overfitting, suggests the model has extracted most of the learnable signal from this data representation.

## Possible Improvements for Subsequent Experiments

1. **More samples per class:** Increase from 1,000 to 5,000-10,000 to give the model more examples of the noisier real-world traces.
2. **Data filtering:** Restrict to HTTPS circuits (`--port 443`) with substantial cell sequences (`--min_len 50`) to reduce noise.
3. **Learning rate scheduling:** Add decay or reduce-on-plateau to escape the validation loss oscillation.
4. **Early stopping:** Monitor validation loss and stop when it stops improving.
5. **Entry-guard data:** To more faithfully reproduce the paper's results, collect or obtain client-side traffic traces rather than exit-relay data.
6. **Data augmentation:** Apply trace-level augmentations (random truncation, noise injection) to improve generalization on noisy data.
