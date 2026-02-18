# Deep Fingerprinting: Replication & Extension Against Modern Defenses

## Context

This project replicates the Deep Fingerprinting (DF) attack from Sirinam et al. (CCS 2018) and extends it by evaluating against two modern defenses — **RegulaTor** (PETS 2022) and **BRO** (zero-delay beta-distribution padding) — that did not exist when the original paper was published. The goal is to determine whether the 2018 CNN architecture remains viable against state-of-the-art traffic shaping.

**Three data sources**:
- **Original DF dataset** (from [github.com/deep-fingerprinting/df](https://github.com/deep-fingerprinting/df)) — pickle files of direction sequences for faithful replication
- **GTT23** (Zenodo, 43.9 GB HDF5) — genuine 2023 Tor traces with timestamps, used for extension experiments with real defense simulations
- **Self-collected traces** — fresh 2026 Tor traffic for concept drift analysis and custom experiments

---

## Project Structure

```
df-website-fingerprinting/
├── configs/
│   ├── closed_world_nodef.yaml
│   ├── closed_world_regulator.yaml
│   ├── closed_world_bro.yaml
│   ├── open_world_nodef.yaml
│   ├── open_world_regulator.yaml
│   └── open_world_bro.yaml
├── data/
│   ├── raw/                          # Original DF pickle files
│   │   ├── ClosedWorld/NoDef/
│   │   └── OpenWorld/NoDef/
│   ├── gtt23/                        # GTT23 HDF5 file
│   ├── collected/                    # Self-collected pcap + trace files
│   ├── traces/                       # Extracted timestamp+direction traces from GTT23
│   └── defended/                     # Defense-applied traces (pickle format)
│       ├── RegulaTor/
│       └── BRO/
├── src/
│   ├── data/
│   │   ├── dataset.py                # PyTorch Dataset (TraceDataset)
│   │   ├── loader.py                 # DataLoader factory functions
│   │   ├── preprocessing.py          # Pickle loading (Python 2 compat), pad/truncate
│   │   └── augmentation.py           # NetAugment: burst merge/split/insert, packet drop
│   ├── models/
│   │   └── df_net.py                 # DF CNN architecture (PyTorch)
│   ├── defenses/
│   │   ├── regulator.py              # RegulaTor simulation wrapper
│   │   ├── bro.py                    # BRO simulation wrapper
│   │   └── gtt23_extract.py          # Extract traces from GTT23 HDF5
│   ├── training/
│   │   └── trainer.py                # Training loop with checkpointing
│   ├── evaluation/
│   │   ├── closed_world.py           # Accuracy, confusion matrix, F1
│   │   ├── open_world.py             # Threshold sweep, TPR/FPR, PR curves
│   │   ├── metrics.py                # Shared metric utilities
│   │   └── adversarial.py            # FGSM/PGD evasion attacks on DF model
│   ├── visualization/
│   │   ├── plots.py                  # Training curves, confusion matrix, PR/ROC, bar charts
│   │   └── gradcam.py                # GradCAM saliency maps for 1D CNN
│   └── utils/
│       ├── config.py                 # YAML config loader
│       ├── device.py                 # CUDA / MPS / CPU selection
│       └── reproducibility.py        # Seed setting
├── scripts/
│   ├── download_df_data.py           # Download original DF dataset from Google Drive
│   ├── extract_gtt23.py              # Extract GTT23 HDF5 → timestamp+direction traces
│   ├── apply_defense.py              # Apply RegulaTor/BRO to traces → pickle
│   ├── collect_traces.py             # Custom data collection via Tor Browser + tcpdump
│   ├── pcap_to_traces.py             # Convert pcap → timestamp+direction format
│   ├── traces_to_pickle.py           # Convert traces → DF-compatible pickle format
│   ├── train.py                      # Main training entry point
│   ├── evaluate.py                   # Main evaluation entry point
│   └── run_all_experiments.py        # Full experiment matrix runner
├── results/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── .gitignore
```

---

## Implementation Phases

### Phase 1: Scaffolding & Utilities

**Files:** `requirements.txt`, `.gitignore`, `configs/*.yaml`, `src/utils/*`

- **requirements.txt**: `torch>=2.0`, `numpy`, `pyyaml`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `pandas`, `h5py`, `dpkt` (pcap parsing), `stem` (Tor control), `selenium` (browser automation), `tbselenium` (Tor Browser driver)
- **Config schema** (YAML): experiment name/scenario/defense/seed, data paths/sequence_length/num_classes, model hyperparams (filters=[32,64,128,256], kernel=8, pool=8, stride=4, dropouts), training params (epochs=30, batch=128, Adamax lr=0.002)
- **device.py**: CUDA → MPS → CPU fallback
- **reproducibility.py**: Seed `random`, `numpy`, `torch`

### Phase 2: Data Loading & Preprocessing

**Files:** `src/data/preprocessing.py`, `src/data/dataset.py`, `src/data/loader.py`

- **preprocessing.py**: Load Python 2 pickles with `encoding='latin1'` fallback. Functions for closed-world splits (`X_train_NoDef.pkl` etc.) and open-world splits (separate `Mon`/`Unmon` test files). Pad/truncate to 5000.
- **dataset.py**: `TraceDataset(Dataset)` — wraps numpy arrays as `(batch, 1, 5000)` tensors (channels-first for PyTorch Conv1d)
- **loader.py**: Factory functions `get_closed_world_loaders(config)` and `get_open_world_loaders(config)` returning DataLoaders

### Phase 3: DF Model Architecture (PyTorch)

**File:** `src/models/df_net.py`

Translation of the Keras model with these critical details:
- **Block 1**: 2x Conv1d(1→32, k=8, padding='same') + BatchNorm + **ELU** + MaxPool(8, stride=4) + Dropout(0.1)
- **Blocks 2-4**: Same structure but **ReLU** instead of ELU, filters 64/128/256
- **Classifier**: FC(512) + BN + ReLU + Dropout(0.7) → FC(512) + BN + ReLU + Dropout(0.5) → FC(num_classes)
- **SameMaxPool1d**: Custom module implementing Keras-style `padding='same'` for MaxPool1d (PyTorch lacks this natively)
- Output raw logits (PyTorch `CrossEntropyLoss` includes softmax)
- Xavier uniform init for FC layers

### Phase 4: Training Pipeline

**Files:** `src/training/trainer.py`, `scripts/train.py`

- `Trainer` class: Adamax optimizer (lr=0.002, betas=(0.9, 0.999)), CrossEntropyLoss, epoch loop with tqdm, validation each epoch, checkpoint saving
- `scripts/train.py`: CLI entry point (`--config configs/closed_world_nodef.yaml`)

### Phase 5: Closed-World Evaluation — Milestone: Reproduce ~98.3%

**Files:** `src/evaluation/closed_world.py`, `src/evaluation/metrics.py`, `scripts/evaluate.py`

- Accuracy, per-class precision/recall/F1 via `sklearn.metrics.classification_report`
- Confusion matrix generation
- **Must verify ~98.3% accuracy on NoDef before proceeding** — this validates the architecture translation

### Phase 6: GTT23 Extraction & Defense Application

**Files:** `src/defenses/gtt23_extract.py`, `src/defenses/regulator.py`, `src/defenses/bro.py`, `scripts/extract_gtt23.py`, `scripts/apply_defense.py`

**GTT23 → traces pipeline:**
1. Read GTT23 HDF5 with `h5py`, inspect structure to identify trace groups
2. Extract per-website traces as `timestamp\tdirection` tab-separated files (the format both RegulaTor and BRO expect)
3. Organize into `data/traces/{website_id}/{website_id}-{instance_id}` file structure

**Defense application:**
- **RegulaTor**: Wrap/invoke `regulator_sim.py` logic (params: `budget=3550, orig_rate=277, dep_rate=0.94, threshold=3.55, upload_ratio=3.95, delay_cap=1.77`). Uses `defense_utils.get_trace()` to parse, applies rate-based regularization, outputs defended traces.
- **BRO**: Wrap/invoke `sim.sh` / underlying Python scripts. Samples dummy packets from beta distribution, injects at randomized positions.
- Both defenses output `timestamp\tdirection` files → convert to direction-only pickle format (pad/truncate to 5000, split 80/10/10 train/val/test) using RegulaTor's `output_pkl()` pattern.

### Phase 7: Open-World Evaluation

**File:** `src/evaluation/open_world.py`

- Model has `num_classes = num_monitored + 1` (unmonitored class)
- Threshold sweep on max softmax probability (15 thresholds: `1 - 1/logspace(0.05, 2, 15)`)
- Compute TP/FP/TN/FN at each threshold → TPR, FPR, Precision, Recall
- Generate Precision-Recall curves and ROC curves

### Phase 8: Visualization & Comparison

**File:** `src/visualization/plots.py`

- `plot_training_curves()` — loss/accuracy over epochs
- `plot_confusion_matrix()` — seaborn heatmap
- `plot_precision_recall_curve()` — open-world PR curves
- `plot_defense_comparison()` — bar chart: accuracy across NoDef vs RegulaTor vs BRO
- `plot_per_class_f1()` — horizontal bar chart of per-class F1

### Phase 9: Full Experiment Runner

**File:** `scripts/run_all_experiments.py`

Experiment matrix (6 experiments):

| Scenario | Defense | num_classes | Dataset |
|---|---|---|---|
| Closed-World | None | 95 | Original DF |
| Closed-World | RegulaTor | 95 | GTT23 + defense |
| Closed-World | BRO | 95 | GTT23 + defense |
| Open-World | None | 96 | Original DF |
| Open-World | RegulaTor | 96 | GTT23 + defense |
| Open-World | BRO | 96 | GTT23 + defense |

---

## Key Technical Decisions

1. **PyTorch** over Keras (user preference, matching BLE project style)
2. **ELU in Block 1 only** — preserves negative direction values (-1 for incoming packets); ReLU in later blocks
3. **SameMaxPool1d** — custom module needed because PyTorch MaxPool1d doesn't support `padding='same'`
4. **Full trace defense simulation** using GTT23 timestamped data rather than direction-only approximation — more faithful to how the defenses actually operate
5. **Original DF dataset for replication** — ensures direct comparison with paper's reported numbers

## Potential Challenges

- **GTT23 HDF5 structure**: Unknown until downloaded; Phase 6 starts by inspecting the file structure with `h5py`
- **Python 2 pickle compatibility**: DF dataset uses cPickle; handled with `encoding='latin1'`
- **Memory**: Full training set is ~1.5 GB as float32; fine for 16GB+ RAM
- **MPS limitations**: Some PyTorch ops may not work on Apple Metal; device.py includes CPU fallback

---

## Possible Extensions & Additional Experiments

### A. Concept Drift / Temporal Analysis

Train on the original 2018 DF dataset, test on self-collected 2026 traces of the same Alexa-100 sites. Measures how much the DF model degrades when website content/structure evolves over time.

- **Setup**: Same 100 monitored sites, model trained on 2018 data, evaluated on fresh data
- **Expected finding**: Significant accuracy drop, quantifying the "shelf life" of a WF model
- **Mitigation experiment**: Fine-tune the 2018 model on a small number of new traces (few-shot adaptation)

### B. Attack Model Comparison

Implement additional WF attacks and compare against DF on the same datasets. Use [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) (PyTorch implementations of 11 attacks) as a starting point.

| Attack | Type | Key Insight |
|---|---|---|
| k-NN | Manual features + kNN | Classical baseline |
| CUMUL | SVM + cumulative packet features | Strong pre-DL baseline |
| Var-CNN | Dilated causal ResNet | Separate timing + direction streams |
| Tik-Tok | Direction x timing CNN | [github.com/msrocean/Tik_Tok](https://github.com/msrocean/Tik_Tok) |
| Triplet Fingerprinting | Siamese network, few-shot | Works with limited training data |

### C. Additional Defenses

Test DF against more defenses beyond RegulaTor and BRO:

| Defense | Mechanism | Expected Impact |
|---|---|---|
| FRONT | Random dummy cells at connection start | Moderate accuracy drop |
| Tamaraw | Fixed-schedule regularization | Near-total defense (~3-5% attack accuracy) but very high overhead |
| Surakav | GAN-generated sending patterns | Strong defense, 50% less overhead than Tamaraw |
| TrafficSliver | Split traffic across multiple entry guards | Partial trace at each guard |

[WFDefProxy](https://github.com/websitefingerprinting/wfdef) implements FRONT, Tamaraw, RegulaTor, and Surakav as real Tor Pluggable Transports.

### D. Data Augmentation (NetAugment)

Apply traffic-level augmentations during training to improve robustness ([Bahramali et al., CCS 2023](https://arxiv.org/abs/2309.10147)):

- **Burst merging**: Combine adjacent bursts (simulates low-latency paths)
- **Burst splitting**: Split bursts into sub-bursts (simulates jitter)
- **Burst insertion**: Insert random dummy bursts (simulates background traffic)
- **Random packet dropping**: Simulates congestion/loss
- **Timestamp noise**: Gaussian noise on inter-packet delays

### E. Input Length Sensitivity

Test how classification accuracy varies with sequence length:

| Length | What It Tests |
|---|---|
| 500, 1000, 2000 | Early-stage classification (page still loading) |
| 3000 | Reduced input |
| 5000 | Standard (paper default) |
| 8000, 10000 | Extra context (mostly zero-padding for most pages) |

Generates an accuracy-vs-length curve.

### F. Feature Importance / GradCAM Visualization

Apply Grad-CAM to the final convolutional layer to produce saliency maps over the 5000-element input:

- Reveals which temporal positions (e.g., initial burst, specific handshake packets) are most discriminative
- Compare saliency maps across defended vs undefended traces to understand what defenses successfully obscure

### G. Adversarial Evasion

Test robustness of the DF model to adversarial perturbations:

- **FGSM/PGD attacks**: Gradient-based perturbations on input sequences (add/remove dummy packets)
- Measure accuracy drop as a function of perturbation budget (number of dummy cells added)

### H. Architecture Variants

Swap the DF CNN backbone for modern architectures and compare:

- **ResNet-based** (Var-CNN style dilated causal convolutions)
- **Transformer-based** (following TMWF / TrafficFormer)
- **Self-supervised pre-training** (NetCLR-style contrastive learning before fine-tuning)

### I. Bandwidth Overhead vs. Accuracy Tradeoff

Generate a unified plot showing all tested defenses on a single chart:
- X-axis: bandwidth overhead (%)
- Y-axis: DF attack accuracy (%)
- Each defense is a point (or line if testing multiple configurations)
- Visualizes the fundamental tradeoff defenders face

---

## Verification

1. **Phase 5 milestone**: Closed-world NoDef accuracy should be ~98.3% (paper reports 98.27%)
2. **Defense experiments**: Compare against published RegulaTor/BRO results where available
3. **Run all tests**: `python scripts/train.py --config configs/closed_world_nodef.yaml` then `python scripts/evaluate.py --config configs/closed_world_nodef.yaml --checkpoint results/checkpoints/best.pt`
4. **Visual inspection**: Confusion matrices should show diagonal dominance for NoDef, degradation for defended scenarios
5. **Open-world sanity**: PR curves should show high precision at low recall thresholds
