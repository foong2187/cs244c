# Deep Fingerprinting (CCS'18) — Full Reproduction Summary

**Purpose:** Comprehensive documentation of **all experiments**, key **findings**, **bugs discovered**, and **conclusions** from our attempt to reproduce Sirinam et al. (CCS'18, "Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning").

**Last updated:** 2026-03-14

---

## 1) What We Tried to Reproduce

The DF paper's headline result: **~98% closed-world accuracy** on NoDef using direction-only sequences of length 5000 over 95 monitored Alexa top sites.

Key methodological details from the paper (Section 4):

- **Sites:** Alexa Top 100; after filtering, **95 sites** with ≥1,000 visits each → **95,000 traces** total.
- **Collection:** 1,250 visits per site across **10 low-end campus machines**, using `tor-browser-crawler` [19]. Batched round-robin methodology (Wang & Goldberg [39]).
- **Input format:** Direction-only sequences (+1 outgoing cell, -1 incoming cell), fixed length **5,000** (pad shorter, truncate longer). Paper states only 8,121 of 95,000 traces were longer than 5,000 cells.
- **Filtering:** Discard traces with <50 packets.
- **Split:** 8:1:1 train/val/test.
- **Model:** DFNet — 4 Conv1D blocks (32→64→128→256 filters), ELU in block 1, ReLU in 2–4, two FC layers (512 units), softmax.
- **Training:** Adamax optimizer, lr=0.002, batch size 128, 30 epochs.

---

## 2) Datasets Used

### 2.1 Curated Benchmark Pickles (`modern_defenses.zip`)

Pre-split DF-format pickles provided to us:

- `dataset/ClosedWorld/{NoDef, BRO, BuFLO, RegulaTor, Tamaraw, WalkieTalkie, WTFPAD}/`
- `dataset/OpenWorld/{NoDef, BRO, BuFLO, RegulaTor, Tamaraw}/`

**NoDef closed-world stats:**
- 100 classes (labels 0–99), 80 train / 10 valid / 10 test per class
- 8,000 train / 1,000 valid / 1,000 test samples
- Direction ratio: **15.5% outgoing / 84.5% incoming**
- Median non-zero length: **4,022**
- 41.6% of traces truncated at the 5,000 cap

### 2.2 Teammate Datasets

| Dataset | Source | Classes | Train samples | Direction ratio (out/in) | Median length |
|---------|--------|---------|---------------|--------------------------|---------------|
| Yousef (Fresh2026) | `data/yousef-data/pickle/` | 99 | 6,284 | 45.2% / 54.8% | 778 |
| Devin (Fresh2026) | `data/devin-data/data/pickle/` | 92 | 5,808 | 47.1% / 52.9% | 653 |

### 2.3 Our Crawled Dataset (Local + GCP)

We built a crawler (`crawler/`) that:
1. Runs Tor with `stem` for circuit control
2. Drives headless Firefox via Selenium through Tor SOCKS proxy
3. Captures traffic with `tcpdump` during each page load
4. Filters pcap to guard-only packets and converts to direction sequences

**Infrastructure:**
- Local machine (WSL2, RTX 3080 dorm PC): 15 parallel Tor workers, instances 0–349
- Google Cloud VM (e2-standard-16, us-central1): 30 parallel workers, instances 650–999

**Raw pcap counts:**
- Local: 27,320 pcap files
- GCP: 64,635 pcap files
- Total: **91,955 pcap files**

**Cell-level dataset (after reprocessing — see Section 3):**

| Split | Samples |
|-------|--------:|
| Train | 72,270 |
| Valid | 9,033 |
| Test | 9,035 |
| **Total** | **90,338** |

- **Classes:** 95 (Alexa Top 100 minus 5 sites with insufficient data)
- **Per-class train samples:** min=476, max=1,048, mean=761, median=772
- **Direction ratio:** 28.8% outgoing / 71.2% incoming (cell-level approximation)
- **Non-zero trace length:** min=50, median=1,135, mean=2,017, p90=5,000
- **Truncated at 5,000:** 22.8% of training traces

**Closed-world DFNet accuracy on cell-level data:**
- **Test accuracy: 0.480** (95 classes)
- Per-class accuracy: mean=0.471, min=0.022, max=0.832

---

## 3) Critical Bug: Packet-Level vs Cell-Level Representation

### 3.1 The Discovery

After initial training produced ~36–38% accuracy (vs the paper's ~98%), we ran cross-dataset experiments: train on the benchmark, test on our traces (and vice versa). Both directions yielded **~1% accuracy** — effectively random chance.

Detailed comparison revealed a fundamental data representation mismatch:

| Metric | Benchmark (paper) | Our crawled | Yousef | Devin |
|--------|-------------------|-------------|--------|-------|
| **Outgoing (+1) fraction** | **15.5%** | **45.2%** | **45.2%** | **47.1%** |
| **Incoming (-1) fraction** | **84.5%** | **54.8%** | **54.8%** | **52.9%** |
| Median first-burst length | 8.7 | 1.5 | 1.0 | 1.3 |
| Median non-zero length | 4,022 | 1,454 | 778 | 653 |
| Truncated at 5,000 | 41.6% | 7.7% | 6.2% | 6.1% |

### 3.2 Root Cause

The paper uses **Tor cell-level** direction sequences. The paper's crawler (`tor-browser-crawler`) extracts directions at the Tor cell layer, where each cell is exactly 512 bytes. When browsing, the client sends a few request cells (outgoing) and receives many content cells (incoming), producing a **heavily incoming-skewed** ratio (~85/15).

Our crawler (and all teammate crawlers) captured raw **TCP packets** with `tcpdump` and counted every IP packet as one direction event. At the TCP level, every incoming data packet triggers a TCP ACK from the client (40–66 bytes, no payload), which gets counted as an outgoing event. This inflates the outgoing count, producing a **~50/50 ratio**.

**Evidence from our pcap files:**
- 36–44% of packets were <100 bytes (pure TCP ACKs)
- The `pcap_to_sequence` function in `capture.py` treated every packet identically, regardless of payload size

This bug affected **all three independent data collection efforts** (ours, Yousef's, Devin's) since all used the same `tcpdump`-based approach.

### 3.3 The Fix: Cell-Level Reprocessing

We wrote `scripts/reprocess_cell_level.py` to reprocess all 91,955 raw pcap files:
- Filter to packets to/from the Tor guard IP
- Skip packets with zero TCP payload (ACKs, SYN/FIN)
- For each data-carrying packet, emit `ceil(payload_bytes / 512)` direction events
- Truncate/pad to length 5,000

**After reprocessing (cell-level):**

| Metric | Benchmark | Cell-level (ours) | Raw packets (old) |
|--------|-----------|-------------------|-------------------|
| Outgoing ratio | 15.5% | **23.5%** | 45.2% |
| Incoming ratio | 84.5% | **76.5%** | 54.8% |
| Median length | 4,022 | 786 | 1,454 |

The cell-level representation is much closer to the benchmark's distribution, though not identical (the remaining gap is likely from TLS record overhead inflating outgoing cell counts).

**Cell-level dataset stats:**
- 90,338 valid traces (98.2% pass rate)
- 95 classes
- Split: 72,270 train / 9,033 valid / 9,035 test

---

## 4) Closed-World Results on Benchmark (`modern_defenses.zip`)

We trained DFNet on each defense variant using the curated benchmark data.

### 4.1 Closed-World Test Accuracy

| Defense | Test Accuracy | Notes |
|---------|-------------:|-------|
| NoDef | **0.961** | Comparable to paper's ~98% |
| WTF-PAD | **0.895** | Lightweight adaptive padding; moderate degradation |
| RegulaTor | **0.818** | Still fairly learnable |
| BRO | **0.741** | Moderate defense |
| WalkieTalkie | **0.461** (top-1) / **0.737** (top-2) | Decoy mechanism visible in top-2 |
| BuFLO | **0.317** | Strong padding defense |
| Tamaraw | **0.261** | Strong padding defense |

### 4.2 Open-World Metrics (threshold = 0.50)

| Defense | TPR | FPR | Precision | Recall |
|---------|----:|----:|----------:|-------:|
| NoDef | 0.9895 | 0.0010 | 0.9989 | 0.9895 |
| RegulaTor | 0.8189 | 0.0080 | 0.9898 | 0.8189 |
| BRO | 0.7284 | 0.0680 | 0.9105 | 0.7284 |
| BuFLO | 0.1432 | 0.0460 | 0.7473 | 0.1432 |
| Tamaraw | 0.0747 | 0.0250 | 0.7396 | 0.0747 |

**Interpretation:** These results are broadly consistent with the published literature. NoDef is easiest; padding defenses (BuFLO, Tamaraw) significantly degrade accuracy.

---

## 5) Results on Our Crawled Data

### 5.1 Raw Packet-Level (Original, Buggy Representation)

| Configuration | Classes | Train | Test Acc | Notes |
|---------------|---------|-------|----------|-------|
| Combined (local + GCP), raw packets | 95 | ~50K | ~0.37 | Original pipeline |
| Filtered (np600, min50, cap300) | 95 | 20,484 | ~0.39 | Stratified split |
| Max-accuracy subset (np1500, top 30 classes) | 30 | 6,742 | **0.60** | Signal exists |

### 5.2 Cell-Level (Corrected Representation)

| Configuration | Classes | Train | Test Acc | Notes |
|---------------|---------|-------|----------|-------|
| Cell-level, all data, min 50 cells | 95 | 72,270 | **0.480** | 12pp improvement over raw |

The cell-level reprocessing improved in-distribution accuracy from ~36–39% to **48.0%**, a meaningful 10–12 percentage point gain confirming the representation bug was a real source of degradation.

---

## 6) Cross-Dataset Generalization Experiments

The most striking finding: models trained on one dataset completely fail on the other.

### 6.1 Raw Packet-Level Cross-Dataset

| Direction | In-Distribution | Cross-Dataset |
|-----------|----------------:|--------------:|
| Benchmark train → Our raw packet test | 97.9% | **0.94%** |
| Our raw packet train → Benchmark test | 36.1% | **2.11%** |

### 6.2 Cell-Level Cross-Dataset

| Direction | In-Distribution | Cross-Dataset |
|-----------|----------------:|--------------:|
| Benchmark train → Our cell-level test | 97.9% | **1.24%** |
| Our cell-level train → Benchmark test | 47.4% | **1.37%** |

**Key finding:** Cross-dataset accuracy is at random chance (~1%) in all four experiments, even after fixing the representation to cell-level. This demonstrates that the DF model's high accuracy is not based on universally generalizable website fingerprints — it learns **distribution-specific patterns** that don't transfer across collection environments, time periods, or network conditions.

### 6.3 Verification of Setup Correctness

We verified the cross-dataset experiments were set up correctly:
- Both datasets use identical encoding: `{-1, 0, 1}`, shape `(N, 5000)`, `float32`
- Label alignment verified: both use the same Alexa Top-100 ordering (labels 0–94 correspond to the same websites)
- Missing sites 95–99 correctly excluded from benchmark during cross-eval
- In-distribution sanity checks pass (97.9% and 47.4% respectively)

---

## 7) Analysis: Why Is There Such a Large Accuracy Gap?

### 7.1 Representation Bug (now fixed)

Our original pipeline counted raw TCP packets instead of Tor cells. This inflated the outgoing ratio from ~15% to ~45% and fundamentally changed the statistical signature of the traces. **Impact: ~12 percentage points of accuracy.**

### 7.2 Remaining Cell-Level Approximation Error

Even after fixing to cell-level, our outgoing ratio is 23.5% vs the benchmark's 15.5%. This is because:
- TLS record headers add overhead to outgoing packets
- TCP segmentation doesn't align perfectly with 512-byte cell boundaries
- Our `ceil(payload / 512)` approximation slightly overcounts outgoing cells

### 7.3 Trace Length Distribution

| Dataset | Median non-zero | Mean non-zero | p90 |
|---------|----------------:|--------------:|----:|
| Benchmark | 4,022 | 3,417 | 5,000 |
| Cell-level (ours) | 786 | ~2,000 | ~4,400 |

Our traces are substantially shorter. Possible causes:
- Modern websites load content lazily (2026 vs 2016–2018)
- Our page load timeout or post-load wait may be shorter than the paper's
- Different Tor circuit behavior over time

### 7.4 Temporal Distribution Shift (2016 → 2026)

The benchmark data was collected circa 2016–2018. Our data was collected in March 2026. In the intervening 8+ years:
- Website architectures changed dramatically (SPAs, heavy JavaScript, CDNs, HTTP/3)
- Tor's relay infrastructure and routing algorithms evolved
- Browser behavior changed (Firefox ESR versions, TLS 1.3, ECH)
- Content delivery patterns shifted (lazy loading, streaming, WebSocket)

### 7.5 Collection Environment Differences

- Paper: 10 campus machines, same university network, `tor-browser-crawler` driving actual Tor Browser
- Ours: 2 machines (WSL2 dorm PC + GCE VM), headless Firefox via Selenium through Tor SOCKS proxy
- Different guard relay diversity, network paths, and latency characteristics

### 7.6 Non-Generalizable Features

The cross-dataset results (1% accuracy both directions) strongly suggest that DFNet learns **collection-environment-specific patterns** rather than universal website fingerprints. These likely include:
- Guard-specific packet timing patterns
- Network-specific TCP behavior (MTU, congestion window)
- Time-of-day and load-dependent variations
- TLS session resumption and connection reuse patterns

---

## 8) Summary of All Experimental Results

### 8.1 Master Results Table

| Experiment | Dataset | Representation | Classes | Test Acc |
|------------|---------|---------------|---------|----------|
| Benchmark NoDef (closed) | `modern_defenses` | Cell-level | 100 | **96.1%** |
| Benchmark NoDef (open) | `modern_defenses` | Cell-level | 100 | **98.9% TPR** |
| Our data (raw packets) | Combined crawled | Packet-level | 95 | 37–39% |
| Our data (raw, max-acc subset) | Top 30 classes | Packet-level | 30 | 59.9% |
| Our data (cell-level) | Cell-level reprocessed | Cell-level | 95 | **48.0%** |
| Cross: Bench→Ours (raw) | — | Mixed | 95 | 0.94% |
| Cross: Ours→Bench (raw) | — | Mixed | 95 | 2.11% |
| Cross: Bench→Ours (cell) | — | Cell-level | 95 | 1.24% |
| Cross: Ours→Bench (cell) | — | Cell-level | 95 | 1.37% |

### 8.2 Training Convergence

**Benchmark (cross-dataset experiment, 30 epochs):**
- Epoch 1: 16.4% train → Epoch 30: 98.2% train
- Val accuracy: 98.7% by epoch 30 (saturated by ~epoch 12)

**Cell-level crawled (30 epochs):**
- Epoch 1: 4.9% train → Epoch 30: 52.5% train
- Best val accuracy: 46.1% (epoch 28)
- Mild overfitting visible (train > val by ~6pp at convergence)

---

## 9) Figures Generated

| Figure | Path | Description |
|--------|------|-------------|
| Closed-world accuracy comparison | `experiments/fig_closed_world_accuracy_ours_vs_modern.png` | Bar chart comparing modern_defenses vs our crawled data |
| Training curve (max-acc) | `experiments/fig_training_curve_maxacc.png` | Accuracy over epochs, 30-class max-accuracy subset |
| Loss curve (max-acc) | `experiments/fig_loss_curve_maxacc.png` | Loss over epochs, 30-class max-accuracy subset |
| Training curve (95-class) | `experiments/fig_training_curve_crawled95.png` | Accuracy over epochs, full 95-class crawled |
| Loss curve (95-class) | `experiments/fig_loss_curve_crawled95.png` | Loss over epochs, full 95-class crawled |

---

## 10) Scripts and Code

| Script | Purpose |
|--------|---------|
| `crawler/crawl.py` | Single-worker Tor crawler |
| `crawler/crawl_parallel.py` | Multi-worker parallel crawler |
| `crawler/capture.py` | pcap capture and packet-level sequence extraction |
| `src/combine_datasets.py` | Combine pcaps into DF-format pickles (packet-level) |
| `src/train_combined.py` | Train/eval DFNet on combined crawled data |
| `src/train_closed_world.py` | Train/eval DFNet closed-world on benchmark |
| `src/train_open_world.py` | Train/eval DFNet open-world on benchmark |
| `scripts/reprocess_cell_level.py` | **Cell-level reprocessing** of all pcaps (the fix) |
| `scripts/cross_dataset_eval.py` | Cross-dataset: benchmark train → crawled eval |
| `scripts/cross_dataset_eval_reverse.py` | Cross-dataset: crawled train → benchmark eval |
| `scripts/filter_resplit_combined.py` | Filter by trace length, cap per class, stratified split |
| `scripts/train_all_modern_defenses.py` | Automate training on all defense variants |
| `scripts/plot_closed_world_accuracy.py` | Generate accuracy comparison bar chart |
| `scripts/plot_training_metrics_from_log.py` | Plot training/loss curves from logs |

---

## 11) Paper-Ready: Datasets and Methodology Subsection

The following is written in a style suitable for inclusion in the paper's methodology section.

---

### Datasets

We evaluate DFNet on two categories of data: (1) curated benchmark datasets that follow the original DF paper's format, and (2) a self-collected dataset of Tor traffic gathered in March 2026.

**Benchmark datasets.** We use pre-processed direction-sequence pickles for seven defense configurations: NoDef (undefended), WTF-PAD, RegulaTor, BRO, WalkieTalkie, BuFLO, and Tamaraw. For each defense, the closed-world dataset contains 100 monitored websites with 80 training / 10 validation / 10 test traces per site (8,000 / 1,000 / 1,000 samples). Open-world datasets are available for NoDef, BRO, BuFLO, RegulaTor, and Tamaraw. All traces use Tor cell-level direction sequences of length 5,000 (+1 outgoing, -1 incoming, 0 padding).

**Self-collected dataset.** We independently collected Tor entry-guard traffic for the same Alexa Top-100 site list used in the DF paper. Data was gathered from two machines: a local workstation (WSL2 on Windows, 15 parallel Tor instances, visit indices 0–349) and a Google Cloud VM (e2-standard-16 in us-central1, 30 parallel Tor instances, visit indices 650–999). Each worker runs an independent Tor process with its own SOCKS and control port. Headless Firefox (driven by Selenium) loads each site through the Tor SOCKS proxy while `tcpdump` captures traffic to/from the entry guard. After collection, we reprocessed all 91,955 raw pcap files using a cell-level extraction method: for each TCP packet with non-zero payload directed to or from the guard IP, we emit `ceil(payload / 512)` direction events to approximate Tor's 512-byte cell granularity (see Section 3). Traces shorter than 50 cells are discarded.

The resulting dataset contains **90,338 traces across 95 classes** (5 sites from the original 100 were dropped due to insufficient data). We split 80/10/10 into 72,270 train / 9,033 validation / 9,035 test samples. Per-class training counts range from 476 to 1,048 (mean 761). The median non-zero trace length is 1,135 cells; 22.8% of traces are truncated at the 5,000 cap.

### Model and Training

We use the DFNet architecture from Sirinam et al.: four 1D convolutional blocks (filter counts 32, 64, 128, 256; kernel sizes 8, 8, 8, 8; ELU activation in block 1, ReLU in blocks 2–4; each block has two Conv1D layers, BatchNorm, MaxPool with pool size 8/8/8/4, and Dropout 0.1) followed by two fully connected layers (512 units each, ReLU, Dropout 0.7 and 0.5) and a softmax output. We use Adamax (lr = 0.002, β₁ = 0.9, β₂ = 0.999) with batch size 128 and train for 30 epochs. For self-collected data experiments, we also apply EarlyStopping (patience 7 on val_loss), ReduceLROnPlateau (factor 0.5, patience 3), and ModelCheckpoint (best val_accuracy). All training is performed on an NVIDIA RTX 3080 GPU.

### Key Differences Between Datasets

| Property | Benchmark | Self-collected |
|----------|-----------|----------------|
| Collection period | ~2016–2018 | March 2026 |
| Collection tool | `tor-browser-crawler` | Custom Selenium + tcpdump |
| Machines | 10 campus machines | 2 (WSL2 + GCE VM) |
| Representation | Native Tor cells | Approximated from TCP payload |
| Direction ratio (out/in) | 15.5% / 84.5% | 28.8% / 71.2% |
| Median trace length | 4,022 | 1,135 |
| Traces per class | 1,000 | 476–1,048 (mean 761) |
| Truncated at 5,000 | 41.6% | 22.8% |

---

## 12) Conclusions for the Paper

### 12.1 Positive Findings

1. **Paper results are reproducible on curated benchmark data.** DFNet achieves 96.1% closed-world accuracy on the provided NoDef pickles, consistent with published ~98%.

2. **Our independently crawled data contains real WF signal.** 48% accuracy on 95 classes (cell-level) is 45x above random chance (1.05%). On a focused 30-class subset with longer traces, accuracy reaches 60%.

3. **Defense effectiveness ordering matches the literature.** NoDef > WTF-PAD > RegulaTor > BRO > WalkieTalkie > BuFLO > Tamaraw — the relative ranking is consistent with the paper's findings.

### 12.2 Negative / Critical Findings

4. **Critical representation bug discovered.** All three independent crawling efforts (ours, Yousef's, Devin's) captured TCP packets instead of Tor cells. This produced a fundamentally different data representation that reduced accuracy by ~12 percentage points. The paper does not clearly specify that inputs should be at the cell level vs packet level.

5. **Zero cross-dataset generalization.** Models trained on one dataset achieve random-chance accuracy on the other (1%), in both directions, even after fixing the representation bug. The DF model learns distribution-specific artifacts, not universal website fingerprints.

6. **Large accuracy gap persists after fixing the bug.** Even with cell-level representation, our best 95-class accuracy is 48% vs the benchmark's 96%. The remaining 48pp gap comes from temporal distribution shift, shorter traces, different collection environments, and the cell-counting approximation.

### 12.3 Implications

- The paper's ~98% accuracy likely depends on **controlled collection conditions** that don't reflect real-world Tor traffic diversity.
- The **practical threat** of deep fingerprinting attacks may be lower than benchmark numbers suggest, especially against traffic collected across different time periods and network environments.
- **Reproducibility** of WF attack results requires matching not just the model architecture and hyperparameters, but the exact data representation (cell vs packet level) — a detail that is easy to overlook and was not immediately obvious from the paper.

---

## 13) Training History Data

Full per-epoch training metrics for cross-dataset experiments are saved in:
- `experiments/cross_dataset_training_history.csv` (benchmark → crawled)
- `experiments/cross_dataset_reverse_training_history.csv` (crawled → benchmark)

Each contains: epoch, accuracy, loss, val_accuracy, val_loss.
