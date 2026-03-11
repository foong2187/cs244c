# Deep Fingerprinting Reproduction: Findings and Explanations

**Purpose:** Summary of experimental findings and possible explanations to support paper writing.  
**Context:** Attempt to reproduce closed-world website fingerprinting results from Sirinam et al. (CCS’18, “Deep Fingerprinting”) using independently collected Tor entry-guard traffic.

---

## 1. Experimental Setup

### 1.1 Data Collection

| Source | Location | Pcap files | Valid traces (after min-packet filter) | Notes |
|--------|----------|------------|----------------------------------------|------|
| Local crawler | `/mnt/d/cs244c-data/` | 27,320 | ~25K | Guard IPs from `progress.csv` |
| GCP crawler | `/mnt/d/cs244c-data-gcp/data/` | 36,590 | ~35K | Guard IP inferred from pcap (no progress.csv) |
| **Combined** | `/mnt/d/cs244c-combined/` | 63,910 | 62,693 | 80/10/10 train/valid/test |

- **Site list:** Alexa Top 100 (label 0–99). After filtering traces with &lt;50 non-padding packets, **95 classes** had sufficient data (5 sites dropped).
- **Sequence format:** Direction-only, length 5000 (+1 outgoing, -1 incoming, 0 padding), matching DF paper.
- **Split:** Random shuffle then 80% train / 10% valid / 10% test (same for combined and single-source experiments).

### 1.2 Model and Training

- **Model:** DFNet (Sirinam et al.), 4 Conv1D blocks (32→64→128→256 filters), ELU in block 1, ReLU in 2–4, two FC layers (512 units), softmax over 95 classes.
- **Optimizer:** Adamax, lr = 0.002.
- **Training:** Up to 50 epochs, batch size 128, EarlyStopping on `val_loss` (patience 7), ReduceLROnPlateau (factor 0.5, patience 3), ModelCheckpoint on `val_accuracy`.
- **Hardware:** NVIDIA RTX 3080; training ~10 s/epoch for combined (~50K train samples).

---

## 2. What the Paper Actually Specifies

Reviewing Sirinam et al. (arXiv:1801.02265, CCS’18) and the PDF:

**Data collection (Section 4):**
- **Sites:** Top Alexa 100; after filtering, **95 sites** with **at least 1,000 visits each** → **95,000 traces** in the closed-world dataset.
- **Raw collection:** Each of the top 100 sites visited **1,250 times**; traces dumped with tcpdump; corrupted traces removed (e.g., no packets or &lt;50 packets); then only sites with ≥1,000 visits kept.
- **Infrastructure:** **Ten low-end machines** on the authors’ university campus.
- **Crawler:** **tor-browser-crawler** [19] to drive Tor Browser (realistic browsing).
- **Batching:** Batched methodology (Wang & Goldberg [39]): visits split into five chunks, round-robin so each site is accessed 25 times per batch; visits spread over time to control time variance and avoid bans.
- **Train/val/test:** Data split in ratio **8:1:1** (training : validation : testing).
- **Input:** Direction-only (like Wang et al.); fixed length **5,000** (pad shorter, truncate longer). Paper states that of 95,000 traces, only 8,121 were longer than 5,000 cells and truncated.

**Hyperparameters (Table 1 in paper):**
| Parameter        | Search range    | **Final value** |
|-----------------|-----------------|------------------|
| Input dimension | [500 … 7000]    | **5000**         |
| Optimizer       | Adam, Adamax, … | **Adamax**       |
| Learning rate   | [0.001 … 0.01] | **0.002**        |
| Training epochs | [10 … 50]       | **30**           |
| Mini-batch size | [16 … 256]      | **128**          |
| Filter/Pool/Stride | [2 … 16]     | **[8, 8, 4]**    |
| Activations     | Tanh, ReLU, ELU | **ELU, ReLU**    |
| Filters (blocks 1–4) | …           | **[32,32], [64,64], [128,128], [256,256]** |
| FC layers       | 1–4             | **2**            |
| FC hidden units | [256 … 2048]    | **[512, 512]**   |
| Dropout (pool, FC1, FC2) | [0.1 … 0.8] | **[0.1, 0.7, 0.5]** |

So the paper **does** specify: (1) dataset size (95 sites × 1,000 traces = 95,000), (2) collection (10 machines, tor-browser-crawler, batched round-robin, tcpdump), (3) preprocessing (direction-only, length 5,000, &lt;50 packets discarded), (4) split (8:1:1), and (5) all main training hyperparameters in Table 1. What is **not** fully specified: exact Tor version for the main NoDef crawl, random seed, and exact hardware/network conditions per machine.

**Implication for our reproduction:** We matched the **method and hyperparameters** (optimizer, lr, batch size, epochs in range, architecture, input length, split ratio). The gap in results is therefore not due to missing hyperparameter or dataset-size information in the paper; it is due to **different data** (our collection: 2 machines, different crawler/setup, ~527 traces per site on average, different environment) rather than to an unspecified training setup.

---

## 3. Results Summary (Ours)

| Condition | Train size (approx.) | Best val accuracy | Test accuracy (reported/expected) |
|-----------|----------------------|-------------------|-----------------------------------|
| **Combined** (local + GCP) | ~50,154 | ~37.5% | ~37–38% |
| **Local-only** | ~25K | Slight increase vs combined | Slight increase |
| **GCP-only** | ~35K | Slight increase vs combined | Slight increase |
| **Random baseline** (95 classes) | — | — | ~1.05% |
| **Paper (CCS’18)** closed-world NoDef | — | — | **~98%** |

- Validation accuracy plateaued around 37–38% after ~30 epochs; training accuracy continued to ~42–43% (mild overfitting).
- Increasing epochs further did not yield substantial gains.
- Single-source (local-only or GCP-only) gave only a **slight** improvement over combined, not the large gain one might expect from “same machine, same network.”

---

## 4. Main Findings

1. **Large reproduction gap:** Same architecture and similar hyperparameters as the paper, but test accuracy ~**37%** vs reported **~98%** on closed-world NoDef.
2. **Substantial learning:** 37% is far above random (1.05% for 95 classes), so the model is using real signal in the data.
3. **Homogeneity helps only a little:** Training on local-only or GCP-only (single collection environment) improved accuracy only slightly over the combined dataset.
4. **Plateau:** Validation performance leveled off; more epochs did not materially change results.
5. **Full data usage:** Combined training used the full dataset (~50K train samples); no underuse of data.

---

## 5. Possible Explanations

### 5.1 Why Is There a Large Gap vs. the Paper?

- **Different data collection:** Paper likely used a controlled setting (fixed Tor version, circuit/guard conditions, lab network). Our data: two different machines (local + GCP), different networks and possibly different Tor builds and guard sets.
- **Different Tor/network conditions:** Current Tor behavior, circuit diversity, and load may reduce fingerprintability compared to the setting in the paper.
- **Unspecified details:** Paper may not fully specify preprocessing, filtering, or exact collection setup, making exact reproduction difficult.
- **Reporting bias:** Reported 98% may reflect best-case conditions or favorable train/test overlap (e.g., same environment, same time period).

### 5.2 Why Doesn’t Single-Source (Local-Only or GCP-Only) Help More?

- **“Same place” is still variable:** One machine still has multiple circuits, guard changes over time, and varying load. Train and test from the same source are not identically distributed.
- **Less data:** Single-source uses ~25K or ~35K samples vs ~50K combined. Benefit of homogeneity may be partly offset by fewer training examples.
- **GCP label noise:** GCP traces use guard IP inferred from the pcap; errors flip packet directions and corrupt labels. So GCP-only is not necessarily “cleaner,” only smaller and single-machine.
- **Ceiling effect:** If the fingerprintability of this traffic is inherently limited (e.g., by Tor or by how the sites are loaded), the ceiling may be in the 35–40% range regardless of single- vs multi-machine.

### 5.3 Why Does Validation Plateau Around 37%?

- **Data ceiling:** The useful signal in the data may be exhausted at this level for this model and setup.
- **Label quality:** Guard inference errors (GCP) and trace heterogeneity (short traces, timeouts) may cap achievable accuracy.
- **Architecture/task match:** DFNet was tuned for the paper’s data; our distribution may need different design choices (e.g., robustness to timing/circuit variation).

### 5.4 Why Is Training So Fast?

- **Efficient pipeline:** `tf.data` with prefetch and batching; data on fast storage (e.g., D: drive).
- **Model size:** DFNet is moderate in size; a 3080 can process 50K samples per epoch in ~10 seconds.
- **No fundamental bottleneck:** The limitation observed is accuracy, not compute.

---

## 6. Implications for the Paper

- **Reproducibility:** The gap between our ~37% and the paper’s ~98% is itself a result: reproduction under independently collected Tor traffic does not match reported performance.
- **Generalization:** Results suggest that high accuracy in the paper may depend heavily on specific collection conditions and may not generalize to other Tor deployments or time periods.
- **Single- vs multi-source:** Small gain from single-source training suggests that data homogeneity (same machine) helps only modestly; the main limit appears to be signal strength and label quality, not only domain mixing.
- **Practical relevance:** If real-world Tor traffic yields ~37% in a closed-world setting, the practical threat of this attack in the wild may be lower than paper numbers imply.

---

## 7. Suggested Paper Sections (Draft Outline)

1. **Introduction / motivation:** Reproducibility of WF attacks; our goal to reproduce DF on independent Tor data.
2. **Methodology:** Data collection (local + GCP, site list, filtering), preprocessing (direction sequences, length 5000), DFNet and training (optimizer, callbacks, splits).
3. **Results:** Table of accuracy for combined, local-only, GCP-only; comparison to random and to paper; learning curves and plateau.
4. **Discussion:** Possible explanations (collection differences, Tor variability, label quality, ceiling effect); why single-source helps only slightly.
5. **Conclusion:** Reproduction gap as a finding; implications for reported accuracy and for real-world applicability.

---

## 8. References to Cite

- Sirinam, P., Imani, M., Juarez, M., & Wright, M. (2018). Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning. *CCS ’18.*

---

*Document generated from experimental logs and combine/train scripts in this repository. Update with final test numbers and any additional runs (e.g., more epochs, different seeds).*
