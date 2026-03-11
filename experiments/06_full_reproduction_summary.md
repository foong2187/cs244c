# Deep Fingerprinting (CCS'18) — Reproduction Summary (Our Runs)

**Purpose:** A paper-ready consolidation of **all experiments**, key **insights**, and the main **reasons for discrepancies** we observed while trying to reproduce Sirinam et al. (CCS'18, “Deep Fingerprinting”).

This repo contains multiple experimental notes. This document is intended to be the single source of truth for the writeup.

---

## 1) What we tried to reproduce (from the DF paper)

The DF paper’s headline closed‑world result is often summarized as **~98% accuracy on NoDef** (closed‑world, direction-only sequences of length 5000, 95 monitored sites).

Key methodological items from the paper (as interpreted for our reproduction):

- **Closed‑world**: top Alexa sites, \(\sim\)95 classes after filtering.
- **Input**: direction-only sequences, fixed length **5000** (\(+1/-1/0\) padding).
- **Filtering**: discard traces with <50 packets.
- **Split**: **8:1:1** train/val/test.
- **Model**: DFNet (Conv1D stack + FC layers).
- **Training**: Adamax, lr 0.002, batch size 128, \(\sim\)30 epochs for NoDef.
- **Scale**: paper reports on the order of **1,000 traces per class** (e.g., 95k total for 95 classes).

---

## 2) Datasets we evaluated

We evaluated two fundamentally different “types” of datasets:

### 2.1 Curated benchmark pickles (from `modern_defenses.zip`)

We were given a zip that contained **pre-split DF-format pickles** under:

- `dataset/ClosedWorld/{NoDef,BRO,BuFLO,RegulaTor,Tamaraw,WalkieTalkie}/`
- `dataset/OpenWorld/{NoDef,BRO,BuFLO,RegulaTor,Tamaraw}/`

These are “benchmark-like”: consistent formatting, clean splits, and defense variants.

### 2.2 Our crawled dataset (local + GCP)

We crawled Tor traffic ourselves and then built DF-format pickles from pcaps.

- Raw pcaps live under `/mnt/d/cs244c-data*/.../crawler-pcap/`
- Combined DF-format pickles live under `/mnt/d/cs244c-combined/` (and variants)

Important: although we aimed to match paper methodology, our crawls are **much noisier and more heterogeneous** (modern web behavior, time variance, multi-machine conditions, partial loads, etc.).

---

## 3) Closed‑world + open‑world results on `modern_defenses.zip`

We trained/evaluated DFNet on each defense (closed + open where available).

Source of truth: `results/modern_defenses_run_20260309_234617/summary.csv`.

### 3.1 Closed‑world (test accuracy)

| Defense | Closed-world test top‑1 | Notes |
|---|---:|---|
| NoDef | **0.961** | Very high, in the “paper-like” regime |
| RegulaTor | **0.818** | Still fairly learnable |
| BRO | **0.741** | Still learnable |
| WalkieTalkie | **0.461** | Top‑1 lower due to decoys |
| WalkieTalkie | **0.737** (top‑2) | Top‑2 > top‑1 matches the W‑T decoy interpretation |
| BuFLO | **0.317** | Strong padding defense, DF struggles |
| Tamaraw | **0.261** | Strong padding defense, DF struggles |

### 3.2 Open‑world (metrics @ threshold=0.50)

Open‑world is not a single “accuracy”; our implementation reports TPR/FPR/Precision/Recall across thresholds and we summarize at 0.50.

| Defense | TPR | FPR | Precision | Recall |
|---|---:|---:|---:|---:|
| NoDef | **0.9895** | **0.0010** | **0.9989** | **0.9895** |
| RegulaTor | **0.8189** | **0.0080** | **0.9898** | **0.8189** |
| BRO | **0.7284** | **0.0680** | **0.9105** | **0.7284** |
| BuFLO | **0.1432** | **0.0460** | **0.7473** | **0.1432** |
| Tamaraw | **0.0747** | **0.0250** | **0.7396** | **0.0747** |

**Interpretation:** These results are broadly consistent with the literature: NoDef is easiest; padding defenses (BuFLO/Tamaraw) degrade performance; WalkieTalkie’s decoy mechanism is visible in top‑2 behavior (closed‑world).

---

## 4) Why our crawled NoDef does not reach paper-like accuracy

Even though we used DF-style preprocessing (direction-only, length 5000) and DFNet, our crawled dataset differs strongly from curated benchmark distributions:

### 4.1 Trace-length distribution dominates

The most important difference we measured is **non-padding length**.

For `modern_defenses` NoDef (curated):

- median nonpad length is **~4,000**
- many traces hit the 5000 cap

For our crawled data (example: GCP-only Combined before strict filtering):

- median nonpad length is **~400**
- a very large fraction of traces are “short” even after the default `<50 packets` filter

This creates a large signal gap: extremely short traces contain far less fingerprinting information.

### 4.2 Heterogeneity hurts

Our crawled dataset mixes:

- time variance (runs spread out over hours/days)
- circuit/guard variance
- page variance (modern dynamic content, consent screens, A/B tests, ads/CDNs)
- multi-machine/network conditions (local vs GCP)

Curated datasets tend to control these factors more aggressively.

### 4.3 Random splitting without stratification can produce unstable evaluation

In some earlier crawled runs, random splits resulted in some classes being weakly represented or even absent in a split, producing brittle per-class accuracy.

We updated our filtering+resplitting procedure to be **stratified per class**.

---

## 5) Experiments on our crawled combined dataset

We used `/mnt/d/cs244c-combined` as the base because it has the most total traces.

### 5.1 “Sweet spot” filter (95-class, balanced-ish) — Option B

We attempted to maximize accuracy while preserving the full 95-class task by:

- filtering to reduce short/noisy traces, and
- capping per-class to reduce imbalance, and
- stratified splitting.

One representative configuration:

- `min_nonpad = 600`
- `min_per_class = 50`
- `cap_per_class = 300`

On one run (after recombining with additional GCP data), this yielded:

- **95 classes**, 20,484 train / 2,562 valid / 2,557 test
- **Test accuracy: 0.3887**

**Interpretation:** even with substantial cleaning, the full 95‑class crawled dataset remains far below curated benchmark accuracy.

### 5.2 Max‑accuracy subset experiment (evidence of real signal)

To answer “is there *any* fingerprinting signal in our own crawled data?”, we ran a deliberately “max accuracy” experiment:

- keep only **long traces**: `min_nonpad = 1500`
- require enough support: drop classes with `<200` samples
- keep only the **top‑K** most represented classes: `keep_top_k_classes = 30`
- cap per class: `cap_per_class = 300`

This produced:

- **30 classes**, 6,742 train / 842 valid / 845 test
- **Test accuracy: 0.5988**
- per-class accuracy mean ~0.60 (min ~0.33, max ~0.93)

**Conclusion:** When we focus on classes where we have enough high-quality data (longer traces, sufficient samples), DFNet accuracy rises substantially. This supports the claim that **our collected data does contain meaningful WF signal**, even if the full 95‑class setting is noisy/hard.

---

## 6) Main conclusion (for the paper)

1. **Paper-style results are reproducible on curated benchmark pickles.**
   - On `modern_defenses` NoDef closed‑world, DFNet reaches **0.961** test accuracy, comparable to the DF paper’s well-known high-accuracy regime.

2. **Our independently crawled Tor data is much harder in the full 95-class setting.**
   - Even after aggressive filtering and balancing, our best 95-class runs remained in the \(\sim\)0.39–0.48 range depending on selection.

3. **There is real signal in our crawled data.**
   - When we restrict to a “max accuracy” subset (long traces + top classes), test accuracy jumps to **0.5988**.

4. **The discrepancy is largely explained by dataset quality and distribution shift**, especially:
   - short/partial traces dominating crawled data
   - higher heterogeneity (time, circuits, dynamic content, multi-network sources)

This supports a nuanced verification claim:

> DFNet behaves as expected on benchmark-like DF-formatted datasets, and our own crawled data contains WF signal; however, achieving paper-level performance in a large 95-class setting appears to require **much more controlled collection** and/or **much higher per-class yields of long, consistent traces** than we currently have.

---

## 7) Figure (paper-style) we generated from our results

We generated a closed‑world accuracy comparison bar chart at:

- `experiments/fig_closed_world_accuracy_ours_vs_modern.png`

This emulates the paper’s style of comparing closed‑world performance across conditions (defenses / dataset variants), but uses our measured results.

---

## 8) Training-convergence figures (our data)

To emulate the paper’s “training convergence” style figure using our own runs, we generated:

### 8.1 Max-accuracy subset (crawled, 30-class)

- Accuracy curve: `experiments/fig_training_curve_maxacc.png`
- Loss curve: `experiments/fig_loss_curve_maxacc.png`

This shows clean convergence and substantially higher validation accuracy when we restrict to long traces and well-supported classes.

### 8.2 Crawled 95-class (cleaned)

- Accuracy curve: `experiments/fig_training_curve_crawled95.png`
- Loss curve: `experiments/fig_loss_curve_crawled95.png`

This shows a much lower validation plateau in the full 95-class setting, consistent with our conclusion that dataset heterogeneity + short/partial traces limit achievable accuracy.

