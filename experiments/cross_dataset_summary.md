# Cross-Dataset Experiment: Benchmark Train → Crawled Eval

Run: 2026-03-13T18:14:15.139998

## Setup
- Train/valid source: `/home/mswisher/cs244c/dataset/ClosedWorld/NoDef` (NoDef, classes 0-94)
- Benchmark test: same source (sanity check)
- Crawled test: `/mnt/d/cs244c-cell-level`
- Classes: 95
- Epochs: 30, batch: 128, lr: 0.002

## Results
- **Benchmark test accuracy** (in-distribution): **0.9789**
- **Crawled test accuracy** (cross-dataset): **0.0124**

## Interpretation
The gap between in-distribution and cross-dataset accuracy quantifies
how much of the benchmark's high accuracy is due to distribution-specific
patterns vs. genuinely generalizable website fingerprints.


---

# Reverse Cross-Dataset: Crawled Train → Benchmark Eval

Run: 2026-03-13T18:23:58.305039

## Setup
- Train/valid source: `/mnt/d/cs244c-cell-level` (our crawled traces)
- Crawled test: same source (in-distribution sanity)
- Benchmark test: `/home/mswisher/cs244c/dataset/ClosedWorld/NoDef` (classes 0-94)
- Classes: 95
- Epochs: 30, batch: 128, lr: 0.002

## Results
- **Crawled test accuracy** (in-distribution): **0.4744**
- **Benchmark test accuracy** (cross-dataset): **0.0137**

## Per-class accuracy on benchmark (top 5)
- class 61: 0.4000 (4/10)
- class 86: 0.2000 (2/10)
- class 90: 0.2000 (2/10)
- class 67: 0.1000 (1/10)
- class 0: 0.1000 (1/10)
