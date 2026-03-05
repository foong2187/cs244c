# Cross-Dataset Summary: DF Replication vs. CCS'18 Paper

This document compares closed-world and open-world results across GTT23, Fresh2026, and Yousef's dataset to the original Deep Fingerprinting (Sirinam et al., CCS 2018) paper.

## Closed-World Results

| Dataset    | Classes | Train size | Val acc (final) | Test acc | Notes                          |
|-----------|--------|------------|------------------|----------|---------------------------------|
| **GTT23** | 95     | 76,000     | ~53.3%           | —        | Exit-relay, passive; 30 epochs |
| **Fresh2026** | 92 | ~5,808     | ~16.8% (peak ~23%) | ~16.4% | Teammate dataset; 30 epochs    |
| **Yousef**| 100    | 6,284      | ~11.4%           | ~10.2%   | data/yousef-data/pickle; 30 epochs |
| **CCS'18 paper** | 95 | 800/class  | —                | **~98%** | Client-side entry-guard, controlled lab |

- All runs use the same DFNet architecture and paper-matched hyperparameters (Adamax lr=0.002, batch 128, 30 epochs).
- The large gap (98% vs 53% vs ~10–16%) is attributed mainly to **data regime**: paper used client-side entry-guard traces in a controlled setting; GTT23 is exit-relay passive; Fresh2026 and Yousef are separate Tor-like collections with different sizes and label sets.

## Open-World Results

| Dataset    | num_monitored | Test Mon | Test Unmon | Key thresholds (TPR / FPR)     | Notes        |
|-----------|----------------|----------|------------|----------------------------------|--------------|
| **Yousef**| 70             | 606      | 299        | 0.50: 0.0165 / 0.0100; 0.70: 0.0033 / 0.0000 | Pooled from same closed-world pickles |
| **GTT23** | 95             | —        | —          | See `results/OpenWorld_NoDef.csv` | Prebuilt open-world splits in dataset/ |
| **CCS'18**| —              | —        | —          | Paper Section 5.7               | Threshold-based TPR/FPR reported     |

- Yousef open-world is derived from the same 100-class closed-world data: first 70 classes = monitored, remaining 30 classes pooled as unmonitored. Low TPR at high thresholds indicates the model is conservative (few monitored traces classified above threshold).

## Interpretation

1. **Implementation fidelity:** The codebase matches the paper's architecture and training setup. The accuracy gap across datasets points to **data**, not implementation, as the main differentiator.
2. **Data quality and setting:** Higher accuracy on the paper's setup is consistent with cleaner, client-side entry-guard traces and controlled visits. GTT23 (exit-relay) and Yousef/Fresh2026 (different collection) show lower but non-random accuracy.
3. **Next steps for replication:** To approach 98%-style results you would need entry-guard (or equivalent client-side) traces and a similar site set; or continue tuning (more data, filtering, augmentation) on existing datasets and document the ceiling.

## Experiment Logs

- `01_closed_world_gtt23_baseline.md` — GTT23 closed-world
- `02_fresh2026_closed_world.md` — Fresh2026 closed-world
- `03_yousef_closed_world.md` — Yousef closed-world
- `04_yousef_open_world.md` — Yousef open-world (70 monitored)
