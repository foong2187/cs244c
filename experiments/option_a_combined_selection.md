# Option A (Recommended) — Combined crawler dataset selection

This note documents the **recommended “sweet spot” selection** for training DF on our **crawled NoDef data** while keeping enough samples/class for stable evaluation.

## Goal

We observed a large performance gap between:

- **Curated benchmark-style datasets** (e.g. `modern_defenses` pickles), which have many long traces and balanced splits, yielding high closed-world accuracy (e.g. ~96% for NoDef on that benchmark), and
- **Our crawled/combined datasets**, which contain many short/partial/noisy traces and heterogeneous sources, yielding much lower closed-world accuracy (~40–55% depending on selection).

The goal of Option A is to move our crawled dataset closer to the benchmark regime by:

- filtering out short traces that contain weak fingerprint signal, and
- keeping **many classes** with **enough examples per class** so test accuracy is meaningful.

## Data source

Use the largest and most robust source:

- **`/mnt/d/cs244c-combined`** (local + GCP merged).

This dominates the other single-source options:

- `/mnt/d/cs244c-gcp-only` (smaller)
- `/mnt/d/cs244c-local-only` (smaller)

## Why `/mnt/d/cs244c-combined` is best

Quick profiling across candidate datasets (all are DF-format, \(5000\)-length direction sequences):

- `cs244c-combined`: **62,693** total traces
- `cs244c-gcp-only`: **35,735** total traces
- `cs244c-local-only`: **26,958** total traces

Since filtering reduces data aggressively, starting from the largest pool gives us room to enforce quality without collapsing the dataset.

## Option A: filter + stratified resplit

### Filter settings

- **Minimum non-pad length**: `min_nonpad = 400`
  - Rationale: removes the weakest traces while retaining large volume.
- **Minimum samples per class** after filtering: `min_per_class = 50`
  - Rationale: prevents classes from disappearing in valid/test; stabilizes per-class accuracy.
- **Cap per class**: `cap_per_class = 300`
  - Rationale: reduces imbalance and avoids a small number of easy/overrepresented sites dominating training.

### Expected yield (from profiling)

On `cs244c-combined` with `nonpad >= 400`:

- **kept**: **31,656 traces** (~50.5% of 62,693)
- **classes alive**: **95**
- **classes with >= 50 samples**: **95**
- **classes with >= 100 samples**: **94**
- per-class median count (post-filter, pre-cap): **342**

This is exactly the “sweet spot” property we want: **high class coverage** with **healthy sample counts**.

### Commands

Create the filtered dataset:

```bash
/home/mswisher/cs244c/.venv/bin/python scripts/filter_resplit_combined.py \
  --data_dir /mnt/d/cs244c-combined \
  --out_dir /mnt/d/cs244c-combined-np400-min50-cap300 \
  --min_nonpad 400 \
  --min_per_class 50 \
  --cap_per_class 300
```

Train DF on the filtered dataset:

```bash
/home/mswisher/cs244c/.venv/bin/python src/train_combined.py \
  --data_dir /mnt/d/cs244c-combined-np400-min50-cap300 \
  --epochs 50
```

## Interpretation

Option A is not meant to “match the paper” exactly; it’s a practical, controlled selection that:

- reduces the short-trace tail that dominates crawled data,
- forces fairer/cleaner splits by ensuring sufficient per-class representation, and
- should yield a materially stronger and more stable closed-world baseline than training on the unfiltered combined data.

If accuracy is still far below the curated benchmark, the remaining gap is likely due to **true distribution differences** (modern site variability, heterogeneous circuits/guards/times, crawler noise) rather than purely preprocessing mistakes.

