# Fresh Data Collection Plan for Concept Drift Analysis

## Purpose

The original Deep Fingerprinting dataset was collected in **2016-2018**. Websites have changed dramatically since then — different CDNs, updated TLS versions, new page structures, heavier JavaScript frameworks, and evolving Tor network characteristics. By collecting fresh traces of the **same websites** in 2026, we can directly measure how much a WF model trained on old data degrades over time ("concept drift"), which is critical for understanding the real-world viability of website fingerprinting attacks.

---

## What We're Measuring

**Concept drift experiment**:
1. Train the DF model on the original 2018 dataset (100 monitored sites)
2. Collect fresh 2026 traces of those same 100 sites
3. Evaluate the 2018-trained model on 2026 traces → measure accuracy degradation
4. Optionally: fine-tune the model on a small number of new traces (few-shot adaptation) to see how quickly it recovers

**Expected outcome**: Significant accuracy drop (potentially from ~98% down to 60-80%), demonstrating that WF models have a limited "shelf life" and require periodic retraining.

---

## Prerequisites

### Hardware
- Linux machine (VM or bare metal) — Ubuntu 22.04+ recommended
- Minimum 50 GB free disk space (pcap files are large)
- Stable internet connection (Tor routing adds latency; expect ~30-60s per page load)

### Software to Install

```bash
# System packages
sudo apt update
sudo apt install -y xvfb tcpdump tshark python3-pip

# Tor Browser Bundle
# Download from https://www.torproject.org/download/
# Extract to /opt/tor-browser/

# Python packages
pip install stem selenium tbselenium dpkt numpy
```

### Time Estimate

- 100 monitored sites x 90 instances = 9,000 visits
- ~60 seconds per visit (page load + overhead) + 10s NEWNYM cooldown = ~70s per visit
- **Total: ~175 hours of continuous collection** (~7 days)
- Recommendation: run in parallel across 2-3 VMs to reduce to ~2-3 days

---

## Step 1: Prepare the Website List

Use the **same 100 websites** from the original DF paper's closed-world experiment. The DF dataset labels sites 0-99 corresponding to the top 100 Alexa websites as of 2016.

**File to create**: `data/collected/site_list.txt`

```
# Top 100 Alexa sites (2016 list used in DF paper)
# Format: label\turl
0	https://www.google.com
1	https://www.youtube.com
2	https://www.facebook.com
3	https://www.baidu.com
4	https://www.wikipedia.org
...
99	https://www.example99.com
```

**Important**: Some 2016 Alexa-100 sites may no longer exist or may redirect. Document any changes:
- If a site redirected permanently, use the new URL but note the change
- If a site is completely gone, note it and exclude from the concept drift comparison
- This itself is a data point — website churn over 8 years

To find the original 2016 Alexa-100 list, check the DF paper's supplementary materials or the Wayback Machine's archived Alexa rankings.

---

## Step 2: Collection Environment Setup

### Tor Configuration (`torrc`)

Create a custom torrc to ensure clean circuit isolation:

```
# /opt/tor-browser/Browser/TorBrowser/Data/Tor/torrc

# Force fresh circuits for every connection
MaxCircuitDirtiness 1
# Disable predictive circuit building
LearnCircuitBuildTimeout 0
# Use a single entry guard for consistency (optional)
# NumEntryGuards 1
```

### Virtual Display (for headless servers)

```bash
# Start Xvfb for headless Tor Browser
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
```

---

## Step 3: Collection Script

**File**: `scripts/collect_traces.py`

The script orchestrates the full collection loop:

```
For each batch (1 through 90):
    Shuffle the site list (randomize visit order within batch)
    For each site in shuffled list:
        1. Signal NEWNYM via stem (fresh Tor circuit)
        2. Wait 10 seconds (NEWNYM rate limit)
        3. Start tcpdump capture → save to data/collected/pcap/{label}-{batch}.pcap
        4. Open URL in TorBrowserDriver
        5. Wait for page load (document.readyState == 'complete') + 5 seconds
        6. Stop tcpdump
        7. Close browser tab / restart browser
        8. Log: site label, batch number, timestamp, page load time, pcap file size
```

### Key parameters:
- **Batches**: 90 (gives 90 instances per site)
- **Timeout**: 60 seconds per page load (kill if not loaded)
- **Post-load wait**: 5 seconds after `readyState == 'complete'` to capture trailing requests
- **Capture filter**: `tcp and host <guard_ip>` (capture only Tor traffic, not other system traffic)

### Crash recovery:
- Log progress to `data/collected/progress.csv` after each visit
- On restart, skip already-completed `(site, batch)` pairs
- Retry failed visits up to 3 times before marking as failed

---

## Step 4: Pcap → Trace Conversion

**File**: `scripts/pcap_to_traces.py`

Convert each pcap file to the standard `timestamp\tdirection` format:

```python
import dpkt
import socket

def pcap_to_trace(pcap_path, guard_ip):
    """
    Convert a pcap file to a list of (timestamp, direction) tuples.

    Direction: +1 = outgoing (client → guard), -1 = incoming (guard → client)
    Timestamps: relative to first packet (seconds)
    """
    trace = []
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        first_ts = None

        for ts, buf in pcap:
            if first_ts is None:
                first_ts = ts

            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            ip = eth.data
            src = socket.inet_ntoa(ip.src)
            dst = socket.inet_ntoa(ip.dst)

            if dst == guard_ip:
                direction = 1    # outgoing
            elif src == guard_ip:
                direction = -1   # incoming
            else:
                continue         # not Tor traffic

            trace.append((ts - first_ts, direction))

    return trace
```

### Output format:

One file per trace at `data/collected/traces/{label}-{batch}`:
```
0.000	1
0.012	-1
0.013	-1
0.045	1
0.089	-1
...
```

---

## Step 5: Traces → DF-Compatible Pickle Format

**File**: `scripts/traces_to_pickle.py`

Convert the trace files into the same format as the original DF dataset:

```python
import numpy as np
import pickle
from pathlib import Path

def traces_to_pickle(traces_dir, output_dir, sequence_length=5000):
    """
    Convert trace files to DF-compatible pickle format.

    Input:  data/collected/traces/{label}-{batch} files
    Output: X_train.pkl, y_train.pkl, X_valid.pkl, y_valid.pkl, X_test.pkl, y_test.pkl

    Split: 80% train, 10% validation, 10% test
    """
    traces_dir = Path(traces_dir)
    all_traces = []
    all_labels = []

    for trace_file in sorted(traces_dir.iterdir()):
        name = trace_file.name  # e.g., "42-7" → site 42, batch 7
        label = int(name.split('-')[0])

        # Read trace, extract direction-only sequence
        directions = []
        with open(trace_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                directions.append(float(parts[1]))

        # Pad or truncate to sequence_length
        if len(directions) >= sequence_length:
            directions = directions[:sequence_length]
        else:
            directions += [0.0] * (sequence_length - len(directions))

        all_traces.append(directions)
        all_labels.append(label)

    X = np.array(all_traces, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Shuffle and split
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    n = len(X)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': (X[:train_end], y[:train_end]),
        'valid': (X[train_end:valid_end], y[train_end:valid_end]),
        'test':  (X[valid_end:], y[valid_end:]),
    }

    for name, (X_split, y_split) in splits.items():
        with open(output_dir / f'X_{name}_Fresh2026.pkl', 'wb') as f:
            pickle.dump(X_split, f)
        with open(output_dir / f'y_{name}_Fresh2026.pkl', 'wb') as f:
            pickle.dump(y_split, f)
        print(f"{name}: {len(X_split)} samples")
```

---

## Step 6: Quality Validation

After collection, run validation checks before using the data:

### A. Completeness Check
```python
# Verify we have ~90 instances for each of the 100 sites
from collections import Counter
counts = Counter(y)
for label in range(100):
    print(f"Site {label}: {counts.get(label, 0)} instances")
# Flag any site with < 80 instances
```

### B. Trace Length Distribution
```python
# Check that traces are reasonable (not too short = failed loads)
lengths = [(trace != 0).sum() for trace in X]
print(f"Mean non-zero length: {np.mean(lengths):.0f}")
print(f"Min: {np.min(lengths)}, Max: {np.max(lengths)}")
print(f"Traces < 50 packets (likely failed): {sum(1 for l in lengths if l < 50)}")
# Remove traces shorter than 50 packets
```

### C. Direction Balance
```python
# Each trace should have both incoming and outgoing packets
for i, trace in enumerate(X):
    real = trace[trace != 0]
    out_frac = (real > 0).mean()
    if out_frac < 0.05 or out_frac > 0.95:
        print(f"Warning: trace {i} has {out_frac:.0%} outgoing — possibly malformed")
```

### D. Visual Spot Check
```python
# Plot a few traces to visually verify they look like web page loads
import matplotlib.pyplot as plt

for site in [0, 25, 50, 75, 99]:
    idx = np.where(y == site)[0][0]
    trace = X[idx]
    real = trace[trace != 0]
    plt.figure(figsize=(12, 2))
    plt.bar(range(len(real)), real, width=1.0, color=['blue' if d > 0 else 'red' for d in real])
    plt.title(f"Site {site} — {len(real)} packets")
    plt.savefig(f"results/figures/trace_site{site}.png")
```

---

## Step 7: Running the Concept Drift Experiment

Once fresh data is collected and validated:

```bash
# 1. Evaluate the 2018-trained model on fresh 2026 data
python scripts/evaluate.py \
    --config configs/closed_world_nodef.yaml \
    --checkpoint results/checkpoints/closed_world_nodef_best.pt \
    --test-data data/collected/pickle/X_test_Fresh2026.pkl \
    --test-labels data/collected/pickle/y_test_Fresh2026.pkl

# 2. Fine-tune with few-shot adaptation (e.g., 5 instances per site)
python scripts/train.py \
    --config configs/closed_world_fresh2026.yaml \
    --pretrained results/checkpoints/closed_world_nodef_best.pt \
    --few-shot 5

# 3. Generate comparison visualizations
python scripts/evaluate.py \
    --config configs/closed_world_fresh2026.yaml \
    --compare-with results/metrics/closed_world_nodef.csv
```

### Expected Results

| Experiment | Expected Accuracy | What It Shows |
|---|---|---|
| 2018 model on 2018 data | ~98.3% | Baseline (replication) |
| 2018 model on 2026 data | ~60-80% | Concept drift magnitude |
| Fine-tuned (5-shot) on 2026 data | ~85-90% | Few-shot adaptation viability |
| Retrained from scratch on 2026 data | ~95-98% | Fresh model upper bound |

---

## Open-World Extension (Optional)

For open-world concept drift analysis, also collect unmonitored traces:

- **9,000 random sites** (1 visit each) from a current Tranco list
- These serve as the unmonitored class in the open-world scenario
- Compare: model trained on 2018 unmonitored sites vs. tested on 2026 unmonitored sites

---

## File Organization Summary

```
data/collected/
├── site_list.txt                # 100 monitored URLs (one per line)
├── progress.csv                 # Collection progress log
├── pcap/                        # Raw packet captures
│   ├── 0-0.pcap                 # Site 0, batch 0
│   ├── 0-1.pcap                 # Site 0, batch 1
│   └── ...
├── traces/                      # Converted timestamp+direction files
│   ├── 0-0                      # Site 0, batch 0
│   ├── 0-1
│   └── ...
└── pickle/                      # DF-compatible pickle format
    ├── X_train_Fresh2026.pkl
    ├── y_train_Fresh2026.pkl
    ├── X_valid_Fresh2026.pkl
    ├── y_valid_Fresh2026.pkl
    ├── X_test_Fresh2026.pkl
    └── y_test_Fresh2026.pkl
```

---

## Pitfalls & Mitigations

| Pitfall | Impact | Mitigation |
|---|---|---|
| Browser cache between visits | Artificially small repeat traces | Clear profile after every visit |
| Connection leakage between tabs | Contaminated captures | Full browser restart between visits |
| Tor circuit reuse | Correlated traces | `NEWNYM` + `MaxCircuitDirtiness 1` |
| Temporal ordering artifacts | Model learns time-of-day, not site | Shuffle site order within each batch |
| Failed page loads | Short/empty traces | Discard traces < 50 packets, retry up to 3x |
| Sites that no longer exist | Missing data points | Document changes, exclude from comparison |
| IP-level vs cell-level capture | Different granularity than DF dataset | DF paper uses cell-level; use Tor's control port or filter by fixed-size cells (512 bytes) |
| Guard IP changes mid-collection | Missed packets | Re-detect guard IP at start of each batch |

---

## References

- [WFP-Collector](https://github.com/irsyadpage/WFP-Collector) — Modern collection framework (2023)
- [tor-browser-selenium](https://github.com/webfp/tor-browser-selenium) — Standard Tor Browser automation
- [tor-browser-crawler](https://github.com/webfp/tor-browser-crawler) — Classic WF trace crawler (archived)
- GTT23 paper (Jansen et al., 2024) — Demonstrates synthetic datasets overestimate WF accuracy vs genuine traces
- NetCLR (Bahramali et al., CCS 2023) — Shows concept drift resilience via contrastive pre-training
