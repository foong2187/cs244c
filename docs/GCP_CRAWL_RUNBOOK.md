# Run remaining crawler traces on GCP

You're doing **instances 350–649** (visit-start=350, visits=300) across 95 sites → 28,500 total. The crawler is resumable: it skips any (site_idx, instance) already marked `ok` in `progress.csv`. So you can run the remainder on a bigger GCP VM and merge data back later.

---

## 1. Create a GCP VM with enough RAM

Use a machine with **≥32 GB RAM** so 5–10 Firefox workers don't thrash.

Example (Ubuntu 22.04, 32 GB RAM):

```bash
# Replace PROJECT_ID and ZONE; pick a name
gcloud compute instances create crawl-vm \
  --project=PROJECT_ID \
  --zone=ZONE \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --maintenance-policy=MIGRATE
```

- **e2-standard-8**: 8 vCPU, 32 GB RAM (often enough for 8 workers).
- For 10 workers or extra headroom: **e2-standard-16** (16 vCPU, 64 GB) or **n2-standard-16**.

SSH in:

```bash
gcloud compute ssh crawl-vm --zone=ZONE --project=PROJECT_ID
```

---

## 2. On your current machine: push repo and progress to GCP

From your **local** (or current) machine, where the repo and existing progress live:

```bash
cd /path/to/cs244c

# Copy repo (exclude large dirs you don't need for crawling)
rsync -avz --exclude '.venv' --exclude 'dataset' --exclude '*.hdf5' \
  --exclude 'data/crawler-pcap' \
  . CRAWL_VM_USER@CRAWL_VM_IP:~/cs244c/

# Or use gcloud scp for the critical pieces:
# 1) progress.csv (so GCP resumes from current state)
gcloud compute scp data/crawler-traces/progress.csv crawl-vm:~/cs244c/data/crawler-traces/ --zone=ZONE
# 2) site list
gcloud compute scp data/yousef-data/site_list.txt crawl-vm:~/cs244c/data/yousef-data/ --zone=ZONE
```

If you prefer a full clone on GCP and only sync progress:

```bash
# On GCP you'll clone the repo (or rsync without data/). Then from local, push progress:
gcloud compute scp data/crawler-traces/progress.csv crawl-vm:~/cs244c/data/crawler-traces/ --zone=ZONE
```

Ensure on GCP you have:

- Repo (with `crawler/`, `data/yousef-data/site_list.txt`, etc.)
- `data/crawler-traces/progress.csv` (from this machine so it resumes)
- `data/crawler-traces/` and `data/crawler-pcap/` (create if missing: `mkdir -p data/crawler-traces data/crawler-pcap`)

---

## 3. On the GCP VM: one-time setup

SSH into the VM, then:

```bash
cd ~/cs244c

# Install deps (Tor, Firefox, geckodriver, tcpdump, Python venv)
sudo bash crawler/setup_vm.sh

# Python venv + crawler deps
python3 -m venv .venv
.venv/bin/pip install -r requirements-crawler.txt

# Geckodriver 0.36 for Firefox 148 (if not already)
./crawler/install_geckodriver.sh
export PATH="$HOME/cs244c/.local/bin:$PATH"

# Optional: earlyoom so the system doesn't lock up
sudo apt install -y earlyoom
sudo systemctl enable --now earlyoom
```

Verify:

```bash
.venv/bin/python -m crawler.verify
```

---

## 4. Run the crawler on GCP (same instance range 350–649)

Same parameters as your current run so progress lines up:

```bash
cd ~/cs244c
export PATH="$HOME/cs244c/.local/bin:$PATH"

# Stop anything left over, then start (5 workers default; use 8 if you have 32GB)
sudo ./crawler/restart_crawl.sh
# Or with more workers:
# CRAWLER_WORKERS=8 sudo ./crawler/restart_crawl.sh
```

The script uses **visit-start=350, visits=300** (instances 350–649). It will skip every (site_idx, instance) already in `progress.csv` and only run the rest.

Monitor:

```bash
tail -f ~/cs244c/crawl.log
grep -c ',ok,' ~/cs244c/data/crawler-traces/progress.csv
```

---

## 5. When the crawl is done: pull data back

From your **local** machine:

```bash
# Progress (updated)
gcloud compute scp crawl-vm:~/cs244c/data/crawler-traces/progress.csv \
  data/crawler-traces/progress.csv --zone=ZONE

# New pcaps (optional; can be large)
gcloud compute scp --recurse crawl-vm:~/cs244c/data/crawler-pcap/*.pcap \
  data/crawler-pcap/ --zone=ZONE

# Or rsync to merge with existing dirs
rsync -avz CRAWL_VM_USER@CRAWL_VM_IP:~/cs244c/data/crawler-traces/ data/crawler-traces/
rsync -avz CRAWL_VM_USER@CRAWL_VM_IP:~/cs244c/data/crawler-pcap/ data/crawler-pcap/
```

Then on the machine where you train (e.g. your 3080 box), run merge/processing as usual:

```bash
.venv/bin/python -m crawler.merge_traces --input_dir data/crawler-traces --output_dir data/yousef-data/pickle
# etc.
```

---

## Quick reference

| What | Value |
|------|--------|
| Instance range | **350–649** (visit-start=350, visits=300) |
| Sites | 95 |
| Total visits | 28,500 |
| Resumes from | `data/crawler-traces/progress.csv` (copy from current machine) |
| GCP VM | ≥32 GB RAM (e2-standard-8 or similar) |
| Command on GCP | `sudo ./crawler/restart_crawl.sh` (or `CRAWLER_WORKERS=8` if 32GB) |
