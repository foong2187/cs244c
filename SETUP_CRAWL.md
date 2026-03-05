# Data Collection Setup Guide

Collect 95,000 website fingerprinting traces (95 sites x 1,000 visits) matching the CCS'18 Deep Fingerprinting paper. Split across multiple machines for speed.

## Quick Start (any fresh Ubuntu 22.04+ machine)

```bash
git clone https://github.com/foong2187/cs244c.git
cd cs244c
git checkout miro-impl
sudo bash crawler/setup_vm.sh
```

That installs everything: Tor, Firefox, geckodriver, tcpdump, Python deps, and the site list.

## Machine Assignments

Each machine crawls a **non-overlapping visit range** so pcaps don't collide when merged.

| Machine | Workers | Visit range | Visits/site | Command |
|---------|---------|-------------|-------------|---------|
| Dorm PC (WSL2) | 15 | 0-349 | 350 | See below |
| Ohio Proxmox | 10 | 350-649 | 300 | See below |
| GCE VM | 30 | 650-999 | 350 | See below |

### Dorm PC

```bash
sudo nohup env PYTHONUNBUFFERED=1 .venv/bin/python -m crawler.crawl_parallel \
  --workers 15 --sites 95 --visits 350 --visit-start 0 > crawl.log 2>&1 &
```

### Ohio Proxmox

```bash
# Find your network interface first
ip route | grep default
# Then run (replace ens18 with your interface):
sudo nohup env PYTHONUNBUFFERED=1 .venv/bin/python -m crawler.crawl_parallel \
  --workers 10 --sites 95 --visits 300 --visit-start 350 \
  --interface ens18 > crawl.log 2>&1 &
```

### GCE VM

```bash
# Create VM
gcloud compute instances create wf-crawler \
  --zone=us-central1-a --machine-type=e2-standard-16 \
  --provisioning-model=SPOT --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud --boot-disk-size=100GB

# SSH in
gcloud compute ssh wf-crawler --zone=us-central1-a

# Setup
git clone https://github.com/foong2187/cs244c.git && cd cs244c && git checkout miro-impl
sudo bash crawler/setup_vm.sh

# Crawl
sudo nohup env PYTHONUNBUFFERED=1 .venv/bin/python -m crawler.crawl_parallel \
  --workers 30 --sites 95 --visits 350 --visit-start 650 \
  --interface ens4 > crawl.log 2>&1 &
```

## Monitoring

```bash
# Live output
tail -f crawl.log

# Count successful traces
grep ',ok,' data/crawler-traces/progress.csv | wc -l

# Check processes are alive
ps aux | grep crawl_parallel | grep -v grep
```

## After Collection

From the dorm PC (or whichever machine has the GPU for training):

```bash
# 1. Copy data from other machines (edit crawler/merge_machines.sh with your SSH hosts)
bash crawler/merge_machines.sh

# Or manually:
scp ohio:~/cs244c/data/crawler-traces/progress.csv data/ohio-progress.csv
rsync -az ohio:~/cs244c/data/crawler-pcap/ data/crawler-pcap/

gcloud compute scp wf-crawler:~/cs244c/data/crawler-traces/progress.csv data/gce-progress.csv --zone=us-central1-a
gcloud compute scp --recurse "wf-crawler:~/cs244c/data/crawler-pcap/*" data/crawler-pcap/ --zone=us-central1-a

# Merge progress CSVs
head -1 data/crawler-traces/progress.csv > data/merged-progress.csv
tail -n+2 data/crawler-traces/progress.csv >> data/merged-progress.csv
tail -n+2 data/ohio-progress.csv >> data/merged-progress.csv
tail -n+2 data/gce-progress.csv >> data/merged-progress.csv
cp data/merged-progress.csv data/crawler-traces/progress.csv

# 2. Process pcaps into train/val/test pickles
.venv/bin/python -m crawler.process

# 3. Check dataset quality
.venv/bin/python -m crawler.analyze

# 4. Train
.venv/bin/python src/train_closed_world.py --defense NoDef --epochs 30

# 5. Don't forget to delete the GCE VM!
gcloud compute instances delete wf-crawler --zone=us-central1-a
```

## Resumability

The crawl is fully resumable. If a machine crashes or you kill the process, just re-run the same command. It reads `progress.csv` and skips already-completed visits.

## Troubleshooting

- **"binary is not a Firefox executable"**: Snap Firefox issue. The setup script installs the real deb from Mozilla PPA to avoid this.
- **"Connection refused" on Tor control port**: Run `echo -e "\nControlPort 9051\nCookieAuthentication 0" | sudo tee -a /etc/tor/torrc && sudo systemctl restart tor@default`
- **geckodriver version warning**: Harmless. It still works.
- **len=0 traces**: Guard IP mismatch. The crawler retries automatically and skips traces with < 50 packets.
