# data-collection

Collects Tor traffic traces for the WF dataset (95 sites × N instances).

## Setup

Spin up a GCP VM (e2-standard-4, Ubuntu 22.04, 100GB disk), then:

```bash
gcloud compute ssh wf-collector --zone=us-central1-a
git clone https://github.com/foong2187/cs244c.git
cd cs244c/data-collection
sudo bash setup.sh
```

## Collection

```bash
# check your interface first: ip route | grep default
.venv/bin/python collect.py --instances 90 --interface ens4
```

Saves a pcap per visit to `data/pcap/`. Resumable — just re-run if it crashes.

Run inside tmux so it survives disconnect:
```bash
tmux new -s collect
.venv/bin/python collect.py --instances 90 --interface ens4
# Ctrl+B D to detach
```

## Processing

```bash
.venv/bin/python process.py
```

Outputs `data/pickle/X_{train,valid,test}_Fresh2026.pkl` and mirrors them to `dataset/ClosedWorld/NoDef/` for the model code.
