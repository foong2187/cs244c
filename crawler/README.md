# Crawler: client-side traces for DF-style training

**First-time setup:** see **[SETUP.md](SETUP.md)** for installing Tor, Firefox, geckodriver, tcpdump, and running the crawler.

This package collects **client-side Tor traffic** (one page load per trace), captures packets between the Tor client and the **entry guard**, and converts them to **direction sequences** (+1 outgoing, -1 incoming, 5000 length) for training the Deep Fingerprinting model. The goal is to get data closer to the CCS'18 paper setting (entry-guard vantage, controlled visits) so you can aim for ~98% closed-world accuracy.

## How it works

1. **Tor** runs with its control port enabled. You visit a URL through Tor (SOCKS) using a **headless Firefox** (Selenium).
2. While the page loads, **tcpdump** records TCP traffic (Tor OR ports 443/9001).
3. After the load, we ask Tor (via **stem**) for the **entry guard IP** of the circuit that was used.
4. We filter the pcap to only packets to/from that guard, sort by time, and map each packet to **+1** (client→guard) or **-1** (guard→client). The sequence is truncated or zero-padded to **5000** (DF input length).
5. Each trace is saved as a pickle: `{"label": site_id, "sequence": np.array}`. Later, **merge_traces.py** combines them into train/valid/test pickles in the format expected by `train_yousef_closed_world.py`.

## Prerequisites

- **Tor** installed and running with control port (e.g. `ControlPort 9051` in `torrc`, or start Tor with stem).
- **Firefox** and **geckodriver** (for Selenium). Install geckodriver and ensure it’s on `PATH`.
- **tcpdump** (packet capture). On Linux you need **root** or `cap_net_raw` (e.g. `sudo` for the crawl).
- **Python 3** with: `stem`, `selenium`, `dpkt`, `numpy`.

### Install Python deps

From the repo root:

```bash
pip install stem selenium dpkt numpy
```

### Tor with control port

If you use system Tor, add to `/etc/tor/torrc` (or `~/.torrc`):

```
ControlPort 9051
CookieAuthentication 0
```

Then restart Tor. If you don’t have password set, stem will connect without a password.

## Usage

### 1. Run the crawler

Capture is done with tcpdump, so you need **root** (or cap_net_raw) on the crawl process:

```bash
cd /path/to/cs244c
sudo python -m crawler.crawl --site_list data/yousef-data/site_list.txt --visits 5
```

Options:

- `--visits N` – number of traces (page loads) per site (default 2).
- `--sites N` – only crawl the first N sites (e.g. `--sites 10` for a quick test).
- `--output_dir` – where to write per-trace pickles (default: `data/crawler-traces`).
- `--pcap_dir` – where to write raw pcaps (default: `data/crawler-pcap`).
- `--new_circuit_per_visit` – send NEWNYM before each visit for a fresh circuit.
- `--timeout` – page load timeout in seconds (default 45).

Example (small test):

```bash
sudo python -m crawler.crawl --sites 3 --visits 2 --new_circuit_per_visit
```

### 2. Merge traces into DF-format pickles

After you have enough traces (e.g. hundreds or thousands), merge them into train/valid/test arrays:

```bash
python -m crawler.merge_traces --input_dir data/crawler-traces --output_dir data/yousef-data/pickle
```

This writes `X_train_Yousef.pkl`, `y_train_Yousef.pkl`, and the same for `valid` and `test`, with an 80/10/10 split. You can point the existing Yousef training script at that directory:

```bash
cd src
python train_yousef_closed_world.py --data_dir ../data/yousef-data/pickle --epochs 30 --save_model
```

## Site list format

Plain text, one line per site: `label<TAB>url`. Lines starting with `#` are skipped. Example:

```
0	https://www.google.com
1	https://www.youtube.com
```

The default path is `data/yousef-data/site_list.txt` (Alexa Top 100 style).

## Notes

- **Guard IP**: If Tor doesn’t report a built circuit yet, we can’t filter by guard; the script still writes a trace (often mostly zeros). Using `--new_circuit_per_visit` and a short delay helps ensure a circuit is built for that load.
- **Noise**: Blocking images (as in the browser module) reduces noise; you can change preferences in `browser.py` (e.g. allow images) to match the paper more closely.
- **Ethics / ToS**: Crawl only sites you’re allowed to access; respect robots.txt and rate limits. Use for research in line with the CCS’18 and Tor project guidelines.
