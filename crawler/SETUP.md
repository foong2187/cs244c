# Crawler setup (from scratch)

Follow these steps **in your terminal** (you'll need to enter your sudo password).

## 1. Run the automated setup script

From the repo root:

```bash
cd /home/mswisher/cs244c
./crawler/setup.sh
```

This will:

- Install: `apt-transport-https`, `gnupg`, `tcpdump`, `firefox`
- Add the official Tor Project APT repo and install `tor`
- Configure Tor to listen on control port **9051** with no password
- Start (or restart) the Tor service
- Download **geckodriver** into `cs244c/.local/bin/`
- Install Python deps: `stem`, `selenium`, `dpkt`, `numpy` (into `.venv` if present)

If the script fails partway (e.g. Tor repo), fix the reported issue and re-run; already-done steps are mostly idempotent.

## 2. Put geckodriver on your PATH

The crawler needs `geckodriver` on `PATH` so Selenium can drive Firefox. Either run:

```bash
export PATH="/home/mswisher/cs244c/.local/bin:$PATH"
```

for the current shell, or add that line to your `~/.bashrc` and run `source ~/.bashrc`.

## 3. Verify everything

From the repo root, using the same environment where you'll run the crawler (and with the PATH above):

```bash
cd /home/mswisher/cs244c
export PATH="/home/mswisher/cs244c/.local/bin:$PATH"
.venv/bin/python -m crawler.verify
```

You should see:

- Python packages OK
- Tor control port connected, SOCKS port shown
- geckodriver found
- Optional: one page load via Tor OK

If Tor fails with **"Connection refused" on 9051**: On Ubuntu the running process is **tor@default**, which reads **/etc/tor/torrc** (it does *not* load files in torrc.d). Add the control port there and restart:

```bash
echo '' | sudo tee -a /etc/tor/torrc
echo 'ControlPort 9051' | sudo tee -a /etc/tor/torrc
echo 'CookieAuthentication 0' | sudo tee -a /etc/tor/torrc
sudo systemctl restart tor@default
```

Then run `python -m crawler.verify` again.

To check status:

```bash
sudo systemctl status tor@default
ss -tlnp | grep 9051
```

## WSL / no display

On WSL (or any environment without a graphical display), Firefox/Selenium often fails. Use the **requests** fallback instead:

```bash
export CRAWLER_USE_REQUESTS=1
# or when crawling:
sudo CRAWLER_USE_REQUESTS=1 .venv/bin/python -m crawler.crawl --sites 1 --visits 1 --use-requests
```

The verify script uses requests by default, so step 4 should pass.

## 4. Run a tiny test crawl (needs sudo for tcpdump)

One site, one visit, to confirm capture works:

```bash
cd /home/mswisher/cs244c
export PATH="/home/mswisher/cs244c/.local/bin:$PATH"
sudo .venv/bin/python -m crawler.crawl --sites 1 --visits 1
```

Output and trace pickles will go to `data/crawler-traces/` and `data/crawler-pcap/`. If that works, run a larger crawl, e.g.:

```bash
sudo .venv/bin/python -m crawler.crawl --sites 5 --visits 2 --new_circuit_per_visit
```

## 5. Merge traces and train

After you have enough traces:

```bash
.venv/bin/python -m crawler.merge_traces --input_dir data/crawler-traces --output_dir data/yousef-data/pickle
cd src && ../.venv/bin/python train_yousef_closed_world.py --data_dir ../data/yousef-data/pickle --epochs 30 --save_model
```

---

## If you prefer to install step-by-step (no script)

1. **Tor (with control port)**  
   - Install: [Tor Project APT repo](https://support.torproject.org/apt/tor-deb-repo/) then `sudo apt install tor deb.torproject.org-keyring`.  
   - Create `/etc/tor/torrc.d/50-control.conf` with:
     ```
     ControlPort 9051
     CookieAuthentication 0
     ```
   - Start Tor: `sudo systemctl start tor`.

2. **System packages**  
   `sudo apt install tcpdump firefox`

3. **geckodriver**  
   Download from [geckodriver releases](https://github.com/mozilla/geckodriver/releases) (e.g. `geckodriver-v0.36.0-linux64.tar.gz`), extract, put `geckodriver` in a directory on your `PATH` (e.g. `~/.local/bin` or `/home/mswisher/cs244c/.local/bin`).

4. **Python**  
   `pip install -r requirements-crawler.txt` (or use the project `.venv`).

Then run steps 3–5 above.
