#!/usr/bin/env python3
# collect.py - visit sites over Tor and capture pcaps for WF dataset
# run setup.sh first, then: .venv/bin/python collect.py

import argparse
import csv
import logging
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import stem
from stem import Signal
from stem.control import Controller
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from tqdm import tqdm

SITES_FILE = Path(__file__).parent / "sites.txt"

# These are set after args are parsed (depend on --user-id and --data-dir)
PCAP_DIR      : Path
PROGRESS_FILE : Path
LOG_FILE      : Path

TOR_SOCKS_PORT   = 9050

TOR_CONTROL_PORT = 9051
TOR_CONTROL_PASS = "cs244c_collection"

PAGE_TIMEOUT     = 60   # seconds to wait for full page load
POST_LOAD_WAIT   = 5    # seconds after readyState==complete
NEWNYM_WAIT      = 5    # seconds to wait after requesting new circuit
MAX_RETRIES      = 2    # retries per (site, instance) before skipping

log = logging.getLogger(__name__)



def load_progress() -> set:
    """Return set of already-completed (site_idx, instance) tuples."""
    done = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((int(row["site_idx"]), int(row["instance"])))
    return done


def log_progress(site_idx: int, instance: int, url: str,
                 status: str, pcap_size: int, elapsed: float,
                 guard_ip: str = ""):
    write_header = not PROGRESS_FILE.exists()
    with open(PROGRESS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "site_idx", "instance", "url",
            "status", "pcap_bytes", "elapsed_s", "guard_ip"
        ])
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "site_idx":  site_idx,
            "instance":  instance,
            "url":       url,
            "status":    status,
            "pcap_bytes": pcap_size,
            "elapsed_s": round(elapsed, 2),
            "guard_ip":  guard_ip,
        })



def get_guard_ip(controller) -> str:
    """Return the IP of the current Tor entry guard, retrying up to 60s."""
    for _ in range(60):
        try:
            for circuit in controller.get_circuits():
                if circuit.status == "BUILT" and circuit.path:
                    fp = circuit.path[0][0]
                    ns = controller.get_network_status(fp)
                    return ns.address
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Could not determine Tor guard IP after 60s")


def new_circuit(controller):
    """Request a fresh Tor circuit and wait for it to build."""
    controller.signal(Signal.NEWNYM)
    time.sleep(NEWNYM_WAIT)



def make_driver() -> webdriver.Firefox:
    """Return a headless Firefox instance routing traffic through Tor."""
    opts = Options()
    opts.add_argument("--headless")

    # Route all traffic through Tor's SOCKS5 proxy
    opts.set_preference("network.proxy.type", 1)
    opts.set_preference("network.proxy.socks", "127.0.0.1")
    opts.set_preference("network.proxy.socks_port", TOR_SOCKS_PORT)
    opts.set_preference("network.proxy.socks_version", 5)
    opts.set_preference("network.proxy.socks_remote_dns", True)

    # Disable telemetry, updates, safe-browsing (reduces background traffic)
    opts.set_preference("datareporting.healthreport.uploadEnabled", False)
    opts.set_preference("app.update.enabled", False)
    opts.set_preference("browser.safebrowsing.enabled", False)
    opts.set_preference("browser.safebrowsing.malware.enabled", False)
    opts.set_preference("privacy.trackingprotection.enabled", False)

    service = Service(log_path=os.devnull)
    return webdriver.Firefox(options=opts, service=service)


def visit_page(driver, url: str) -> bool:
    """
    Navigate to url and wait for the page to fully load.
    Returns True on success, False on timeout/error.
    """
    try:
        driver.set_page_load_timeout(PAGE_TIMEOUT)
        driver.get(f"https://{url}")
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(POST_LOAD_WAIT)
        return True
    except TimeoutException:
        log.warning(f"  Timeout loading {url}")
        return False
    except WebDriverException as e:
        log.warning(f"  WebDriver error on {url}: {e.msg[:80]}")
        return False



def start_capture(pcap_path: Path, guard_ip: str, iface: str) -> subprocess.Popen:
    """Start tcpdump filtering to/from the Tor guard IP."""
    cmd = [
        "sudo", "tcpdump",
        "-i", iface,
        "-w", str(pcap_path),
        "-q",
        f"host {guard_ip}",
    ]
    return subprocess.Popen(cmd, stderr=subprocess.DEVNULL)


def stop_capture(proc: subprocess.Popen):
    """Terminate tcpdump and wait for it to flush."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", type=int, default=200,
                   help="Number of traces per site (default: 200)")
    p.add_argument("--interface", type=str, default="ens4",
                   help="Network interface for tcpdump (default: ens4). "
                        "Run 'ip route' to find yours.")
    p.add_argument("--time-limit", type=int, default=None,
                   help="Stop after this many minutes (default: no limit). "
                        "Progress is saved and the run can be resumed.")
    p.add_argument("--max-sites", type=int, default=None,
                   help="Only collect from the first N sites (default: all). "
                        "Useful for quick smoke tests.")
    p.add_argument("--user-id", type=str, default="user-000",
                   help="Unique identifier for this collector pod. Output is "
                        "written to /app/data/<user-id>/pcap/ so parallel pods "
                        "do not collide. (default: user-000)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Base data directory (default: <script_dir>/data). "
                        "Inside k8s pods this is typically /app/data.")
    p.add_argument("--gcs-bucket", type=str, default=None,
                   help="Unused. Raw pcaps are kept on node disk; only "
                        "process.py and analyze.py push outputs to GCS.")
    p.add_argument("--gcs-prefix", type=str, default="",
                   help="Unused by collect.py. Kept for CLI compatibility.")
    p.add_argument("--shard-index", type=int, default=0,
                   help="0-based index of this pod's shard (default: 0). "
                        "Use with --num-shards to split sites across pods.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards/pods (default: 1 = no sharding).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve paths based on --data-dir and --user-id ─────────────────────
    base_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    user_dir = base_dir / args.user_id

    global PCAP_DIR, PROGRESS_FILE, LOG_FILE
    PCAP_DIR      = user_dir / "pcap"
    PROGRESS_FILE = user_dir / "progress.csv"
    LOG_FILE      = user_dir / "collection.log"

    PCAP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ]
    )

    deadline = None
    if args.time_limit:
        deadline = time.time() + args.time_limit * 60
        log.info(f"Time limit: {args.time_limit} minutes")

    sites = [s.strip() for s in SITES_FILE.read_text().splitlines()
             if s.strip() and not s.startswith("#")]
    if args.max_sites:
        sites = sites[:args.max_sites]

    # Shard sites across pods: pod i gets sites[start:end]
    if args.num_shards > 1:
        shard_size = -(-len(sites) // args.num_shards)  # ceiling division
        start = args.shard_index * shard_size
        end   = min(start + shard_size, len(sites))
        sites = sites[start:end]
        log.info(f"Shard {args.shard_index}/{args.num_shards}: "
                 f"sites[{start}:{end}] ({len(sites)} sites)")

    log.info(f"Loaded {len(sites)} sites, collecting {args.instances} "
             f"instances each = {len(sites) * args.instances} total visits")

    done = load_progress()
    remaining = sum(
        1 for i in range(len(sites))
        for j in range(args.instances)
        if (i, j) not in done
    )
    log.info(f"Already done: {len(done)}, remaining: {remaining}")

    log.info("Connecting to Tor control port...")
    with Controller.from_port(port=TOR_CONTROL_PORT) as ctrl:
        ctrl.authenticate(password=TOR_CONTROL_PASS)
        log.info("Tor authenticated")

        new_circuit(ctrl)
        guard_ip = get_guard_ip(ctrl)
        log.info(f"Current guard IP: {guard_ip}")

        driver = make_driver()
        log.info("Firefox started")

        total_visits = len(sites) * args.instances
        pbar = tqdm(total=remaining, unit="visit", dynamic_ncols=True)

        try:
            for instance in range(args.instances):
                order = list(range(len(sites)))
                random.shuffle(order)

                for site_idx in order:
                    if deadline and time.time() >= deadline:
                        log.info("Time limit reached — stopping. Resume with the same command.")
                        raise KeyboardInterrupt

                    if (site_idx, instance) in done:
                        continue

                    url = sites[site_idx]
                    pcap_path = PCAP_DIR / f"{site_idx:03d}-{instance:03d}.pcap"

                    success = False
                    for attempt in range(MAX_RETRIES):
                        try:
                            new_circuit(ctrl)
                            guard_ip = get_guard_ip(ctrl)
                        except RuntimeError as e:
                            log.error(f"Circuit error: {e}")
                            time.sleep(5)
                            continue

                        t0 = time.time()
                        cap = start_capture(pcap_path, guard_ip, args.interface)
                        time.sleep(0.3)

                        ok = visit_page(driver, url)
                        stop_capture(cap)
                        elapsed = time.time() - t0

                        pcap_size = pcap_path.stat().st_size if pcap_path.exists() else 0

                        if ok and pcap_size > 500:
                            log_progress(site_idx, instance, url, "ok",
                                         pcap_size, elapsed, guard_ip)
                            done.add((site_idx, instance))
                            success = True
                            break
                        else:
                            status = "empty" if pcap_size <= 500 else "timeout"
                            log.warning(f"  attempt {attempt+1}/{MAX_RETRIES} "
                                        f"failed ({status}) for {url}")
                            log_progress(site_idx, instance, url, status,
                                         pcap_size, elapsed, guard_ip)
                            if attempt == 1:
                                try:
                                    driver.quit()
                                except Exception:
                                    pass
                                driver = make_driver()

                    if not success:
                        log.error(f"SKIP {url} instance {instance} after "
                                  f"{MAX_RETRIES} attempts")

                    pbar.update(1)
                    pbar.set_postfix(site=url[:30], inst=instance,
                                     guard=guard_ip)

        except KeyboardInterrupt:
            log.info("Interrupted by user — progress saved, safe to resume")
        finally:
            pbar.close()
            try:
                driver.quit()
            except Exception:
                pass

    collected = len(load_progress())
    log.info(f"Done. Total successful traces: {collected} / {total_visits}")
    log.info(f"pcap files are in: {PCAP_DIR}")
    log.info("Data persists on node disk. Next step: process.py will read it.")


if __name__ == "__main__":
    main()
