"""
Crawl sites over Tor and capture client-guard traffic for DF training.

Best practices combined from the CCS'18 paper methodology and team's data-collection:
  - Reusable browser (one Firefox per worker, not per visit)
  - Per-visit guard IP tracking (stored in progress CSV for accurate processing)
  - Retry logic with browser restart on repeated failure
  - Randomized site visit order (reduces temporal correlation)
  - Quality filter (discard traces < MIN_PACKETS)
  - Resumable (skips completed visits based on progress CSV)

Usage (single worker):
  sudo python -m crawler.crawl --sites 5 --visits 10

For parallel crawling with N Tor instances:
  sudo python -m crawler.crawl_parallel --workers 10 --sites 95 --visits 100
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crawler.config import DEFAULT_OUTPUT_DIR, DEFAULT_PCAP_DIR, SEQUENCE_LENGTH, load_site_list
from crawler.capture import start_capture, stop_capture, pcap_to_sequence
from crawler.browser import make_driver, visit_page, quit_driver
from crawler.tor_client import get_controller, get_socks_port, get_entry_guard_ip, new_circuit

MIN_PACKETS = 50
MAX_RETRIES = 3


def _progress_path(output_dir):
    return os.path.join(output_dir, "progress.csv")


def load_progress(output_dir):
    """Return set of completed (site_idx, instance) tuples."""
    path = _progress_path(output_dir)
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((int(row["site_idx"]), int(row["instance"])))
    return done


def log_progress(output_dir, site_idx, instance, url, status,
                 pcap_bytes, elapsed, guard_ip):
    """Append one row to progress CSV."""
    path = _progress_path(output_dir)
    write_header = not os.path.exists(path)
    fields = ["timestamp", "site_idx", "instance", "url",
              "status", "pcap_bytes", "elapsed_s", "guard_ip"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "site_idx": site_idx,
            "instance": instance,
            "url": url,
            "status": status,
            "pcap_bytes": pcap_bytes,
            "elapsed_s": round(elapsed, 2),
            "guard_ip": guard_ip or "",
        })


def _get_guard_ip_with_retry(controller, socks_port, max_wait=15):
    """Get guard IP, retrying for up to max_wait seconds. Triggers a circuit if needed."""
    for _ in range(max_wait):
        gip = get_entry_guard_ip(controller)
        if gip:
            return gip
        time.sleep(1)
    # Force a circuit by making a request
    try:
        import requests as req
        proxy = f"socks5://127.0.0.1:{socks_port}"
        req.get("https://check.torproject.org", proxies={"https": proxy}, timeout=15)
    except Exception:
        pass
    time.sleep(2)
    return get_entry_guard_ip(controller)


def run_crawl(sites, socks_port, control_port, output_dir, pcap_dir,
              visits=100, visit_start=0, page_timeout=60, post_load_wait=5,
              interface="any", worker_id=None):
    """
    Core crawl loop with browser reuse, retry, randomized order, and progress tracking.

    Args:
        sites: List of (label, url) tuples.
        socks_port: Tor SOCKS port.
        control_port: Tor control port.
        output_dir: Directory for trace pickles + progress.csv.
        pcap_dir: Directory for raw pcaps.
        visits: Number of instances per site.
        visit_start: Starting instance number (for multi-machine partitioning).
        page_timeout: Selenium page load timeout (seconds).
        post_load_wait: Seconds to wait after page load for async resources.
        interface: Network interface for tcpdump.
        worker_id: Optional ID for log prefixing.

    Returns:
        Number of successful traces collected.
    """
    visit_end = visit_start + visits
    tag = f"[W{worker_id}] " if worker_id is not None else ""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pcap_dir, exist_ok=True)

    # Connect to Tor
    try:
        controller = get_controller(port=control_port)
        controller.connect()
        controller.authenticate()
    except Exception as e:
        print(f"{tag}Tor control port {control_port} failed: {e}")
        return 0

    # Get initial guard
    guard_ip = _get_guard_ip_with_retry(controller, socks_port)
    print(f"{tag}Guard: {guard_ip}, SOCKS: {socks_port}, "
          f"sites: {len(sites)}, visits: {visit_start}-{visit_end - 1}")

    # Create reusable browser
    driver = make_driver(socks_port=socks_port, page_timeout=page_timeout)
    print(f"{tag}Firefox ready")

    done = load_progress(output_dir)
    remaining = sum(1 for lbl, _ in sites for v in range(visit_start, visit_end)
                    if (lbl, v) not in done)
    print(f"{tag}Already done: {len(done)}, remaining: {remaining}")

    total_ok = 0
    consecutive_fails = 0

    # Iterate: for each instance round, visit all sites in random order
    for instance in range(visit_start, visit_end):
        order = list(range(len(sites)))
        random.shuffle(order)

        for site_pos in order:
            label, url = sites[site_pos]
            if (label, instance) in done:
                continue

            pcap_path = os.path.join(pcap_dir, f"{label:03d}-{instance:03d}.pcap")
            success = False

            for attempt in range(MAX_RETRIES):
                # New circuit for each visit attempt
                new_circuit(controller)
                guard_ip = _get_guard_ip_with_retry(controller, socks_port, max_wait=10)
                if not guard_ip:
                    print(f"{tag}  no guard IP, retrying...")
                    time.sleep(3)
                    continue

                bpf = f"host {guard_ip}"
                t0 = time.time()
                cap = start_capture(pcap_path, interface=interface, extra_filter=bpf)
                time.sleep(0.3)

                ok = visit_page(driver, url, post_load_wait=post_load_wait)
                time.sleep(1)  # catch trailing packets
                stop_capture(cap)
                elapsed = time.time() - t0

                pcap_size = os.path.getsize(pcap_path) if os.path.exists(pcap_path) else 0

                # Check quality: pcap must have enough real packets
                if pcap_size > 500:
                    seq = pcap_to_sequence(pcap_path, guard_ip)
                    nonpad = int((seq != 0).sum())
                else:
                    nonpad = 0

                if nonpad >= MIN_PACKETS:
                    log_progress(output_dir, label, instance, url, "ok",
                                 pcap_size, elapsed, guard_ip)
                    done.add((label, instance))
                    total_ok += 1
                    consecutive_fails = 0
                    success = True
                    print(f"{tag}[{label}] {url} i={instance} "
                          f"OK len={nonpad} ({elapsed:.0f}s)")
                    break
                else:
                    status = "short" if ok else "timeout"
                    log_progress(output_dir, label, instance, url, status,
                                 pcap_size, elapsed, guard_ip)
                    if attempt >= 1:
                        # Restart browser on 2nd+ failure
                        quit_driver(driver)
                        driver = make_driver(socks_port=socks_port,
                                             page_timeout=page_timeout)

            if not success:
                consecutive_fails += 1
                print(f"{tag}[{label}] {url} i={instance} "
                      f"SKIP after {MAX_RETRIES} attempts")
                if consecutive_fails >= 10:
                    print(f"{tag}10 consecutive failures, restarting browser...")
                    quit_driver(driver)
                    driver = make_driver(socks_port=socks_port,
                                         page_timeout=page_timeout)
                    consecutive_fails = 0

    quit_driver(driver)
    try:
        controller.close()
    except Exception:
        pass

    print(f"{tag}Done. {total_ok} successful traces.")
    return total_ok


def parse_args():
    p = argparse.ArgumentParser(description="Crawl sites over Tor (single worker)")
    p.add_argument("--site_list", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pcap_dir", type=str, default=DEFAULT_PCAP_DIR)
    p.add_argument("--visits", type=int, default=100)
    p.add_argument("--visit-start", type=int, default=0,
                   help="Starting instance number (for multi-machine partitioning)")
    p.add_argument("--sites", type=int, default=None)
    p.add_argument("--control_port", type=int, default=9051)
    p.add_argument("--timeout", type=int, default=60,
                   help="Page load timeout (seconds)")
    p.add_argument("--wait", type=int, default=5,
                   help="Post-load wait for async resources (seconds)")
    p.add_argument("--interface", type=str, default="any")
    return p.parse_args()


def main():
    args = parse_args()
    sites = load_site_list(args.site_list)
    if not sites:
        print("No sites loaded. Check --site_list.")
        return 1
    if args.sites is not None:
        sites = sites[:args.sites]

    socks_port_val = 9050
    try:
        c = get_controller(port=args.control_port)
        c.connect()
        c.authenticate()
        socks_port_val = get_socks_port(c)
        c.close()
    except Exception:
        pass

    run_crawl(
        sites, socks_port=socks_port_val, control_port=args.control_port,
        output_dir=args.output_dir, pcap_dir=args.pcap_dir,
        visits=args.visits, visit_start=args.visit_start,
        page_timeout=args.timeout,
        post_load_wait=args.wait, interface=args.interface,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
