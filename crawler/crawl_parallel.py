"""
Parallel crawl: spawn N independent Tor instances and crawl concurrently.

Each worker gets its own Tor process (unique SocksPort, ControlPort, DataDirectory),
its own entry guard, its own Firefox, and its own tcpdump capture. No traffic mixing.

Usage:
  sudo .venv/bin/python -m crawler.crawl_parallel --workers 10 --sites 95 --visits 100

Background run:
  sudo nohup .venv/bin/python -m crawler.crawl_parallel --workers 10 --sites 95 --visits 100 > crawl.log 2>&1 &

Monitor:
  tail -f crawl.log
  ls data/crawler-pcap/*.pcap | wc -l
  grep ',ok,' data/crawler-traces/progress.csv | wc -l
"""

import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crawler.config import DEFAULT_OUTPUT_DIR, DEFAULT_PCAP_DIR, load_site_list

BASE_SOCKS_PORT = 9060
BASE_CONTROL_PORT = 9061


def _launch_tor_instance(worker_id, data_dir):
    """Launch a Tor process for this worker. Returns (tor_process, socks_port, control_port)."""
    from stem.process import launch_tor_with_config

    socks_port = BASE_SOCKS_PORT + worker_id * 2
    control_port = BASE_CONTROL_PORT + worker_id * 2

    print(f"[W{worker_id}] Starting Tor (SOCKS={socks_port}, Control={control_port})...")
    tor_process = launch_tor_with_config(
        config={
            "SocksPort": str(socks_port),
            "ControlPort": str(control_port),
            "DataDirectory": data_dir,
            "CookieAuthentication": "0",
            "Log": "notice stderr",
        },
        timeout=90,
        take_ownership=True,
    )
    print(f"[W{worker_id}] Tor ready.")
    return tor_process, socks_port, control_port


def _worker_main(worker_id, sites, visits, visit_start, output_dir, pcap_dir,
                 page_timeout, post_load_wait, interface):
    """Entry point for each worker process."""
    data_dir = os.path.join(tempfile.gettempdir(), f"tor-crawler-{worker_id}")
    os.makedirs(data_dir, exist_ok=True)

    tor_proc = None
    try:
        tor_proc, socks_port, control_port = _launch_tor_instance(worker_id, data_dir)

        from crawler.crawl import run_crawl
        run_crawl(
            sites, socks_port=socks_port, control_port=control_port,
            output_dir=output_dir, pcap_dir=pcap_dir,
            visits=visits, visit_start=visit_start,
            page_timeout=page_timeout,
            post_load_wait=post_load_wait, interface=interface,
            worker_id=worker_id,
        )
    except Exception as e:
        print(f"[W{worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tor_proc:
            try:
                tor_proc.kill()
            except Exception:
                pass
        try:
            shutil.rmtree(data_dir, ignore_errors=True)
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Parallel crawl with N Tor instances")
    p.add_argument("--workers", type=int, default=5,
                   help="Number of parallel workers (each gets its own Tor + Firefox)")
    p.add_argument("--site_list", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pcap_dir", type=str, default=DEFAULT_PCAP_DIR)
    p.add_argument("--visits", type=int, default=100,
                   help="Instances per site")
    p.add_argument("--visit-start", type=int, default=0,
                   help="Starting instance number (for multi-machine partitioning)")
    p.add_argument("--sites", type=int, default=None,
                   help="Limit to first N sites (default: all)")
    p.add_argument("--timeout", type=int, default=60,
                   help="Page load timeout (seconds)")
    p.add_argument("--wait", type=int, default=5,
                   help="Post-load wait (seconds)")
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

    n_workers = min(args.workers, len(sites))

    visit_start = args.visit_start
    total_visits = len(sites) * args.visits
    est_per_visit = args.timeout + args.wait + 15
    est_hours = (total_visits * est_per_visit) / n_workers / 3600

    print(f"=== Parallel crawl ===")
    print(f"  Workers      : {n_workers}")
    print(f"  Sites        : {len(sites)}")
    print(f"  Visits/site  : {args.visits} (instances {visit_start}-{visit_start + args.visits - 1})")
    print(f"  Total visits : {total_visits}")
    print(f"  Est. time    : {est_hours:.1f} hours")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pcap_dir, exist_ok=True)

    # All workers share the same output/pcap dirs and progress CSV.
    # Sites are split across workers (round-robin for balance).
    # Each worker handles ALL visits for its assigned sites.
    chunks = [[] for _ in range(n_workers)]
    for i, site in enumerate(sites):
        chunks[i % n_workers].append(site)

    processes = []
    for wid in range(n_workers):
        p = multiprocessing.Process(
            target=_worker_main,
            args=(wid, chunks[wid], args.visits, visit_start,
                  args.output_dir, args.pcap_dir,
                  args.timeout, args.wait, args.interface),
            daemon=True,
        )
        p.start()
        processes.append(p)
        time.sleep(3)  # stagger Tor launches

    for p in processes:
        p.join()

    # Summary
    from crawler.crawl import load_progress
    done = load_progress(args.output_dir)
    print(f"\n=== All workers finished. {len(done)} successful traces. ===")
    print(f"  pcaps: {args.pcap_dir}")
    print(f"  progress: {args.output_dir}/progress.csv")
    print(f"\nNext: python -m crawler.process")
    return 0


if __name__ == "__main__":
    sys.exit(main())
