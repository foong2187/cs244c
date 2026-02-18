#!/usr/bin/env python3
"""Collect website fingerprinting traces using Tor Browser and tcpdump.

Usage:
    python scripts/collect_traces.py --config configs/default.yaml
                                     [--tor-browser-path /opt/tor-browser]
                                     [--start-batch 0]
                                     [--end-batch 89]
                                     [--resume]

Prerequisites:
    - Tor Browser Bundle installed at --tor-browser-path
    - tcpdump installed and accessible (may require sudo/capabilities)
    - Xvfb running if on a headless server (export DISPLAY=:99)
    - data/collected/site_list.txt populated with label<TAB>url entries

Output:
    data/collected/pcap/{label}-{batch}.pcap  (one per visit)
    data/collected/progress.csv               (logging)
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT, ensure_dirs, load_config

logger = logging.getLogger(__name__)


class TraceCollector:
    """Manages the full trace collection lifecycle."""

    def __init__(self, config: dict, tor_browser_path: str, resume: bool = False):
        """Initialize the collector.

        Args:
            config: Loaded YAML config dict.
            tor_browser_path: Path to Tor Browser Bundle directory.
            resume: If True, skip already-completed (site, batch) pairs.
        """
        self.config = config
        self.tor_browser_path = tor_browser_path
        self.resume = resume

        self.collected_dir = PROJECT_ROOT / config["data"]["collected_dir"]
        self.pcap_dir = self.collected_dir / "pcap"
        self.progress_file = self.collected_dir / "progress.csv"
        self.log_file = self.collected_dir / "collection.log"

        self.tor_process = None
        self.controller = None
        self.active_tcpdump = None  # Track for signal handler cleanup
        self.completed_pairs: set[tuple[int, int]] = set()

        # Collection parameters
        coll = config["collection"]
        self.page_load_timeout = coll["page_load_timeout"]
        self.post_load_wait = coll["post_load_wait"]
        self.newnym_wait = coll["newnym_wait"]
        self.max_retries = coll["max_retries"]

    def load_site_list(self) -> list[tuple[int, str]]:
        """Load site list from data/collected/site_list.txt.

        Format: label<TAB>url (one per line, # comments allowed)

        Returns:
            List of (label, url) tuples.

        Raises:
            FileNotFoundError: If site list file does not exist.
            ValueError: If file is empty or malformed.
        """
        site_list_path = PROJECT_ROOT / self.config["data"]["site_list"]

        if not site_list_path.exists():
            raise FileNotFoundError(
                f"Site list not found: {site_list_path}\n"
                "Create it with lines: label<TAB>url (e.g., '0\\thttps://www.google.com')"
            )

        sites = []
        with open(site_list_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(
                        f"site_list.txt:{line_num}: expected 'label<TAB>url', "
                        f"got: {line!r}"
                    )
                label = int(parts[0])
                url = parts[1]
                sites.append((label, url))

        if not sites:
            raise ValueError(f"Site list is empty: {site_list_path}")

        logger.info(f"Loaded {len(sites)} sites from {site_list_path}")
        return sites

    def load_progress(self) -> set[tuple[int, int]]:
        """Load completed (label, batch) pairs from progress.csv.

        Only pairs with status='success' are considered completed.

        Returns:
            Set of (label, batch) tuples.
        """
        if not self.progress_file.exists():
            return set()

        completed = set()
        with open(self.progress_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "success":
                    completed.add((int(row["label"]), int(row["batch"])))

        logger.info(f"Loaded {len(completed)} completed pairs from progress.csv")
        return completed

    def log_progress(
        self,
        label: int,
        batch: int,
        status: str,
        duration: float = 0.0,
        pcap_size: int = 0,
        error_msg: str = "",
    ) -> None:
        """Append a row to progress.csv.

        Creates the file with headers if it doesn't exist.
        """
        write_header = not self.progress_file.exists()

        with open(self.progress_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "label", "batch", "timestamp", "status",
                    "duration_sec", "pcap_size_bytes", "error_msg",
                ])
            writer.writerow([
                label, batch,
                datetime.now().isoformat(timespec="seconds"),
                status, f"{duration:.1f}", pcap_size, error_msg,
            ])

    def start_tor(self) -> None:
        """Launch Tor and open a stem controller connection.

        Uses tbselenium's utility to launch the Tor process with the
        Tor Browser Bundle, then connects via stem's Controller.
        """
        from stem import Signal
        from stem.control import Controller
        from tbselenium.utils import launch_tbb_tor_with_stem

        logger.info("Starting Tor...")

        tbb_path = self.tor_browser_path
        self.tor_process = launch_tbb_tor_with_stem(tbb_path=tbb_path)

        # Connect controller on default control port
        self.controller = Controller.from_port(port=9051)
        self.controller.authenticate()

        logger.info("Tor started and controller connected")

        # Wait for a circuit to be established
        self._wait_for_circuit(timeout=60)

    def stop_tor(self) -> None:
        """Stop the Tor process and close the controller."""
        if self.controller:
            try:
                self.controller.close()
            except Exception:
                pass
            self.controller = None

        if self.tor_process:
            try:
                self.tor_process.kill()
            except Exception:
                pass
            self.tor_process = None

        logger.info("Tor stopped")

    def _wait_for_circuit(self, timeout: int = 60) -> None:
        """Wait until at least one circuit is established."""
        start = time.time()
        while time.time() - start < timeout:
            circuits = self.controller.get_circuits()
            established = [c for c in circuits if c.status == "BUILT"]
            if established:
                logger.info(f"Circuit established ({len(established)} built)")
                return
            time.sleep(2)
        raise RuntimeError(f"No Tor circuit built within {timeout}s")

    def get_guard_ip(self) -> str:
        """Detect the current guard relay IP address via stem.

        Finds the first hop (guard) of an established circuit and
        resolves its IP address from the network status.

        Returns:
            Guard relay IP address as a string.

        Raises:
            RuntimeError: If no guard can be detected.
        """
        circuits = self.controller.get_circuits()
        established = [c for c in circuits if c.status == "BUILT"]

        if not established:
            self._wait_for_circuit()
            circuits = self.controller.get_circuits()
            established = [c for c in circuits if c.status == "BUILT"]

        if not established:
            raise RuntimeError("No established Tor circuits found")

        # Get the guard (first hop) fingerprint
        guard_fp = established[0].path[0][0]
        guard_status = self.controller.get_network_status(guard_fp)
        guard_ip = guard_status.address

        logger.info(f"Guard relay IP: {guard_ip} (fingerprint: {guard_fp[:8]}...)")
        return guard_ip

    def send_newnym(self) -> None:
        """Send NEWNYM signal for a fresh Tor circuit."""
        from stem import Signal

        self.controller.signal(Signal.NEWNYM)
        time.sleep(self.newnym_wait)

    def start_tcpdump(self, pcap_path: Path, guard_ip: str) -> subprocess.Popen:
        """Start a tcpdump capture process.

        Args:
            pcap_path: Path to save the PCAP file.
            guard_ip: Guard relay IP for the capture filter.

        Returns:
            Popen object for the tcpdump process.

        Raises:
            PermissionError: If tcpdump requires root privileges.
            FileNotFoundError: If tcpdump is not installed.
        """
        pcap_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "tcpdump",
            "-i", "any",
            "-w", str(pcap_path),
            f"tcp and host {guard_ip}",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "tcpdump not found. Install with: sudo apt install tcpdump"
            )
        except PermissionError:
            raise PermissionError(
                "tcpdump requires elevated privileges. Either:\n"
                "  1. Run this script with sudo, or\n"
                "  2. Grant capture capability: "
                "sudo setcap cap_net_raw+ep $(which tcpdump)"
            )

        # Give tcpdump a moment to start capturing
        time.sleep(0.5)

        # Check it didn't immediately fail
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"tcpdump exited immediately: {stderr}")

        self.active_tcpdump = proc
        return proc

    def stop_tcpdump(self, proc: subprocess.Popen) -> None:
        """Stop a tcpdump capture process gracefully.

        Sends SIGTERM, waits up to 5 seconds, then SIGKILL if needed.
        """
        if proc.poll() is not None:
            # Already terminated
            self.active_tcpdump = None
            return

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

        self.active_tcpdump = None

    def visit_site(self, url: str, timeout: int | None = None) -> float:
        """Visit a URL using TorBrowserDriver and measure load time.

        Creates a fresh driver instance per visit to avoid cache
        contamination between sites.

        Args:
            url: URL to visit.
            timeout: Page load timeout in seconds.

        Returns:
            Page load duration in seconds.
        """
        from tbselenium.tbdriver import TorBrowserDriver

        if timeout is None:
            timeout = self.page_load_timeout

        start = time.time()

        driver = TorBrowserDriver(
            self.tor_browser_path,
            tor_cfg=0,  # Use system Tor (we manage it via stem)
            tbb_logfile_path=str(self.collected_dir / "tbb.log"),
        )

        try:
            driver.set_page_load_timeout(timeout)
            driver.load_url(url, wait_on_page=timeout, wait_for_page_body=True)
            time.sleep(self.post_load_wait)
            duration = time.time() - start
        finally:
            driver.quit()

        return duration

    def collect_single_visit(
        self, label: int, url: str, batch: int, guard_ip: str
    ) -> None:
        """Collect a single trace with retry logic.

        Args:
            label: Site label.
            url: Site URL.
            batch: Batch number.
            guard_ip: Guard relay IP for tcpdump filter.
        """
        pcap_path = self.pcap_dir / f"{label}-{batch}.pcap"

        for attempt in range(1, self.max_retries + 1):
            tcpdump_proc = None
            try:
                # Fresh circuit
                self.send_newnym()

                # Start capture
                tcpdump_proc = self.start_tcpdump(pcap_path, guard_ip)

                # Visit site
                duration = self.visit_site(url)

                # Stop capture
                self.stop_tcpdump(tcpdump_proc)

                # Verify PCAP was written
                pcap_size = pcap_path.stat().st_size if pcap_path.exists() else 0

                self.log_progress(
                    label, batch, "success",
                    duration=duration, pcap_size=pcap_size,
                )
                logger.info(
                    f"[{label}-{batch}] Success: {url} "
                    f"({duration:.1f}s, {pcap_size} bytes)"
                )
                return  # Success

            except Exception as e:
                # Ensure tcpdump is stopped on error
                if tcpdump_proc:
                    self.stop_tcpdump(tcpdump_proc)

                # Remove partial PCAP
                if pcap_path.exists():
                    pcap_path.unlink()

                if attempt < self.max_retries:
                    logger.warning(
                        f"[{label}-{batch}] Attempt {attempt}/{self.max_retries} "
                        f"failed: {e}. Retrying..."
                    )
                else:
                    logger.error(
                        f"[{label}-{batch}] All {self.max_retries} attempts failed: {e}"
                    )
                    self.log_progress(
                        label, batch, "failed",
                        error_msg=str(e),
                    )

    def collect_batch(
        self, batch_num: int, sites: list[tuple[int, str]], guard_ip: str
    ) -> None:
        """Collect one batch of traces for all sites.

        Shuffles site order within the batch to avoid temporal ordering
        artifacts.

        Args:
            batch_num: Batch number.
            sites: List of (label, url) tuples.
            guard_ip: Guard relay IP.
        """
        # Shuffle sites for this batch (seeded for reproducibility)
        shuffled = list(sites)
        rng = random.Random(batch_num)
        rng.shuffle(shuffled)

        for label, url in shuffled:
            # Skip if already completed (resume mode)
            if (label, batch_num) in self.completed_pairs:
                continue

            self.collect_single_visit(label, url, batch_num, guard_ip)

    def run(self, start_batch: int = 0, end_batch: int = 89) -> None:
        """Main collection loop.

        Args:
            start_batch: First batch number (inclusive).
            end_batch: Last batch number (inclusive).
        """
        # Setup signal handler for clean shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def cleanup_handler(signum, frame):
            logger.info("Interrupt received, cleaning up...")
            if self.active_tcpdump:
                self.stop_tcpdump(self.active_tcpdump)
            self.stop_tor()
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

        try:
            sites = self.load_site_list()

            if self.resume:
                self.completed_pairs = self.load_progress()

            self.start_tor()

            total_batches = end_batch - start_batch + 1
            for batch_num in range(start_batch, end_batch + 1):
                logger.info(
                    f"=== Batch {batch_num}/{end_batch} "
                    f"({batch_num - start_batch + 1}/{total_batches}) ==="
                )

                # Re-detect guard IP at start of each batch
                guard_ip = self.get_guard_ip()

                self.collect_batch(batch_num, sites, guard_ip)

            logger.info("Collection complete!")

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise

        finally:
            self.stop_tor()
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        # Print summary
        if self.progress_file.exists():
            completed = self.load_progress()
            total_expected = len(sites) * total_batches
            logger.info(
                f"Summary: {len(completed)} successful visits "
                f"out of {total_expected} expected"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--tor-browser-path", type=str, default=None,
        help="Path to Tor Browser Bundle (default: from config)",
    )
    parser.add_argument(
        "--start-batch", type=int, default=0,
        help="First batch number (default: 0)",
    )
    parser.add_argument(
        "--end-batch", type=int, default=None,
        help="Last batch number (default: num_batches - 1 from config)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume collection, skipping completed (site, batch) pairs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    tor_browser_path = (
        args.tor_browser_path or config["collection"]["tor_browser_path"]
    )
    end_batch = (
        args.end_batch
        if args.end_batch is not None
        else config["collection"]["num_batches"] - 1
    )

    # Setup logging: both console and file
    log_file = PROJECT_ROOT / config["data"]["collected_dir"] / "collection.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # Check prerequisites
    if not Path(tor_browser_path).exists():
        logger.error(f"Tor Browser not found at: {tor_browser_path}")
        logger.error("Install Tor Browser Bundle or pass --tor-browser-path")
        sys.exit(1)

    if not os.environ.get("DISPLAY"):
        logger.warning(
            "DISPLAY not set. If running headless, start Xvfb first:\n"
            "  Xvfb :99 -screen 0 1920x1080x24 &\n"
            "  export DISPLAY=:99"
        )

    # Check tcpdump availability
    try:
        subprocess.run(
            ["tcpdump", "--version"],
            capture_output=True, check=True,
        )
    except FileNotFoundError:
        logger.error("tcpdump not found. Install with: sudo apt install tcpdump")
        sys.exit(1)
    except subprocess.CalledProcessError:
        # --version may return non-zero on some systems but still means it exists
        pass

    collector = TraceCollector(config, tor_browser_path, resume=args.resume)
    collector.run(start_batch=args.start_batch, end_batch=end_batch)


if __name__ == "__main__":
    main()
