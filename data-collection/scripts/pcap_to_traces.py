#!/usr/bin/env python3
"""Convert PCAP files to timestamp+direction trace format.

Usage:
    python scripts/pcap_to_traces.py [--config configs/default.yaml]
                                     [--guard-ip GUARD_IP]
                                     [--pcap-dir data/collected/pcap]
                                     [--output-dir data/collected/traces]

Each PCAP file {label}-{batch}.pcap produces a trace file {label}-{batch}
containing tab-separated lines: timestamp<TAB>direction
"""

import argparse
import logging
import socket
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dpkt
from tqdm import tqdm

from src.utils.config import PROJECT_ROOT, ensure_dirs, load_config

logger = logging.getLogger(__name__)


def detect_guard_ip_from_pcap(pcap_path: str | Path) -> str:
    """Auto-detect the guard relay IP from a PCAP file.

    Since tcpdump was run with 'tcp and host <guard_ip>', all packets
    should involve the guard IP on one side. The most frequent non-local
    IP across all packets is the guard.

    Args:
        pcap_path: Path to a PCAP file.

    Returns:
        Guard relay IP address as a string.

    Raises:
        RuntimeError: If guard IP cannot be determined.
    """
    ip_counts = Counter()

    with open(pcap_path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except (dpkt.dpkt.NeedData, ValueError) as e:
            raise RuntimeError(f"Cannot read PCAP {pcap_path}: {e}")

        for _ts, buf in pcap:
            ip_pkt = _extract_ip(pcap.datalink(), buf)
            if ip_pkt is None:
                continue

            src, dst = _get_ip_addrs(ip_pkt)
            if src:
                ip_counts[src] += 1
            if dst:
                ip_counts[dst] += 1

    if not ip_counts:
        raise RuntimeError(f"No IP packets found in {pcap_path}")

    # The guard IP appears in every packet (as src or dst).
    # The local client IP also appears in every packet.
    # If there are only two IPs, pick the non-private one.
    # If ambiguous, pick the most common one (they should be equal).
    candidates = ip_counts.most_common()

    # Filter out common private/local ranges
    non_local = [
        (ip, count)
        for ip, count in candidates
        if not _is_private_ip(ip)
    ]

    if non_local:
        guard_ip = non_local[0][0]
    else:
        # All IPs are local (unlikely in Tor traffic); pick most common
        guard_ip = candidates[0][0]

    logger.info(f"Auto-detected guard IP: {guard_ip}")
    return guard_ip


def _is_private_ip(ip: str) -> bool:
    """Check if an IP address is in a private/local range."""
    private_prefixes = (
        "10.", "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
        "172.30.", "172.31.", "192.168.", "127.",
    )
    return ip.startswith(private_prefixes)


def _extract_ip(datalink: int, buf: bytes):
    """Extract the IP packet from a raw frame buffer.

    Handles Ethernet (DLT_EN10MB) and Linux cooked capture (DLT_LINUX_SLL).

    Returns:
        dpkt.ip.IP or dpkt.ip6.IP6 packet, or None if not IP.
    """
    try:
        if datalink == dpkt.pcap.DLT_EN10MB:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return eth.data
        elif datalink == dpkt.pcap.DLT_LINUX_SLL:
            sll = dpkt.sll.SLL(buf)
            if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return sll.data
        elif datalink == dpkt.pcap.DLT_RAW:
            # Raw IP, no link layer header
            version = (buf[0] >> 4) & 0xF
            if version == 4:
                return dpkt.ip.IP(buf)
            elif version == 6:
                return dpkt.ip6.IP6(buf)
        else:
            return None
    except (dpkt.dpkt.UnpackError, IndexError):
        return None
    return None


def _get_ip_addrs(ip_pkt) -> tuple[str | None, str | None]:
    """Extract source and destination IP addresses as strings."""
    try:
        if isinstance(ip_pkt, dpkt.ip.IP):
            src = socket.inet_ntoa(ip_pkt.src)
            dst = socket.inet_ntoa(ip_pkt.dst)
        elif isinstance(ip_pkt, dpkt.ip6.IP6):
            src = socket.inet_ntop(socket.AF_INET6, ip_pkt.src)
            dst = socket.inet_ntop(socket.AF_INET6, ip_pkt.dst)
        else:
            return None, None
        return src, dst
    except (socket.error, ValueError):
        return None, None


def pcap_to_trace(
    pcap_path: str | Path, guard_ip: str
) -> list[tuple[float, int]]:
    """Convert a single PCAP file to a list of (timestamp, direction) tuples.

    Args:
        pcap_path: Path to the PCAP file.
        guard_ip: IP address of the Tor guard relay.

    Returns:
        List of (relative_timestamp, direction) tuples.
        Direction: +1 = outgoing (client -> guard), -1 = incoming (guard -> client).
    """
    trace = []
    first_ts = None

    with open(pcap_path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except (dpkt.dpkt.NeedData, ValueError) as e:
            logger.warning(f"Cannot read PCAP {pcap_path}: {e}")
            return []

        datalink = pcap.datalink()

        for ts, buf in pcap:
            ip_pkt = _extract_ip(datalink, buf)
            if ip_pkt is None:
                continue

            src, dst = _get_ip_addrs(ip_pkt)
            if src is None or dst is None:
                continue

            if dst == guard_ip:
                direction = 1  # outgoing
            elif src == guard_ip:
                direction = -1  # incoming
            else:
                continue  # not Tor traffic

            if first_ts is None:
                first_ts = ts

            trace.append((ts - first_ts, direction))

    return trace


def write_trace(trace: list[tuple[float, int]], output_path: Path) -> None:
    """Write a trace as tab-separated timestamp+direction lines.

    Args:
        trace: List of (timestamp, direction) tuples.
        output_path: Path to the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ts, direction in trace:
            f.write(f"{ts:.6f}\t{direction}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--guard-ip", type=str, default=None,
        help="Guard relay IP address. If not provided, auto-detected from first PCAP.",
    )
    parser.add_argument(
        "--pcap-dir", type=str, default=None,
        help="Directory containing PCAP files (default: data/collected/pcap)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for trace files (default: data/collected/traces)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    ensure_dirs(config)

    pcap_dir = Path(args.pcap_dir) if args.pcap_dir else PROJECT_ROOT / config["data"]["collected_dir"] / "pcap"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / config["data"]["collected_dir"] / "traces"

    pcap_files = sorted(pcap_dir.glob("*.pcap"))
    if not pcap_files:
        logger.error(f"No PCAP files found in {pcap_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pcap_files)} PCAP files in {pcap_dir}")

    # Determine guard IP
    guard_ip = args.guard_ip
    if guard_ip is None:
        logger.info("Auto-detecting guard IP from first PCAP...")
        guard_ip = detect_guard_ip_from_pcap(pcap_files[0])

    logger.info(f"Using guard IP: {guard_ip}")

    # Convert each PCAP
    converted = 0
    skipped = 0
    for pcap_path in tqdm(pcap_files, desc="Converting PCAPs"):
        trace = pcap_to_trace(pcap_path, guard_ip)

        if not trace:
            logger.warning(f"Empty trace from {pcap_path.name}, skipping")
            skipped += 1
            continue

        # Output filename: same stem as PCAP (e.g., "0-5.pcap" -> "0-5")
        output_path = output_dir / pcap_path.stem
        write_trace(trace, output_path)
        converted += 1

    logger.info(
        f"Done: {converted} traces converted, {skipped} skipped. "
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
