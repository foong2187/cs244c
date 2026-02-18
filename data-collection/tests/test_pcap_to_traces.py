"""Tests for scripts/pcap_to_traces.py."""

import io
import socket
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dpkt

from scripts.pcap_to_traces import (
    _get_ip_addrs,
    _is_private_ip,
    detect_guard_ip_from_pcap,
    pcap_to_trace,
    write_trace,
)


def _build_ip_packet(src_ip: str, dst_ip: str) -> bytes:
    """Build a minimal IP/TCP packet with given src and dst addresses."""
    tcp = dpkt.tcp.TCP(sport=12345, dport=443)
    tcp.data = b""
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src_ip),
        dst=socket.inet_aton(dst_ip),
        p=dpkt.ip.IP_PROTO_TCP,
    )
    ip.data = tcp
    ip.len = len(ip)
    return bytes(ip)


def _build_ethernet_frame(src_ip: str, dst_ip: str) -> bytes:
    """Build a minimal Ethernet frame wrapping an IP/TCP packet."""
    ip_bytes = _build_ip_packet(src_ip, dst_ip)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\x00" * 6,
        src=b"\x00" * 6,
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip_bytes,
    )
    return bytes(eth)


def _create_synthetic_pcap(
    tmp_path: Path, packets: list[tuple[float, str, str]], filename: str = "test.pcap"
) -> Path:
    """Create a PCAP file with the given packets.

    Args:
        tmp_path: Temporary directory.
        packets: List of (timestamp, src_ip, dst_ip) tuples.
        filename: Output filename.

    Returns:
        Path to the created PCAP file.
    """
    pcap_path = tmp_path / filename
    with open(pcap_path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for ts, src_ip, dst_ip in packets:
            frame = _build_ethernet_frame(src_ip, dst_ip)
            writer.writepkt(frame, ts)
    return pcap_path


def _create_empty_pcap(tmp_path: Path, filename: str = "empty.pcap") -> Path:
    """Create an empty PCAP file (header only, no packets)."""
    pcap_path = tmp_path / filename
    with open(pcap_path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        # Write no packets — just the PCAP header
    return pcap_path


# --- Tests ---


class TestIsPrivateIp:
    def test_private_10(self):
        assert _is_private_ip("10.0.0.1") is True

    def test_private_192_168(self):
        assert _is_private_ip("192.168.1.1") is True

    def test_private_172_16(self):
        assert _is_private_ip("172.16.0.1") is True

    def test_private_127(self):
        assert _is_private_ip("127.0.0.1") is True

    def test_public_ip(self):
        assert _is_private_ip("8.8.8.8") is False

    def test_public_ip_2(self):
        assert _is_private_ip("203.0.113.1") is False


class TestGetIpAddrs:
    def test_ipv4(self):
        ip_pkt = dpkt.ip.IP(
            src=socket.inet_aton("192.168.1.1"),
            dst=socket.inet_aton("8.8.8.8"),
        )
        src, dst = _get_ip_addrs(ip_pkt)
        assert src == "192.168.1.1"
        assert dst == "8.8.8.8"

    def test_unknown_type(self):
        src, dst = _get_ip_addrs("not a packet")
        assert src is None
        assert dst is None


class TestWriteTrace:
    def test_write_and_read(self, tmp_path):
        """Writes correct tab-separated format."""
        trace = [(0.0, 1), (0.012, -1), (0.045, 1)]
        output = tmp_path / "trace_out"
        write_trace(trace, output)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "0.000000\t1"
        assert lines[1] == "0.012000\t-1"
        assert lines[2] == "0.045000\t1"

    def test_write_empty(self, tmp_path):
        """Empty trace creates empty file."""
        output = tmp_path / "empty_trace"
        write_trace([], output)
        assert output.read_text() == ""

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if missing."""
        output = tmp_path / "sub" / "dir" / "trace"
        write_trace([(0.0, 1)], output)
        assert output.exists()


class TestPcapToTrace:
    def test_basic_conversion(self, tmp_path):
        """Convert a PCAP with known packets."""
        guard_ip = "203.0.113.50"
        client_ip = "192.168.1.100"

        packets = [
            (1000.0, client_ip, guard_ip),   # outgoing: +1
            (1000.01, guard_ip, client_ip),  # incoming: -1
            (1000.02, guard_ip, client_ip),  # incoming: -1
            (1000.05, client_ip, guard_ip),  # outgoing: +1
        ]
        pcap_path = _create_synthetic_pcap(tmp_path, packets)

        trace = pcap_to_trace(pcap_path, guard_ip)
        assert len(trace) == 4

        # Check directions
        assert trace[0][1] == 1   # outgoing
        assert trace[1][1] == -1  # incoming
        assert trace[2][1] == -1  # incoming
        assert trace[3][1] == 1   # outgoing

        # Check relative timestamps
        assert trace[0][0] == pytest.approx(0.0)
        assert trace[1][0] == pytest.approx(0.01, abs=0.001)
        assert trace[3][0] == pytest.approx(0.05, abs=0.001)

    def test_empty_pcap(self, tmp_path):
        """Empty PCAP returns empty list."""
        pcap_path = _create_empty_pcap(tmp_path)
        trace = pcap_to_trace(pcap_path, "8.8.8.8")
        assert trace == []

    def test_no_matching_guard(self, tmp_path):
        """All packets have different IPs than guard → empty list."""
        packets = [
            (1000.0, "192.168.1.1", "10.0.0.1"),
            (1000.1, "10.0.0.1", "192.168.1.1"),
        ]
        pcap_path = _create_synthetic_pcap(tmp_path, packets)

        trace = pcap_to_trace(pcap_path, "203.0.113.99")
        assert trace == []

    def test_nonexistent_pcap(self, tmp_path):
        """Non-existent PCAP file returns empty list (logged warning)."""
        # pcap_to_trace opens the file directly, which will raise FileNotFoundError
        # but the function doesn't catch it - it will propagate
        with pytest.raises(FileNotFoundError):
            pcap_to_trace(tmp_path / "nonexistent.pcap", "8.8.8.8")


class TestDetectGuardIp:
    def test_detect_known_guard(self, tmp_path):
        """Detects guard IP from PCAP with public guard and private client."""
        guard_ip = "203.0.113.50"
        client_ip = "192.168.1.100"

        packets = [
            (1000.0, client_ip, guard_ip),
            (1000.01, guard_ip, client_ip),
            (1000.02, guard_ip, client_ip),
            (1000.03, client_ip, guard_ip),
            (1000.04, guard_ip, client_ip),
        ]
        pcap_path = _create_synthetic_pcap(tmp_path, packets)

        detected = detect_guard_ip_from_pcap(pcap_path)
        assert detected == guard_ip

    def test_empty_pcap_raises(self, tmp_path):
        """Empty PCAP raises RuntimeError."""
        pcap_path = _create_empty_pcap(tmp_path)
        with pytest.raises(RuntimeError, match="No IP packets"):
            detect_guard_ip_from_pcap(pcap_path)
