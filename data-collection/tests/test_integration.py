"""End-to-end integration test for the data collection pipeline.

Tests the full flow: synthetic PCAP -> traces -> pickle, verifying
the output at each stage matches expectations.
"""

import pickle
import socket
import sys
from pathlib import Path

import dpkt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pcap_to_traces import detect_guard_ip_from_pcap, pcap_to_trace, write_trace
from scripts.traces_to_pickle import load_trace_file, save_splits, split_data, validate_traces
from src.data.preprocessing import load_pickle, pad_or_truncate


def _build_ethernet_frame(src_ip: str, dst_ip: str) -> bytes:
    """Build a minimal Ethernet frame wrapping an IP/TCP packet."""
    tcp = dpkt.tcp.TCP(sport=12345, dport=443)
    tcp.data = b""
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src_ip),
        dst=socket.inet_aton(dst_ip),
        p=dpkt.ip.IP_PROTO_TCP,
    )
    ip.data = tcp
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        dst=b"\x00" * 6,
        src=b"\x00" * 6,
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=bytes(ip),
    )
    return bytes(eth)


class TestFullPipeline:
    """End-to-end: PCAP creation -> trace extraction -> pickle conversion."""

    NUM_SITES = 3
    INSTANCES_PER_SITE = 10
    GUARD_IP = "203.0.113.50"
    CLIENT_IP = "192.168.1.100"

    def _create_synthetic_pcaps(self, pcap_dir: Path) -> None:
        """Create synthetic PCAPs simulating Tor traffic for multiple sites."""
        rng = np.random.RandomState(42)

        for label in range(self.NUM_SITES):
            for batch in range(self.INSTANCES_PER_SITE):
                pcap_path = pcap_dir / f"{label}-{batch}.pcap"
                num_packets = rng.randint(100, 300)

                with open(pcap_path, "wb") as f:
                    writer = dpkt.pcap.Writer(f)
                    ts = 1000.0

                    for _ in range(num_packets):
                        # Randomly choose direction
                        if rng.random() < 0.4:
                            frame = _build_ethernet_frame(self.CLIENT_IP, self.GUARD_IP)
                        else:
                            frame = _build_ethernet_frame(self.GUARD_IP, self.CLIENT_IP)
                        writer.writepkt(frame, ts)
                        ts += rng.exponential(0.02)

    def test_pcap_to_traces_stage(self, tmp_path):
        """Stage 1: PCAP -> trace text files."""
        pcap_dir = tmp_path / "pcap"
        pcap_dir.mkdir()
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        self._create_synthetic_pcaps(pcap_dir)

        # Auto-detect guard IP from first PCAP
        first_pcap = sorted(pcap_dir.glob("*.pcap"))[0]
        detected_ip = detect_guard_ip_from_pcap(first_pcap)
        assert detected_ip == self.GUARD_IP

        # Convert all PCAPs
        converted = 0
        for pcap_path in sorted(pcap_dir.glob("*.pcap")):
            trace = pcap_to_trace(pcap_path, self.GUARD_IP)
            assert len(trace) > 0, f"Empty trace from {pcap_path.name}"

            output_path = traces_dir / pcap_path.stem
            write_trace(trace, output_path)
            converted += 1

        assert converted == self.NUM_SITES * self.INSTANCES_PER_SITE

        # Verify trace file format
        sample_trace = traces_dir / "0-0"
        assert sample_trace.exists()
        lines = sample_trace.read_text().strip().split("\n")
        assert len(lines) > 50

        parts = lines[0].split("\t")
        assert len(parts) == 2
        assert float(parts[0]) == 0.0  # First timestamp is 0
        assert int(parts[1]) in (1, -1)

    def test_traces_to_pickle_stage(self, tmp_path):
        """Stage 2: trace text files -> pickle."""
        pcap_dir = tmp_path / "pcap"
        pcap_dir.mkdir()
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        pickle_dir = tmp_path / "pickle"

        # Create PCAPs and convert to traces
        self._create_synthetic_pcaps(pcap_dir)
        for pcap_path in sorted(pcap_dir.glob("*.pcap")):
            trace = pcap_to_trace(pcap_path, self.GUARD_IP)
            write_trace(trace, traces_dir / pcap_path.stem)

        # Load trace files and build arrays
        all_traces = []
        all_labels = []
        for trace_file in sorted(traces_dir.iterdir()):
            label = int(trace_file.name.split("-")[0])
            directions = load_trace_file(trace_file)
            padded = pad_or_truncate(np.array(directions, dtype=np.float32), 5000)
            all_traces.append(padded)
            all_labels.append(label)

        X = np.array(all_traces, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        assert X.shape == (self.NUM_SITES * self.INSTANCES_PER_SITE, 5000)
        assert y.shape == (self.NUM_SITES * self.INSTANCES_PER_SITE,)

        # Validate
        X, y, stats = validate_traces(X, y, min_packets=50)
        assert stats["num_removed_short"] == 0  # All synthetic traces have 100+ packets
        assert stats["num_labels"] == self.NUM_SITES

        # Split
        splits = split_data(X, y, 0.8, 0.1, seed=42)
        assert len(splits["train"][0]) + len(splits["valid"][0]) + len(splits["test"][0]) == len(X)

        # Verify stratified: each label in all splits
        for split_name in ["train", "valid", "test"]:
            labels_in_split = set(splits[split_name][1].tolist())
            assert labels_in_split == set(range(self.NUM_SITES))

        # Save and verify roundtrip
        save_splits(splits, pickle_dir, "Test")

        for split_name in ["train", "valid", "test"]:
            X_loaded = load_pickle(pickle_dir / f"X_{split_name}_Test.pkl")
            y_loaded = load_pickle(pickle_dir / f"y_{split_name}_Test.pkl")
            np.testing.assert_array_equal(X_loaded, splits[split_name][0])
            np.testing.assert_array_equal(y_loaded, splits[split_name][1])

    def test_trace_values_are_valid(self, tmp_path):
        """Verify that trace values are only +1, -1, or 0 (padding)."""
        pcap_dir = tmp_path / "pcap"
        pcap_dir.mkdir()
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        self._create_synthetic_pcaps(pcap_dir)
        for pcap_path in sorted(pcap_dir.glob("*.pcap")):
            trace = pcap_to_trace(pcap_path, self.GUARD_IP)
            write_trace(trace, traces_dir / pcap_path.stem)

        for trace_file in sorted(traces_dir.iterdir()):
            directions = load_trace_file(trace_file)
            padded = pad_or_truncate(np.array(directions, dtype=np.float32), 5000)

            unique_vals = set(np.unique(padded).tolist())
            assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
                f"{trace_file.name}: unexpected values {unique_vals - {-1.0, 0.0, 1.0}}"
            )
