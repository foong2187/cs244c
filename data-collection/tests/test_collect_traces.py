"""Tests for scripts/collect_traces.py.

All external dependencies (stem, tbselenium, subprocess) are mocked.
Tests focus on logic correctness of the TraceCollector class.
"""

import csv
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.collect_traces import TraceCollector


@pytest.fixture
def collector(tmp_path):
    """Create a TraceCollector with tmp_path-based config."""
    config = {
        "data": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "collected_dir": str(tmp_path / "data" / "collected"),
            "site_list": str(tmp_path / "data" / "collected" / "site_list.txt"),
            "sequence_length": 5000,
        },
        "collection": {
            "num_batches": 2,
            "page_load_timeout": 5,
            "post_load_wait": 0,
            "newnym_wait": 0,
            "max_retries": 2,
            "min_trace_packets": 50,
            "tor_browser_path": "/opt/tor-browser",
        },
        "preprocessing": {
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "seed": 42,
    }

    # Create required directories
    collected_dir = tmp_path / "data" / "collected"
    collected_dir.mkdir(parents=True)
    (collected_dir / "pcap").mkdir()

    tc = TraceCollector(config, "/opt/tor-browser", resume=False)
    # Override paths to use tmp_path
    tc.collected_dir = collected_dir
    tc.pcap_dir = collected_dir / "pcap"
    tc.progress_file = collected_dir / "progress.csv"
    return tc


class TestLoadSiteList:
    def test_valid_site_list(self, collector, tmp_path):
        """Parses valid site_list.txt correctly."""
        site_list = tmp_path / "data" / "collected" / "site_list.txt"
        site_list.write_text(
            "# Comment line\n"
            "0\thttps://www.example.com\n"
            "1\thttps://www.google.com\n"
            "\n"
            "2\thttps://www.wikipedia.org\n"
        )

        sites = collector.load_site_list()
        assert len(sites) == 3
        assert sites[0] == (0, "https://www.example.com")
        assert sites[1] == (1, "https://www.google.com")
        assert sites[2] == (2, "https://www.wikipedia.org")

    def test_missing_file(self, collector):
        """Raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            collector.load_site_list()

    def test_malformed_line(self, collector, tmp_path):
        """Raises ValueError on bad format."""
        site_list = tmp_path / "data" / "collected" / "site_list.txt"
        site_list.write_text("bad line without tab\n")

        with pytest.raises(ValueError, match="expected"):
            collector.load_site_list()

    def test_empty_file(self, collector, tmp_path):
        """Raises ValueError for file with no entries."""
        site_list = tmp_path / "data" / "collected" / "site_list.txt"
        site_list.write_text("# Only comments\n\n")

        with pytest.raises(ValueError, match="empty"):
            collector.load_site_list()


class TestLoadProgress:
    def test_parses_success_only(self, collector, sample_progress_csv):
        """Returns only (label, batch) pairs with status='success'."""
        collector.progress_file = sample_progress_csv
        completed = collector.load_progress()

        assert (0, 0) in completed
        assert (2, 0) in completed
        assert (0, 1) in completed
        assert (1, 0) not in completed  # status='failed'
        assert len(completed) == 3

    def test_missing_file(self, collector):
        """Returns empty set when file doesn't exist."""
        completed = collector.load_progress()
        assert completed == set()


class TestLogProgress:
    def test_creates_file_with_headers(self, collector):
        """Creates CSV with headers on first call."""
        collector.log_progress(0, 0, "success", duration=45.2, pcap_size=12345)

        assert collector.progress_file.exists()
        with open(collector.progress_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "label" in header
            assert "status" in header
            row = next(reader)
            assert row[0] == "0"  # label
            assert row[1] == "0"  # batch
            assert row[3] == "success"

    def test_appends_without_duplicate_headers(self, collector):
        """Second call appends without re-writing headers."""
        collector.log_progress(0, 0, "success", duration=45.2)
        collector.log_progress(1, 0, "failed", error_msg="timeout")

        with open(collector.progress_file) as f:
            lines = f.readlines()

        # 1 header + 2 data rows = 3 lines
        assert len(lines) == 3


class TestStopTor:
    def test_already_stopped(self, collector):
        """No error when controller/process are None."""
        collector.controller = None
        collector.tor_process = None
        collector.stop_tor()  # Should not raise

    def test_closes_controller_and_kills_process(self, collector):
        """Closes controller and kills Tor process."""
        mock_controller = MagicMock()
        mock_process = MagicMock()
        collector.controller = mock_controller
        collector.tor_process = mock_process

        collector.stop_tor()

        mock_controller.close.assert_called_once()
        mock_process.kill.assert_called_once()
        assert collector.controller is None
        assert collector.tor_process is None


class TestStopTcpdump:
    def test_already_terminated(self, collector):
        """No error when process already exited."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Already exited
        collector.active_tcpdump = mock_proc

        collector.stop_tcpdump(mock_proc)
        mock_proc.terminate.assert_not_called()
        assert collector.active_tcpdump is None

    def test_graceful_termination(self, collector):
        """Sends SIGTERM and waits for process to exit."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        mock_proc.wait.return_value = 0
        collector.active_tcpdump = mock_proc

        collector.stop_tcpdump(mock_proc)
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)

    def test_force_kill(self, collector):
        """SIGTERM timeout leads to SIGKILL."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="tcpdump", timeout=5),
            0,
        ]
        collector.active_tcpdump = mock_proc

        collector.stop_tcpdump(mock_proc)
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


class TestCollectBatch:
    def test_skips_completed(self, collector):
        """Resume mode skips (label, batch) in completed_pairs."""
        collector.completed_pairs = {(0, 0), (1, 0)}
        sites = [(0, "https://a.com"), (1, "https://b.com"), (2, "https://c.com")]

        with patch.object(collector, "collect_single_visit") as mock_visit:
            collector.collect_batch(0, sites, "203.0.113.50")

        # Only site 2 should be visited (0 and 1 are completed)
        assert mock_visit.call_count == 1
        call_args = mock_visit.call_args
        assert call_args[0][0] == 2  # label
        assert call_args[0][2] == 0  # batch

    def test_visits_all_when_no_resume(self, collector):
        """Without resume, all sites are visited."""
        collector.completed_pairs = set()
        sites = [(0, "https://a.com"), (1, "https://b.com")]

        with patch.object(collector, "collect_single_visit") as mock_visit:
            collector.collect_batch(0, sites, "203.0.113.50")

        assert mock_visit.call_count == 2

    def test_shuffles_order(self, collector):
        """Different batch numbers produce different visit orders."""
        collector.completed_pairs = set()
        sites = [(i, f"https://site{i}.com") for i in range(20)]

        orders = []
        for batch_num in range(3):
            visited = []
            with patch.object(collector, "collect_single_visit") as mock_visit:
                mock_visit.side_effect = lambda label, url, batch, guard: visited.append(label)
                collector.collect_batch(batch_num, sites, "203.0.113.50")
            orders.append(visited)

        # At least two of the three orders should differ
        assert orders[0] != orders[1] or orders[1] != orders[2]
