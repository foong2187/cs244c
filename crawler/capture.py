"""
Packet capture and conversion to DF-style direction sequences.

Capture: run tcpdump (requires root or cap_net_raw) to record traffic.
Conversion: filter pcap by entry-guard IP, sort by time, map each packet to
+1 (outgoing from client) or -1 (incoming to client), truncate/pad to SEQUENCE_LENGTH.
"""

import subprocess

import numpy as np

from .config import SEQUENCE_LENGTH, OUTGOING, INCOMING, PAD

try:
    import dpkt
    from dpkt import pcap as dpkt_pcap
    from dpkt import sll
    from dpkt import sll2
except ImportError:
    dpkt = None
    dpkt_pcap = None
    sll = None
    sll2 = None


def start_capture(output_pcap_path, interface="any", extra_filter=None):
    """
    Start tcpdump in the background. Call stop_capture() when done.
    Requires root or cap_net_raw (e.g. sudo).

    Args:
        output_pcap_path: Where to write the pcap file.
        interface: Capture interface (e.g. "any", "eth0").
        extra_filter: Optional BPF filter (e.g. "tcp port 443 or tcp port 9001").

    Returns:
        subprocess.Popen for the tcpdump process.
    """
    cmd = [
        "tcpdump",
        "-i", interface,
        "-w", output_pcap_path,
        "-U",
    ]
    if extra_filter:
        cmd.extend([extra_filter])
    else:
        # Capture Tor OR ports (guard/mid/exit) and local SOCKS (9050) so we see traffic
        # when using requests (Python -> 127.0.0.1:9050 -> Tor -> guard)
        cmd.append("tcp port 443 or tcp port 9001 or tcp port 9050")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_capture(proc, timeout=5):
    """Stop tcpdump gracefully (SIGTERM)."""
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    _ = proc.stderr.read()


def _get_ip_layer(buf, datalink):
    """Parse buf (raw pcap packet) into IP layer; return (src, dst) or None. Handles Ethernet and Linux cooked (SLL/SLL2)."""
    try:
        if datalink == dpkt_pcap.DLT_EN10MB or datalink == 1:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
        elif datalink == dpkt_pcap.DLT_LINUX_SLL or datalink == 113:
            sll_pkt = sll.SLL(buf)
            ip = sll_pkt.data
        elif datalink == dpkt_pcap.DLT_LINUX_SLL2 or datalink == 276:
            sll2_pkt = sll2.SLL2(buf)
            ip = sll2_pkt.data
        else:
            return None
        if not isinstance(ip, dpkt.ip.IP):
            return None
        return (_ip_to_str(ip.src), _ip_to_str(ip.dst))
    except Exception:
        return None


def pcap_to_sequence(pcap_path, guard_ip, max_len=None):
    """
    Read pcap, keep only packets to/from guard_ip, order by time,
    map to direction (+1 outgoing, -1 incoming), truncate/pad to max_len.

    If guard_ip is None or "0.0.0.0" and no events are found, infers guard from
    pcap (remote IP with most packets) and retries.

    Args:
        pcap_path: Path to pcap file.
        guard_ip: Entry guard IP (string), or None to infer from pcap.
        max_len: Output length (default SEQUENCE_LENGTH).

    Returns:
        np.ndarray of dtype float32, shape (max_len,), values in {-1, 0, 1}.
    """
    if dpkt is None:
        raise ImportError("dpkt is required for pcap parsing: pip install dpkt")
    max_len = max_len or SEQUENCE_LENGTH

    def collect_events(gip):
        events = []
        with open(pcap_path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
                datalink = getattr(reader, "datalink", lambda: dpkt_pcap.DLT_EN10MB)()
                if callable(datalink):
                    datalink = datalink()
            except dpkt.dpkt.UnpackError:
                return []
            for ts, buf in reader:
                pair = _get_ip_layer(buf, datalink)
                if pair is None:
                    continue
                src, dst = pair
                if src != gip and dst != gip:
                    continue
                if dst == gip:
                    events.append((ts, OUTGOING))
                else:
                    events.append((ts, INCOMING))
        return events

    events = collect_events(guard_ip or "0.0.0.0")
    if not events and (not guard_ip or guard_ip == "0.0.0.0"):
        inferred = infer_guard_ip_from_pcap(pcap_path)
        if inferred:
            events = collect_events(inferred)
    if not events:
        return pcap_to_sequence_all_packets(pcap_path, max_len)
    events.sort(key=lambda x: x[0])
    directions = np.array([d for _, d in events], dtype=np.float32)
    if len(directions) > max_len:
        directions = directions[:max_len]
    elif len(directions) < max_len:
        pad = np.zeros(max_len - len(directions), dtype=np.float32) + PAD
        directions = np.concatenate([directions, pad])
    return directions


def _ip_to_str(ip_bytes):
    if isinstance(ip_bytes, (str, bytes)) and len(ip_bytes) == 4:
        return ".".join(str(b) for b in ip_bytes)
    return str(ip_bytes)


def _is_local_ip(ip_str):
    """True if IP is localhost or private (we are the client)."""
    if not ip_str or ip_str == "0.0.0.0":
        return True
    parts = ip_str.split(".")
    if len(parts) != 4:
        return True
    try:
        a, b = int(parts[0]), int(parts[1])
        if a == 127:
            return True
        if a == 10:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
    except ValueError:
        return True
    return False


def infer_guard_ip_from_pcap(pcap_path):
    """
    When stem doesn't give us the guard, infer it from the pcap: the remote IP
    (not local) that has the most packets is likely the first hop (guard).
    Returns None if pcap is empty or unreadable.
    """
    if dpkt is None:
        return None
    counts = {}
    try:
        with open(pcap_path, "rb") as f:
            reader = dpkt.pcap.Reader(f)
            datalink = getattr(reader, "datalink", lambda: dpkt_pcap.DLT_EN10MB)()
            if callable(datalink):
                datalink = datalink()
            for _, buf in reader:
                pair = _get_ip_layer(buf, datalink)
                if pair is None:
                    continue
                src, dst = pair
                for addr in (src, dst):
                    if not _is_local_ip(addr):
                        counts[addr] = counts.get(addr, 0) + 1
    except Exception:
        return None
    if not counts:
        return None
    return max(counts, key=counts.get)


def pcap_to_sequence_all_packets(pcap_path, max_len=None):
    """
    Fallback: use ALL IP packets, infer 'our' IP as the most common one in the pcap,
    then direction = outgoing if we are src, else incoming. Use when guard filtering
    yields nothing but pcap has traffic.
    """
    if dpkt is None:
        raise ImportError("dpkt is required for pcap parsing: pip install dpkt")
    max_len = max_len or SEQUENCE_LENGTH
    counts = {}
    events = []
    try:
        with open(pcap_path, "rb") as f:
            reader = dpkt.pcap.Reader(f)
            datalink = getattr(reader, "datalink", lambda: dpkt_pcap.DLT_EN10MB)()
            if callable(datalink):
                datalink = datalink()
            for ts, buf in reader:
                pair = _get_ip_layer(buf, datalink)
                if pair is None:
                    continue
                src, dst = pair
                for addr in (src, dst):
                    counts[addr] = counts.get(addr, 0) + 1
                events.append((ts, src, dst))
    except Exception:
        return np.zeros(max_len, dtype=np.float32) + PAD
    if not events or not counts:
        return np.zeros(max_len, dtype=np.float32) + PAD
    # "Our" IP = the one that appears most (we're in every packet)
    our_ip = max(counts, key=counts.get)
    directions = []
    for ts, src, dst in events:
        if src == our_ip:
            directions.append((ts, OUTGOING))
        elif dst == our_ip:
            directions.append((ts, INCOMING))
    directions.sort(key=lambda x: x[0])
    directions = np.array([d for _, d in directions], dtype=np.float32)
    if len(directions) == 0:
        return np.zeros(max_len, dtype=np.float32) + PAD
    if len(directions) > max_len:
        directions = directions[:max_len]
    else:
        pad = np.zeros(max_len - len(directions), dtype=np.float32) + PAD
        directions = np.concatenate([directions, pad])
    return directions
