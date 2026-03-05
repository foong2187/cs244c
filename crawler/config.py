"""
Crawler configuration: site list, paths, and constants.

Site list format: label<TAB>url (one per line; lines starting with # are skipped).
"""

import os

# Default paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SITE_LIST = os.path.join(REPO_ROOT, "data", "yousef-data", "site_list.txt")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "crawler-traces")
DEFAULT_PCAP_DIR = os.path.join(REPO_ROOT, "data", "crawler-pcap")

# DF paper: direction sequence length
SEQUENCE_LENGTH = 5000

# Convention: +1 = packet from client to guard (outgoing), -1 = guard to client (incoming)
OUTGOING = 1
INCOMING = -1
PAD = 0


def load_site_list(path=None):
    """Load site list from file. Returns list of (label int, url str)."""
    path = path or DEFAULT_SITE_LIST
    sites = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    label = int(parts[0])
                    url = parts[1].strip()
                    sites.append((label, url))
                except ValueError:
                    continue
    return sites
