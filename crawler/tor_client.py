"""
Tor client control via stem: start/stop, get SOCKS port, get entry-guard IP.

Requires: Tor installed and either running (use_port()) or we launch it (launch_tor()).
"""

import time

try:
    from stem import CircStatus
    from stem.control import Controller
except ImportError:
    Controller = None
    CircStatus = None


def get_controller(port=9051):
    """Connect to Tor control port. Default 9051; authenticate with no password."""
    if Controller is None:
        raise ImportError("stem is required: pip install stem")
    return Controller.from_port(port=port)


def get_socks_port(controller):
    """Return the SOCKS port (int) for TCP. Tor may report unix:socket; we use 9050 for browser."""
    raw = controller.get_conf("SocksPort", "9050")
    if raw is None:
        return 9050
    # Can be "9050" or "unix:/run/tor/socks WorldWritable" or multiple lines
    for part in str(raw).replace(",", " ").split():
        part = part.strip()
        if part.isdigit():
            return int(part)
        # e.g. "9050" in "127.0.0.1:9050"
        if ":" in part and part.split(":")[-1].isdigit():
            return int(part.split(":")[-1])
    return 9050


def get_entry_guard_ip(controller):
    """
    Return the IP address of the entry guard for the most recently built circuit.
    If no circuit exists yet, returns None (caller should trigger a request first).
    """
    for circ in controller.get_circuits():
        if circ.status != CircStatus.BUILT:
            continue
        if not circ.path:
            continue
        fingerprint = circ.path[0][0]
        try:
            ns = controller.get_network_status(fingerprint)
            if ns and ns.address:
                return ns.address
        except Exception:
            continue
    return None


def new_circuit(controller):
    """Request a new circuit (NEWNYM). Waits a few seconds for it to take effect."""
    controller.signal("NEWNYM")
    time.sleep(3)
