"""RegulaTor defense simulator.

Rate-based traffic regularization defense that controls packet sending
rates with exponential decay and burst detection.

Reference: Holland & Hopper, "RegulaTor: A Straightforward Website
Fingerprinting Defense", PETS 2022.
Algorithm faithfully reproduced from https://github.com/jkhollandjr/RegulaTor
"""

import numpy as np


DEFAULT_PARAMS = {
    "budget": 3550,
    "orig_rate": 277.0,
    "dep_rate": 0.94,
    "threshold": 3.55,
    "upload_ratio": 3.95,
    "delay_cap": 1.77,
}

# Cutoffs from original code
CUTOFF_TIME = 120  # seconds
CUTOFF_LENGTH = 20000  # packets


def _regulator_download(download_pkts: list[float], params: dict,
                        rng: np.random.RandomState) -> list[float]:
    """Apply RegulaTor defense to download (incoming) traffic."""
    if not download_pkts:
        return []

    orig_rate = params["orig_rate"]
    dep_rate = params["dep_rate"]
    threshold = params["threshold"]
    max_budget = params["budget"]

    padding_budget = rng.randint(0, max_budget)
    output = []

    # Initial constant-rate phase (circuit setup)
    init_interval = 0.1
    current_time = 0.0
    pkt_idx = 0

    # Send initial packets at constant rate until first real packet
    if download_pkts:
        while current_time < download_pkts[0]:
            output.append(current_time)
            current_time += init_interval
            padding_budget -= 1
            if padding_budget <= 0:
                break

    burst_time = current_time if current_time > 0 else 0.0
    target_rate = orig_rate

    while pkt_idx < len(download_pkts):
        elapsed = current_time - burst_time
        target_rate = orig_rate * (dep_rate ** elapsed) if elapsed > 0 else orig_rate

        if target_rate < 1:
            target_rate = 1

        interval = 1.0 / target_rate

        # Count queued real packets
        queued = 0
        for i in range(pkt_idx, len(download_pkts)):
            if download_pkts[i] <= current_time:
                queued += 1
            else:
                break

        if queued > 0:
            # Send a real packet
            output.append(current_time)
            pkt_idx += 1

            # Check burst condition
            if queued > threshold * target_rate:
                burst_time = current_time
        elif padding_budget > 0:
            # Send a dummy packet
            output.append(current_time)
            padding_budget -= 1
        # else: skip this time slot

        current_time += interval

    return output


def _regulator_upload(defended_download: list[float],
                      upload_pkts: list[float],
                      params: dict) -> list[float]:
    """Apply RegulaTor defense to upload (outgoing) traffic."""
    if not upload_pkts:
        return []

    upload_ratio = params["upload_ratio"]
    delay_cap = params["delay_cap"]

    # Create upload schedule from download trace
    n_slots = max(1, int(len(defended_download) / upload_ratio))

    if n_slots >= len(defended_download):
        schedule = list(defended_download)
    else:
        indices = np.linspace(0, len(defended_download) - 1, n_slots, dtype=int)
        schedule = [defended_download[i] for i in indices]

    # Add initial constant-rate slots
    init_interval = 0.2
    init_time = 0.0
    init_slots = []
    if schedule:
        while init_time < schedule[0]:
            init_slots.append(init_time)
            init_time += init_interval
    schedule = init_slots + schedule
    schedule.sort()

    output = []
    slot_idx = 0

    for pkt_time in upload_pkts:
        # Find next available slot at or after packet time
        while slot_idx < len(schedule) and schedule[slot_idx] < pkt_time:
            slot_idx += 1

        if slot_idx < len(schedule) and (schedule[slot_idx] - pkt_time) <= delay_cap:
            output.append(schedule[slot_idx])
            slot_idx += 1
        else:
            # No slot available within delay cap; delay by delay_cap
            output.append(pkt_time + delay_cap)

    return output


def simulate(trace: list[list[float]], params: dict | None = None,
             rng: np.random.RandomState | None = None) -> list[list[float]]:
    """Apply RegulaTor defense to a trace.

    Args:
        trace: List of [timestamp, direction] pairs.
        params: Defense parameters dict. Uses DEFAULT_PARAMS if None.
        rng: Random state for reproducibility.

    Returns:
        Defended trace as list of [timestamp, direction] pairs.
    """
    if rng is None:
        rng = np.random.RandomState()
    if params is None:
        params = DEFAULT_PARAMS

    if not trace:
        return trace

    # Apply cutoffs
    filtered = [[t, d] for t, d in trace
                if t <= CUTOFF_TIME][:CUTOFF_LENGTH]

    if not filtered:
        return trace

    download_pkts = [t for t, d in filtered if d < 0]
    upload_pkts = [t for t, d in filtered if d > 0]

    defended_download = _regulator_download(download_pkts, params, rng)
    defended_upload = _regulator_upload(defended_download, upload_pkts, params)

    # Merge
    result = [[t, -1] for t in defended_download]
    result.extend([t, 1] for t in defended_upload)
    result.sort(key=lambda x: x[0])

    return result
