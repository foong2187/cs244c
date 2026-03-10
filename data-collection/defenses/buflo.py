"""BuFLO (Buffered Fixed-Length Obfuscation) defense simulator.

Sends packets at a single constant rate in both directions, padding
with dummy packets when no real data is available. Continues sending
until a minimum duration has elapsed and all real packets are sent.

Reference: Dyer et al., "Peek-a-Boo, I Still See You: Why Efficient
Traffic Analysis Countermeasures Fail", IEEE S&P 2012.
"""

import math


DEFAULT_PARAMS = {
    "rho": 0.02,       # 20ms between packets (50 pkt/s per direction)
    "min_duration": 10, # minimum trace duration in seconds
}


def simulate(trace: list[list[float]],
             params: dict | None = None) -> list[list[float]]:
    """Apply BuFLO defense to a trace.

    Both directions are rate-controlled independently at the same rate.
    Dummy packets fill in when no real data is available.
    Continues past the last real packet until min_duration is reached.

    Args:
        trace: List of [timestamp, direction] pairs.
        params: Defense parameters dict. Uses DEFAULT_PARAMS if None.

    Returns:
        Defended trace as list of [timestamp, direction] pairs.
    """
    if params is None:
        params = DEFAULT_PARAMS

    if not trace:
        return trace

    rho = params["rho"]
    min_duration = params["min_duration"]

    outgoing = sorted([t for t, d in trace if d > 0])
    incoming = sorted([t for t, d in trace if d < 0])

    # Determine end time: max of last real packet and min_duration
    last_real = max(
        outgoing[-1] if outgoing else 0,
        incoming[-1] if incoming else 0,
    )
    end_time = max(last_real, min_duration)

    result = []

    # Outgoing direction
    start = outgoing[0] if outgoing else 0
    current_time = start
    pkt_idx = 0
    while current_time <= end_time or pkt_idx < len(outgoing):
        if pkt_idx < len(outgoing) and outgoing[pkt_idx] <= current_time:
            pkt_idx += 1
        # Send packet (real or dummy) -- both look the same to observer
        result.append([current_time, 1])
        current_time += rho

    # Incoming direction
    start = incoming[0] if incoming else 0
    current_time = start
    pkt_idx = 0
    while current_time <= end_time or pkt_idx < len(incoming):
        if pkt_idx < len(incoming) and incoming[pkt_idx] <= current_time:
            pkt_idx += 1
        result.append([current_time, -1])
        current_time += rho

    result.sort(key=lambda x: x[0])
    return result
