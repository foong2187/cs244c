"""Tamaraw defense simulator.

Fixed dual-rate constant padding defense. Sends outgoing packets at one
constant rate and incoming packets at another, padding total counts to
multiples of a threshold.

Reference: Cai et al., "A Systematic Approach to Developing and
Evaluating Website Fingerprinting Defenses", CCS 2014.
Algorithm faithfully reproduced from Tao Wang's implementation.
"""

import math


DEFAULT_PARAMS = {
    "rho_out": 0.04,   # 40ms between outgoing packets (25 pkt/s)
    "rho_in": 0.012,   # 12ms between incoming packets (~83 pkt/s)
    "pad_l": 100,       # pad to next multiple of pad_l
}


def _sign(x: float) -> int:
    return 1 if x > 0 else -1


def _anoa(trace: list[list[float]], rho_out: float,
          rho_in: float) -> list[list[float]]:
    """Rate-control phase: send at constant rates per direction."""
    if not trace:
        return []

    # Separate by direction, keeping timestamps
    outgoing = [t for t, d in trace if d > 0]
    incoming = [t for t, d in trace if d < 0]

    result = []

    # Process outgoing at rho_out intervals
    if outgoing:
        current_time = outgoing[0]
        pkt_idx = 0
        while pkt_idx < len(outgoing):
            if outgoing[pkt_idx] <= current_time:
                result.append([current_time, 1])
                pkt_idx += 1
            else:
                # Send dummy
                result.append([current_time, 1])
            current_time += rho_out
        # No trailing padding in this phase
    # Process incoming at rho_in intervals
    if incoming:
        current_time = incoming[0]
        pkt_idx = 0
        while pkt_idx < len(incoming):
            if incoming[pkt_idx] <= current_time:
                result.append([current_time, -1])
                pkt_idx += 1
            else:
                result.append([current_time, -1])
            current_time += rho_in

    return result


def _anoa_pad(trace: list[list[float]], pad_l: int,
              rho_out: float, rho_in: float) -> list[list[float]]:
    """Padding phase: pad total count to next multiple of pad_l."""
    if not trace:
        return []

    outgoing = [[t, d] for t, d in trace if d > 0]
    incoming = [[t, d] for t, d in trace if d < 0]

    # Pad outgoing to next multiple of pad_l
    n_out = len(outgoing)
    target_out = math.ceil(n_out / pad_l) * pad_l
    if outgoing:
        last_time = outgoing[-1][0]
        for i in range(target_out - n_out):
            last_time += rho_out
            outgoing.append([last_time, 1])

    # Pad incoming to next multiple of pad_l
    n_in = len(incoming)
    target_in = math.ceil(n_in / pad_l) * pad_l
    if incoming:
        last_time = incoming[-1][0]
        for i in range(target_in - n_in):
            last_time += rho_in
            incoming.append([last_time, -1])

    result = outgoing + incoming
    result.sort(key=lambda x: x[0])
    return result


def simulate(trace: list[list[float]],
             params: dict | None = None) -> list[list[float]]:
    """Apply Tamaraw defense to a trace.

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

    rho_out = params["rho_out"]
    rho_in = params["rho_in"]
    pad_l = params["pad_l"]

    # Phase 1: Rate control
    rate_controlled = _anoa(trace, rho_out, rho_in)

    # Phase 2: Length padding
    defended = _anoa_pad(rate_controlled, pad_l, rho_out, rho_in)

    return defended
