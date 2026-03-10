"""BRO (Beta Randomized Obfuscation) defense simulator.

Zero-delay defense that injects dummy packets sampled from a beta
distribution. Only adds packets, never delays real ones.

Reference: McGuan et al., "BRO: Beta Randomized Obfuscation",
Computer Communications, 2024.
Algorithm faithfully reproduced from https://github.com/csmcguan/bro
"""

import numpy as np


# Default configs from the BRO paper (b1 and b2)
CONFIGS = {
    "b1": {
        "client_min": 1, "client_max": 1500,
        "server_min": 1, "server_max": 1500,
        "min_win": 1, "max_win": 14,
        "a_min": 1, "a_max": 10,
        "b_min": 1, "b_max": 10,
    },
    "b2": {
        "client_min": 1, "client_max": 2250,
        "server_min": 1, "server_max": 2250,
        "min_win": 1, "max_win": 14,
        "a_min": 1, "a_max": 10,
        "b_min": 1, "b_max": 10,
    },
}


def _get_injection(params: dict, rng: np.random.RandomState) -> np.ndarray:
    """Generate dummy injection packets using beta-distributed timestamps."""
    # Client (outgoing) dummies
    client_win = rng.uniform(params["min_win"], params["max_win"])
    client_num = rng.randint(params["client_min"], params["client_max"] + 1)
    a = rng.uniform(params["a_min"], params["a_max"])
    b = rng.uniform(params["b_min"], params["b_max"])
    client_times = rng.beta(a, b, client_num) * client_win
    client_dummies = np.column_stack([client_times, np.ones(client_num)])

    # Server (incoming) dummies
    server_win = rng.uniform(params["min_win"], params["max_win"])
    server_num = rng.randint(params["server_min"], params["server_max"] + 1)
    a = rng.uniform(params["a_min"], params["a_max"])
    b = rng.uniform(params["b_min"], params["b_max"])
    server_times = rng.beta(a, b, server_num) * server_win
    server_dummies = np.column_stack([server_times, -np.ones(server_num)])

    injection = np.vstack([client_dummies, server_dummies])
    return injection[injection[:, 0].argsort()]


def simulate(trace: list[list[float]], config: str = "b1",
             rng: np.random.RandomState | None = None) -> list[list[float]]:
    """Apply BRO defense to a trace.

    Args:
        trace: List of [timestamp, direction] pairs. Direction: +1 or -1.
        config: BRO configuration name ("b1" or "b2").
        rng: Random state for reproducibility.

    Returns:
        Defended trace as list of [timestamp, direction] pairs.
    """
    if rng is None:
        rng = np.random.RandomState()

    params = CONFIGS[config]
    arr = np.array(trace, dtype=np.float64)
    if len(arr) == 0:
        return trace

    injection = _get_injection(params, rng)

    # Separate real packets by direction
    client_real = arr[arr[:, 1] > 0]
    server_real = arr[arr[:, 1] < 0]

    # Separate dummies by direction
    client_dummies = injection[injection[:, 1] > 0]
    server_dummies = injection[injection[:, 1] < 0]

    # Clip client dummies: remove those beyond last real outgoing packet
    if len(client_real) > 0 and len(client_dummies) > 0:
        last_client = client_real[-1, 0]
        client_dummies = client_dummies[client_dummies[:, 0] <= last_client]

    # Offset and clip server dummies
    if len(server_real) > 0 and len(server_dummies) > 0:
        first_server = server_real[0, 0]
        last_server = server_real[-1, 0]
        server_dummies[:, 0] += first_server
        server_dummies = server_dummies[server_dummies[:, 0] <= last_server]

    # Merge all packets
    parts = [arr]
    if len(client_dummies) > 0:
        parts.append(client_dummies)
    if len(server_dummies) > 0:
        parts.append(server_dummies)

    defended = np.vstack(parts)
    defended = defended[defended[:, 0].argsort()]

    return defended.tolist()
