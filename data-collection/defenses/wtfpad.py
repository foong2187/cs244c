"""WTF-PAD defense simulator wrapper.

Adaptive padding defense that injects dummy packets based on
inter-arrival time distributions, using separate distributions for
burst and gap states.

Reference: Juarez et al., "Toward an Efficient Website Fingerprinting
Defense", ESORICS 2016.

This module wraps the original WTF-PAD simulator at
https://github.com/wtfpad/wtfpad via subprocess. Run setup_defenses.sh
first to clone the repo.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config from the paper (best defense effectiveness)
DEFAULT_CONFIG = "normal_rcv"


def _find_wtfpad_repo() -> Path | None:
    """Locate the cloned WTF-PAD repository."""
    # Check relative to this file
    defenses_dir = Path(__file__).resolve().parent
    repo = defenses_dir / "repos" / "wtfpad"
    if repo.exists():
        return repo
    return None


def simulate_directory(input_dir: str | Path, output_dir: str | Path,
                       config: str = DEFAULT_CONFIG) -> bool:
    """Apply WTF-PAD to all traces in a directory via subprocess.

    Unlike other defenses, WTF-PAD is called on the whole directory
    at once because the original tool operates on directories.

    Args:
        input_dir: Directory containing trace files (timestamp\\tlength format).
        output_dir: Directory to write defended traces.
        config: WTF-PAD configuration name.

    Returns:
        True if successful, False otherwise.
    """
    repo = _find_wtfpad_repo()
    if repo is None:
        logger.error(
            "WTF-PAD repo not found. Run: bash scripts/setup_defenses.sh"
        )
        return False

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # WTF-PAD expects length (not just direction), so we convert
    # our +1/-1 to +1/-1 (it extracts direction from sign of length)
    # The simulator will set all lengths to MTU internally.

    try:
        result = subprocess.run(
            [sys.executable, "src/main.py", "-c", config, str(input_dir)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            logger.error(f"WTF-PAD failed: {result.stderr}")
            return False

        # Find the results directory (named {config}_{timestamp})
        results_base = repo / "results"
        if not results_base.exists():
            logger.error("WTF-PAD produced no results directory")
            return False

        # Get most recent results directory matching config
        result_dirs = sorted(
            [d for d in results_base.iterdir()
             if d.is_dir() and d.name.startswith(config)],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if not result_dirs:
            logger.error(f"No WTF-PAD results found for config {config}")
            return False

        src_dir = result_dirs[0]

        # Copy results and convert format back to timestamp\tdirection
        for trace_file in src_dir.iterdir():
            if trace_file.is_file() and not trace_file.name.startswith("."):
                _convert_wtfpad_output(trace_file, output_dir / trace_file.name)

        # Clean up results directory
        shutil.rmtree(src_dir, ignore_errors=True)

        return True

    except subprocess.TimeoutExpired:
        logger.error("WTF-PAD timed out after 1 hour")
        return False
    except FileNotFoundError:
        logger.error("Python not found for WTF-PAD subprocess")
        return False


def _convert_wtfpad_output(src: Path, dst: Path) -> None:
    """Convert WTF-PAD output (timestamp\\tsigned_length) to our format
    (timestamp\\tdirection)."""
    lines = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            timestamp = parts[0]
            length = int(float(parts[1]))
            direction = 1 if length > 0 else -1
            lines.append(f"{timestamp}\t{direction}\n")

    with open(dst, "w") as f:
        f.writelines(lines)


def simulate(trace: list[list[float]], config: str = DEFAULT_CONFIG) -> list[list[float]]:
    """Apply WTF-PAD defense to a single trace.

    Creates temporary files for the subprocess-based simulator.

    Args:
        trace: List of [timestamp, direction] pairs.
        config: WTF-PAD configuration name.

    Returns:
        Defended trace as list of [timestamp, direction] pairs.
    """
    repo = _find_wtfpad_repo()
    if repo is None:
        logger.error(
            "WTF-PAD repo not found. Run: bash scripts/setup_defenses.sh"
        )
        return trace

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write trace in WTF-PAD format
        input_file = Path(tmpdir) / "input" / "0-0"
        input_file.parent.mkdir()
        with open(input_file, "w") as f:
            for ts, direction in trace:
                f.write(f"{ts:.6f}\t{int(direction)}\n")

        # Run simulator
        result = subprocess.run(
            [sys.executable, "src/main.py", "-c", config,
             str(input_file.parent)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"WTF-PAD failed: {result.stderr}")
            return trace

        # Find output
        results_base = repo / "results"
        result_dirs = sorted(
            [d for d in results_base.iterdir()
             if d.is_dir() and d.name.startswith(config)],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if not result_dirs:
            return trace

        output_file = result_dirs[0] / "0-0"
        if not output_file.exists():
            return trace

        # Read and convert
        defended = []
        with open(output_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    ts = float(parts[0])
                    length = int(float(parts[1]))
                    direction = 1 if length > 0 else -1
                    defended.append([ts, direction])

        # Cleanup
        shutil.rmtree(result_dirs[0], ignore_errors=True)

        return defended
