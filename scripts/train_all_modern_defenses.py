import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class ClosedWorldResult:
    defense: str
    test_acc: float
    top2_acc: Optional[float] = None


@dataclass
class OpenWorldResult:
    defense: str
    threshold: float
    tpr: float
    fpr: float
    precision: float
    recall: float


def _run(cmd: List[str]) -> str:
    proc = subprocess.Popen(
        cmd,
        cwd=str(SRC_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    chunks: List[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    rc = proc.wait()
    out = "".join(chunks)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}\n\n{out}")
    return out


def _parse_closed_world_stdout(defense: str, out: str) -> ClosedWorldResult:
    m = re.search(r"Testing accuracy:\s*([0-9]*\.[0-9]+)", out)
    if not m:
        raise ValueError(f"Could not find Testing accuracy for {defense}")
    test_acc = float(m.group(1))

    m2 = re.search(r"Top-2 Accuracy:\s*([0-9]*\.[0-9]+)", out)
    top2 = float(m2.group(1)) if m2 else None
    return ClosedWorldResult(defense=defense, test_acc=test_acc, top2_acc=top2)


def _load_open_world_threshold_row(defense: str, threshold: float = 0.5) -> OpenWorldResult:
    path = RESULTS_DIR / f"OpenWorld_{defense}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results CSV: {path}")
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    target = None
    for r in rows:
        if float(r["threshold"]) == float(threshold):
            target = r
            break
    if target is None:
        raise ValueError(f"Threshold {threshold} not found in {path}")
    return OpenWorldResult(
        defense=defense,
        threshold=float(target["threshold"]),
        tpr=float(target["TPR"]),
        fpr=float(target["FPR"]),
        precision=float(target["Precision"]),
        recall=float(target["Recall"]),
    )


def train_closed_world(defense: str, epochs: int = 30, top_n: Optional[int] = None) -> ClosedWorldResult:
    cmd = [
        str(PYTHON),
        "train_closed_world.py",
        "--defense",
        defense,
        "--epochs",
        str(epochs),
        "--verbose",
        "2",
        "--save_model",
    ]
    if top_n is not None:
        cmd += ["--top_n", str(top_n)]
    out = _run(cmd)
    return _parse_closed_world_stdout(defense, out)


def train_open_world(defense: str, epochs: int = 30) -> OpenWorldResult:
    cmd = [
        str(PYTHON),
        "train_open_world.py",
        "--defense",
        defense,
        "--epochs",
        str(epochs),
        "--verbose",
        "2",
        "--save_model",
    ]
    _run(cmd)
    return _load_open_world_threshold_row(defense, threshold=0.5)


def main() -> int:
    closed_defenses = ["NoDef", "BRO", "BuFLO", "RegulaTor", "Tamaraw", "WalkieTalkie"]
    open_defenses = ["NoDef", "BRO", "BuFLO", "RegulaTor", "Tamaraw"]

    closed_results: List[ClosedWorldResult] = []
    open_results: List[OpenWorldResult] = []

    run_dir = REPO_ROOT / "results" / f"modern_defenses_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = run_dir / "summary.csv"

    for d in closed_defenses:
        print(f"\n=== CLOSED-WORLD: {d} ===\n")
        top_n = 2 if d == "WalkieTalkie" else None
        closed_results.append(train_closed_world(d, epochs=30, top_n=top_n))

    for d in open_defenses:
        print(f"\n=== OPEN-WORLD: {d} ===\n")
        open_results.append(train_open_world(d, epochs=30))

    # Print a compact report (also easy to copy/paste)
    print("\n=== Closed-world test accuracy ===")
    for r in closed_results:
        if r.top2_acc is not None:
            print(f"{r.defense:>12}  top-1={r.test_acc:.6f}  top-2={r.top2_acc:.6f}")
        else:
            print(f"{r.defense:>12}  top-1={r.test_acc:.6f}")

    print("\n=== Open-world @ threshold=0.50 ===")
    print(f"{'Defense':>12}  {'TPR':>8}  {'FPR':>8}  {'Prec':>8}  {'Recall':>8}")
    for r in open_results:
        print(f"{r.defense:>12}  {r.tpr:8.4f}  {r.fpr:8.4f}  {r.precision:8.4f}  {r.recall:8.4f}")

    # Write machine-readable summary
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "defense", "metric", "value"])
        for r in closed_results:
            w.writerow(["closed", r.defense, "test_top1_acc", f"{r.test_acc:.6f}"])
            if r.top2_acc is not None:
                w.writerow(["closed", r.defense, "test_top2_acc", f"{r.top2_acc:.6f}"])
        for r in open_results:
            w.writerow(["open", r.defense, "threshold", f"{r.threshold:.2f}"])
            w.writerow(["open", r.defense, "TPR", f"{r.tpr:.6f}"])
            w.writerow(["open", r.defense, "FPR", f"{r.fpr:.6f}"])
            w.writerow(["open", r.defense, "Precision", f"{r.precision:.6f}"])
            w.writerow(["open", r.defense, "Recall", f"{r.recall:.6f}"])
    print(f"\nSummary CSV written to {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

