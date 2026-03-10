#!/usr/bin/env bash
# Setup defense simulators for website fingerprinting evaluation.
#
# Clones BRO, RegulaTor, WTF-PAD, and wfes (Tamaraw + BuFLO) repos
# into data-collection/defenses/repos/.
#
# Usage: bash scripts/setup_defenses.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOS_DIR="$PROJECT_DIR/defenses/repos"

mkdir -p "$REPOS_DIR"

clone_or_update() {
    local name="$1"
    local url="$2"
    local dir="$REPOS_DIR/$name"

    if [ -d "$dir" ]; then
        echo "[$name] Already cloned, pulling latest..."
        git -C "$dir" pull --ff-only 2>/dev/null || echo "[$name] Pull skipped (detached HEAD or conflicts)"
    else
        echo "[$name] Cloning $url..."
        git clone "$url" "$dir"
    fi
}

echo "=== Setting up defense simulators ==="

# BRO: Zero-delay beta-distribution padding
clone_or_update "bro" "https://github.com/csmcguan/bro.git"

# RegulaTor: Rate-based regularization (PETS 2022)
clone_or_update "regulator" "https://github.com/jkhollandjr/RegulaTor.git"

# WTF-PAD: Adaptive padding (ESORICS 2016)
clone_or_update "wtfpad" "https://github.com/wtfpad/wtfpad.git"

# wfes: Website Fingerprinting Evaluation Suite (includes Tamaraw + BuFLO)
clone_or_update "wfes" "https://github.com/gchers/wfes.git"

echo ""
echo "=== All defense repos cloned to $REPOS_DIR ==="
echo ""
echo "Repos:"
ls -1 "$REPOS_DIR"
echo ""
echo "Next: run 'python scripts/simulate_defenses.py --help' to apply defenses to traces."
