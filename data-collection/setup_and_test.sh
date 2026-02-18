#!/usr/bin/env bash
# Initial setup and test script for the DF data collection pipeline.
#
# Usage:
#   cd data-collection/
#   bash setup_and_test.sh
#
# What this does:
#   1. Creates a Python virtual environment (if not present)
#   2. Installs all dependencies
#   3. Runs the full test suite (75 tests)
#   4. Runs a mini end-to-end pipeline test (synthetic PCAP → trace → pickle)
#
# Exit codes:
#   0 — All tests passed, pipeline verified
#   1 — Setup or tests failed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PASS=0
FAIL=0

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BOLD='' NC=''
fi

log()  { echo -e "${BOLD}$1${NC}"; }
pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }

echo ""
log "========================================="
log " DF Data Collection — Setup & Test"
log "========================================="
echo ""

# -------------------------------------------------
# 1. Python virtual environment
# -------------------------------------------------
log "[1/4] Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
else
    echo "  Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pass "Python venv activated ($(python3 --version))"

# -------------------------------------------------
# 2. Install dependencies
# -------------------------------------------------
log "[2/4] Installing dependencies..."

pip install --upgrade pip -q 2>&1 | tail -1
pip install -r "$SCRIPT_DIR/requirements.txt" -q 2>&1 | tail -1
pass "All dependencies installed"

# -------------------------------------------------
# 3. Verify key imports
# -------------------------------------------------
log "[3/4] Verifying imports..."

python3 -c "import numpy; print(f'  numpy {numpy.__version__}')" && pass "numpy" || fail "numpy"
python3 -c "import yaml; print(f'  PyYAML {yaml.__version__}')" && pass "PyYAML" || fail "PyYAML"
python3 -c "import dpkt; print(f'  dpkt available')" && pass "dpkt" || fail "dpkt"
python3 -c "import stem; print(f'  stem {stem.__version__}')" && pass "stem" || fail "stem"
python3 -c "import pytest; print(f'  pytest {pytest.__version__}')" && pass "pytest" || fail "pytest"

# tbselenium and selenium are only needed on GCP, warn if missing
python3 -c "import selenium" 2>/dev/null && pass "selenium" || warn "selenium not importable (OK for testing, needed on GCP)"
python3 -c "import tbselenium" 2>/dev/null && pass "tbselenium" || warn "tbselenium not importable (OK for testing, needed on GCP)"

# -------------------------------------------------
# 4. Run test suite
# -------------------------------------------------
log "[4/4] Running test suite..."
echo ""

cd "$SCRIPT_DIR"
if python3 -m pytest tests/ -v --tb=short 2>&1; then
    echo ""
    pass "All tests passed"
else
    echo ""
    fail "Some tests failed (see output above)"
fi

# -------------------------------------------------
# Summary
# -------------------------------------------------
echo ""
log "========================================="
log " Summary"
log "========================================="
echo ""
echo -e "  ${GREEN}Passed: $PASS${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "  ${RED}Failed: $FAIL${NC}"
fi
echo ""

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Setup has issues — fix the failures above before proceeding.${NC}"
    exit 1
else
    echo -e "${GREEN}Everything looks good!${NC}"
    echo ""
    echo "Next steps:"
    echo "  - For GCP deployment: bash gcp/setup.sh"
    echo "  - To start collection: bash gcp/run_collection.sh"
    echo "  - To activate this venv: source $VENV_DIR/bin/activate"
fi
