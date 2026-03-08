#!/usr/bin/env bash
# Run 2: 20 sites × 500 traces in a persistent tmux session with Xvfb.
#
# Usage:
#   bash gcp/run_collection_v2.sh                                    # Full run (batches 0-499)
#   bash gcp/run_collection_v2.sh --start-batch 0 --end-batch 99    # Partial (for parallel VMs)
#
# Data is written to data/collected_v2/ (separate from run 1).
# The collection runs inside a tmux session named "collection_v2".
# Detach with Ctrl+B then D. Reattach with: tmux attach -t collection_v2
# The --resume flag is always passed, so restarting is safe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$HOME/df-env"
TOR_BROWSER_DIR="/opt/tor-browser"
SESSION_NAME="collection_v2"

# Pass through any extra args (e.g., --start-batch 0 --end-batch 99)
EXTRA_ARGS="${*}"

# -------------------------------------------------------
# 1. Verify setup
# -------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Python venv not found at $VENV_DIR"
    echo "Run setup.sh first: bash gcp/setup.sh"
    exit 1
fi

if [ ! -d "$TOR_BROWSER_DIR/Browser" ]; then
    echo "Error: Tor Browser not found at $TOR_BROWSER_DIR"
    echo "Run setup.sh first: bash gcp/setup.sh"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/data/collected_v2/site_list.txt" ]; then
    echo "Error: site_list.txt not found at $PROJECT_DIR/data/collected_v2/site_list.txt"
    exit 1
fi

# -------------------------------------------------------
# 2. Create output directories
# -------------------------------------------------------
mkdir -p "$PROJECT_DIR/data/collected_v2/pcap"
mkdir -p "$PROJECT_DIR/data/collected_v2/traces"
mkdir -p "$PROJECT_DIR/data/collected_v2/pickle"

# -------------------------------------------------------
# 3. Start Xvfb if not running
# -------------------------------------------------------
if ! pgrep -x Xvfb > /dev/null; then
    echo "Starting Xvfb on :99..."
    Xvfb :99 -screen 0 1920x1080x24 &
    sleep 1
    echo "  Xvfb started (PID: $!)"
else
    echo "Xvfb already running."
fi
export DISPLAY=:99

# -------------------------------------------------------
# 4. Check if tmux session already exists
# -------------------------------------------------------
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "  Attach with: tmux attach -t $SESSION_NAME"
    echo "  Kill first if you want to restart: tmux kill-session -t $SESSION_NAME"
    exit 0
fi

# -------------------------------------------------------
# 5. Build the collection command
# -------------------------------------------------------
COLLECT_CMD="source $VENV_DIR/bin/activate && \
cd $PROJECT_DIR && \
DISPLAY=:99 python scripts/collect_traces.py \
    --config configs/run2.yaml \
    --tor-browser-path $TOR_BROWSER_DIR \
    --resume \
    $EXTRA_ARGS"

# -------------------------------------------------------
# 6. Launch in tmux
# -------------------------------------------------------
echo "Starting Run 2 (20 sites × 500 traces) in tmux session '$SESSION_NAME'..."
echo "  Config: configs/run2.yaml"
echo "  Data:   data/collected_v2/"
echo "  Command: python scripts/collect_traces.py --config configs/run2.yaml --resume $EXTRA_ARGS"
echo ""

tmux new-session -d -s "$SESSION_NAME" "$COLLECT_CMD"

echo "Collection started."
echo ""
echo "Useful commands:"
echo "  Attach to session:   tmux attach -t $SESSION_NAME"
echo "  Detach from session: Ctrl+B then D"
echo "  Monitor progress:    tail -f $PROJECT_DIR/data/collected_v2/collection.log"
echo "  Check PCAP count:    ls $PROJECT_DIR/data/collected_v2/pcap/ | wc -l"
echo "  View progress CSV:   tail $PROJECT_DIR/data/collected_v2/progress.csv"
