#!/usr/bin/env bash
# GCP VM provisioning script for DF website fingerprinting data collection.
#
# Run on a fresh Ubuntu 22.04 LTS GCP VM:
#   bash gcp/setup.sh
#
# This script is idempotent — safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOR_BROWSER_DIR="/opt/tor-browser"
TOR_VERSION="14.0.4"
TOR_ARCHIVE="tor-browser-linux-x86_64-${TOR_VERSION}.tar.xz"
TOR_URL="https://archive.torproject.org/tor-package-archive/torbrowser/${TOR_VERSION}/${TOR_ARCHIVE}"
VENV_DIR="$HOME/df-env"

echo "=== DF Website Fingerprinting — GCP VM Setup ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# -------------------------------------------------------
# 1. System packages
# -------------------------------------------------------
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    xvfb \
    tcpdump \
    python3-pip \
    python3-venv \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    wget \
    tmux \
    > /dev/null

echo "  Done."

# -------------------------------------------------------
# 2. Python virtual environment
# -------------------------------------------------------
echo "[2/6] Setting up Python virtual environment at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "  Done. Activate with: source $VENV_DIR/bin/activate"

# -------------------------------------------------------
# 3. Tor Browser Bundle
# -------------------------------------------------------
echo "[3/6] Installing Tor Browser ${TOR_VERSION}..."
if [ -d "$TOR_BROWSER_DIR/Browser" ]; then
    echo "  Tor Browser already installed at $TOR_BROWSER_DIR, skipping."
else
    cd /tmp
    if [ ! -f "$TOR_ARCHIVE" ]; then
        echo "  Downloading Tor Browser..."
        wget -q "$TOR_URL" -O "$TOR_ARCHIVE"
    fi
    echo "  Extracting..."
    sudo mkdir -p "$TOR_BROWSER_DIR"
    sudo tar -xf "$TOR_ARCHIVE" -C /opt/
    # The archive extracts to "tor-browser/", rename if needed
    if [ -d "/opt/tor-browser_en-US" ]; then
        sudo rm -rf "$TOR_BROWSER_DIR"
        sudo mv "/opt/tor-browser_en-US" "$TOR_BROWSER_DIR"
    fi
    sudo chown -R "$USER:$USER" "$TOR_BROWSER_DIR"
    echo "  Done."
fi

# -------------------------------------------------------
# 4. Tor configuration
# -------------------------------------------------------
echo "[4/6] Configuring Tor (torrc)..."
TORRC_DIR="$TOR_BROWSER_DIR/Browser/TorBrowser/Data/Tor"
mkdir -p "$TORRC_DIR"
cat > "$TORRC_DIR/torrc" << 'EOF'
# Force fresh circuits for every connection
MaxCircuitDirtiness 1

# Disable predictive circuit building
LearnCircuitBuildTimeout 0
EOF
echo "  Done."

# -------------------------------------------------------
# 5. tcpdump capabilities
# -------------------------------------------------------
echo "[5/6] Granting tcpdump packet capture capability..."
TCPDUMP_PATH="$(which tcpdump)"
sudo setcap cap_net_raw+ep "$TCPDUMP_PATH"
echo "  Done. Verify: $(getcap "$TCPDUMP_PATH")"

# -------------------------------------------------------
# 6. Project directories
# -------------------------------------------------------
echo "[6/6] Creating project directories..."
mkdir -p "$PROJECT_DIR/data/collected/pcap"
mkdir -p "$PROJECT_DIR/data/collected/traces"
mkdir -p "$PROJECT_DIR/data/collected/pickle"
mkdir -p "$PROJECT_DIR/data/raw/ClosedWorld/NoDef"
mkdir -p "$PROJECT_DIR/data/raw/OpenWorld/NoDef"
echo "  Done."

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Checklist:"
echo "  [x] System packages installed"
echo "  [x] Python venv at $VENV_DIR"
echo "  [x] Tor Browser at $TOR_BROWSER_DIR"
echo "  [x] torrc configured (MaxCircuitDirtiness 1)"
echo "  [x] tcpdump has CAP_NET_RAW"
echo "  [x] Data directories created"
echo ""
echo "Next steps:"
echo "  1. Verify site list:   cat $PROJECT_DIR/data/collected/site_list.txt"
echo "  2. Start collection:   bash $PROJECT_DIR/gcp/run_collection.sh"
echo "  3. Monitor progress:   tail -f $PROJECT_DIR/data/collected/collection.log"
echo "  4. Attach to session:  tmux attach -t collection"
