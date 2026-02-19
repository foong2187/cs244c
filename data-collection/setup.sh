#!/bin/bash
# one-time VM setup: installs tor, firefox, geckodriver, python deps
# usage: sudo bash setup.sh
set -e

echo "=== [1/6] Updating packages ==="
apt-get update -q
apt-get install -y \
    tor \
    tcpdump \
    python3-pip \
    python3-venv \
    wget \
    curl \
    net-tools \
    iproute2 \
    software-properties-common

# firefox-esr is Debian-only; on Ubuntu 22.04 Firefox ships as a snap which
# breaks geckodriver. Install the real deb from Mozilla's PPA instead.
echo "=== [1b/6] Installing Firefox deb from Mozilla PPA ==="
add-apt-repository -y ppa:mozillateam/ppa
# Prefer PPA package over the snap stub
cat > /etc/apt/preferences.d/mozilla-firefox << 'PREF'
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
PREF
apt-get update -q
apt-get install -y firefox

echo "=== [2/6] Installing geckodriver ==="
GECKO_VER="v0.35.0"
wget -q "https://github.com/mozilla/geckodriver/releases/download/${GECKO_VER}/geckodriver-${GECKO_VER}-linux64.tar.gz"
tar -xzf "geckodriver-${GECKO_VER}-linux64.tar.gz"
mv geckodriver /usr/local/bin/geckodriver
chmod +x /usr/local/bin/geckodriver
rm "geckodriver-${GECKO_VER}-linux64.tar.gz"
echo "Geckodriver installed: $(geckodriver --version | head -1)"

echo "=== [3/6] Configuring Tor ==="
TOR_PASS="cs244c_collection"
HASHED=$(tor --hash-password "$TOR_PASS" | tail -1)

# Append to torrc (avoid duplicates)
grep -q "ControlPort 9051" /etc/tor/torrc || cat >> /etc/tor/torrc << EOF

# WF data collection settings
ControlPort 9051
HashedControlPassword ${HASHED}
MaxCircuitDirtiness 10
EOF

systemctl restart tor
systemctl enable tor

# Wait for Tor to bootstrap
echo "Waiting for Tor to connect..."
for i in $(seq 1 30); do
    if systemctl is-active --quiet tor; then
        sleep 2
        break
    fi
    sleep 1
done
echo "Tor status: $(systemctl is-active tor)"

echo "=== [4/6] Granting tcpdump without sudo password ==="
CURRENT_USER="${SUDO_USER:-$USER}"
echo "${CURRENT_USER} ALL=(ALL) NOPASSWD: /usr/bin/tcpdump" \
    > /etc/sudoers.d/tcpdump-collection
chmod 440 /etc/sudoers.d/tcpdump-collection

echo "=== [5/6] Installing Python dependencies ==="
cd /home/${CURRENT_USER}/cs244c/data-collection 2>/dev/null || \
cd "$(dirname "$0")"

# Create and activate venv
sudo -u "$CURRENT_USER" python3 -m venv .venv
sudo -u "$CURRENT_USER" .venv/bin/pip install --quiet \
    stem \
    selenium \
    tqdm \
    dpkt \
    numpy \
    requests

echo "=== [6/6] Detecting network interface ==="
IFACE=$(ip route | grep default | awk '{print $5}' | head -1)
echo "Primary interface: ${IFACE}"
echo "Update NETWORK_IFACE in collect.py if this doesn't match."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify Tor is running: systemctl status tor"
echo "  2. Run the collector:     .venv/bin/python collect.py"
echo "  3. After collection:      .venv/bin/python process.py"
