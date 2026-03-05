#!/bin/bash
# One-command setup for a fresh Ubuntu VM (22.04+) to run the crawler.
# Installs: Tor, Firefox (real deb, not snap), geckodriver, tcpdump, Python venv + deps.
#
# Usage: sudo bash crawler/setup_vm.sh
# Then:  sudo .venv/bin/python -m crawler.crawl_parallel --workers 10 --sites 95 --visits 300
set -e

GECKO_VER="v0.35.0"
CURRENT_USER="${SUDO_USER:-$USER}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== [1/6] System packages ==="
apt-get update -q
apt-get install -y tor tcpdump python3-pip python3-venv wget curl \
    net-tools iproute2 software-properties-common

echo "=== [2/6] Firefox (real deb from Mozilla PPA, not snap) ==="
if command -v firefox >/dev/null 2>&1 && firefox --version 2>/dev/null | grep -q "Mozilla Firefox"; then
    echo "Firefox already installed: $(firefox --version 2>/dev/null)"
else
    add-apt-repository -y ppa:mozillateam/ppa
    cat > /etc/apt/preferences.d/mozilla-firefox << 'PREF'
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
PREF
    apt-get update -q
    apt-get install -y firefox
fi

echo "=== [3/6] geckodriver ==="
if command -v geckodriver >/dev/null 2>&1; then
    echo "geckodriver already installed: $(geckodriver --version 2>/dev/null | head -1)"
else
    wget -q "https://github.com/mozilla/geckodriver/releases/download/${GECKO_VER}/geckodriver-${GECKO_VER}-linux64.tar.gz"
    tar -xzf "geckodriver-${GECKO_VER}-linux64.tar.gz"
    mv geckodriver /usr/local/bin/geckodriver
    chmod +x /usr/local/bin/geckodriver
    rm "geckodriver-${GECKO_VER}-linux64.tar.gz"
    echo "Installed: $(geckodriver --version | head -1)"
fi

echo "=== [4/6] Tor configuration ==="
grep -q "ControlPort 9051" /etc/tor/torrc || cat >> /etc/tor/torrc << 'EOF'

# Crawler control settings
ControlPort 9051
CookieAuthentication 0
EOF
systemctl restart tor || systemctl restart tor@default || true
sleep 2
echo "Tor status: $(systemctl is-active tor 2>/dev/null || systemctl is-active tor@default 2>/dev/null || echo 'unknown')"

echo "=== [5/6] Allow tcpdump without password prompt ==="
echo "${CURRENT_USER} ALL=(ALL) NOPASSWD: /usr/bin/tcpdump" \
    > /etc/sudoers.d/tcpdump-collection
chmod 440 /etc/sudoers.d/tcpdump-collection

echo "=== [6/6] Python venv + dependencies ==="
cd "$REPO_ROOT"
if [ ! -d .venv ]; then
    sudo -u "$CURRENT_USER" python3 -m venv .venv
fi
sudo -u "$CURRENT_USER" .venv/bin/pip install --quiet \
    stem selenium dpkt numpy "requests[socks]"

echo ""
echo "=== Setup complete ==="
echo ""
IFACE=$(ip route | grep default | awk '{print $5}' | head -1)
echo "Network interface: ${IFACE}"
echo ""
echo "Verify:  .venv/bin/python -m crawler.verify"
echo ""
echo "Run crawl:"
echo "  sudo nohup env PYTHONUNBUFFERED=1 .venv/bin/python -m crawler.crawl_parallel \\"
echo "    --workers 10 --sites 95 --visits 300 --visit-start 350 \\"
echo "    --interface ${IFACE} > crawl.log 2>&1 &"
