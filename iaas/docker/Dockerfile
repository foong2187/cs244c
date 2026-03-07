# Base image with Python and system dependencies
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TOR_CONTROL_PASS=cs244c_collection
ENV PYTHONUNBUFFERED=1

# Install system dependencies (based on setup.sh)
RUN apt-get update -q && \
    apt-get install -y \
        sudo \
        tor \
        tcpdump \
        python3-pip \
        python3-venv \
        wget \
        curl \
        net-tools \
        iproute2 \
        software-properties-common \
        ca-certificates \
        gnupg \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Install Firefox from Mozilla PPA (avoiding snap issues)
RUN add-apt-repository -y ppa:mozillateam/ppa && \
    echo 'Package: *' > /etc/apt/preferences.d/mozilla-firefox && \
    echo 'Pin: release o=LP-PPA-mozillateam' >> /etc/apt/preferences.d/mozilla-firefox && \
    echo 'Pin-Priority: 1001' >> /etc/apt/preferences.d/mozilla-firefox && \
    apt-get update -q && \
    apt-get install -y firefox && \
    rm -rf /var/lib/apt/lists/*

# Install geckodriver
RUN GECKO_VER="v0.35.0" && \
    wget -q "https://github.com/mozilla/geckodriver/releases/download/${GECKO_VER}/geckodriver-${GECKO_VER}-linux64.tar.gz" && \
    tar -xzf "geckodriver-${GECKO_VER}-linux64.tar.gz" && \
    mv geckodriver /usr/local/bin/geckodriver && \
    chmod +x /usr/local/bin/geckodriver && \
    rm "geckodriver-${GECKO_VER}-linux64.tar.gz"

# Configure Tor
RUN TOR_PASS="cs244c_collection" && \
    HASHED=$(tor --hash-password "$TOR_PASS" | tail -1) && \
    echo "" >> /etc/tor/torrc && \
    echo "# WF data collection settings" >> /etc/tor/torrc && \
    echo "ControlPort 9051" >> /etc/tor/torrc && \
    echo "HashedControlPassword ${HASHED}" >> /etc/tor/torrc && \
    echo "MaxCircuitDirtiness 10" >> /etc/tor/torrc

# Create a non-root user for security
RUN useradd -m -s /bin/bash collector && \
    echo "collector ALL=(ALL) NOPASSWD: /usr/bin/tcpdump, /usr/sbin/service" > /etc/sudoers.d/collector && \
    chmod 440 /etc/sudoers.d/collector

# Set up Python environment
WORKDIR /app
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python3 -m venv .venv && \
    .venv/bin/pip install --no-cache-dir --upgrade pip && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt

# Additional dependencies for analysis
RUN .venv/bin/pip install --no-cache-dir matplotlib seaborn

# Copy application code
COPY data-collection/ ./

# Create necessary directories
RUN mkdir -p data/pcap data/pickle data/analysis && \
    chown -R collector:collector /app

# Switch to non-root user
USER collector

# Create startup script
RUN cat > start.sh << 'EOF'
#!/bin/bash
set -e

# Start Tor in the background
sudo service tor start

# Wait for Tor to bootstrap by checking SOCKS port (no systemd in Docker)
echo "Waiting for Tor to connect..."
for i in $(seq 1 30); do
    if ss -tnlp 2>/dev/null | grep -q ':9050'; then
        break
    fi
    sleep 1
done

# Execute the command passed to the container
exec "$@"
EOF

RUN chmod +x start.sh

# Set the entrypoint
ENTRYPOINT ["./start.sh"]

# Default command
CMD [".venv/bin/python", "collect.py", "--help"]