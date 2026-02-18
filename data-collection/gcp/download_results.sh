#!/usr/bin/env bash
# Download collection results from a GCP VM to your local machine.
#
# Usage:
#   bash gcp/download_results.sh VM_NAME ZONE [PROJECT]
#
# Examples:
#   bash gcp/download_results.sh df-collector-1 us-central1-a
#   bash gcp/download_results.sh df-collector-1 us-central1-a my-gcp-project
#
# Downloads data/collected/ (pcap, traces, pickle, progress.csv, logs)
# to a local directory: ./collected-VM_NAME/
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash gcp/download_results.sh VM_NAME ZONE [PROJECT]"
    echo ""
    echo "Examples:"
    echo "  bash gcp/download_results.sh df-collector-1 us-central1-a"
    echo "  bash gcp/download_results.sh df-collector-1 us-central1-a my-gcp-project"
    exit 1
fi

VM_NAME="$1"
ZONE="$2"
PROJECT_FLAG=""
if [ $# -ge 3 ]; then
    PROJECT_FLAG="--project $3"
fi

LOCAL_DIR="./collected-${VM_NAME}"
REMOTE_DIR="~/df-website-fingerprinting/data/collected"

echo "=== Downloading results from ${VM_NAME} ==="
echo "  Zone: $ZONE"
echo "  Remote: $REMOTE_DIR"
echo "  Local:  $LOCAL_DIR"
echo ""

mkdir -p "$LOCAL_DIR"

# Download progress.csv and logs first (small files, quick check)
echo "[1/4] Downloading progress.csv and logs..."
gcloud compute scp $PROJECT_FLAG --zone="$ZONE" \
    "${VM_NAME}:${REMOTE_DIR}/progress.csv" \
    "$LOCAL_DIR/" 2>/dev/null || echo "  No progress.csv found."

gcloud compute scp $PROJECT_FLAG --zone="$ZONE" \
    "${VM_NAME}:${REMOTE_DIR}/collection.log" \
    "$LOCAL_DIR/" 2>/dev/null || echo "  No collection.log found."

# Show progress summary
if [ -f "$LOCAL_DIR/progress.csv" ]; then
    TOTAL=$(tail -n +2 "$LOCAL_DIR/progress.csv" | wc -l | tr -d ' ')
    SUCCESS=$(grep -c ",success," "$LOCAL_DIR/progress.csv" || true)
    FAILED=$(grep -c ",failed," "$LOCAL_DIR/progress.csv" || true)
    echo "  Progress: $SUCCESS successful, $FAILED failed, $TOTAL total visits"
fi

# Download pickle files (final output, most important)
echo ""
echo "[2/4] Downloading pickle files..."
gcloud compute scp $PROJECT_FLAG --zone="$ZONE" --recurse \
    "${VM_NAME}:${REMOTE_DIR}/pickle/" \
    "$LOCAL_DIR/pickle/" 2>/dev/null || echo "  No pickle files found."

# Download trace files
echo ""
echo "[3/4] Downloading trace files..."
gcloud compute scp $PROJECT_FLAG --zone="$ZONE" --recurse \
    "${VM_NAME}:${REMOTE_DIR}/traces/" \
    "$LOCAL_DIR/traces/" 2>/dev/null || echo "  No trace files found."

# Download PCAPs (large — ask first)
echo ""
PCAP_COUNT=$(gcloud compute ssh $PROJECT_FLAG --zone="$ZONE" "$VM_NAME" \
    --command "ls ${REMOTE_DIR}/pcap/*.pcap 2>/dev/null | wc -l" 2>/dev/null || echo "0")
PCAP_COUNT=$(echo "$PCAP_COUNT" | tr -d '[:space:]')

if [ "$PCAP_COUNT" -gt 0 ]; then
    echo "[4/4] Found $PCAP_COUNT PCAP files on the VM."
    echo "  PCAPs can be very large (50+ GB total). Download them?"
    read -p "  Download PCAPs? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Downloading PCAPs (this may take a while)..."
        gcloud compute scp $PROJECT_FLAG --zone="$ZONE" --recurse \
            "${VM_NAME}:${REMOTE_DIR}/pcap/" \
            "$LOCAL_DIR/pcap/"
    else
        echo "  Skipping PCAP download."
    fi
else
    echo "[4/4] No PCAP files found on VM."
fi

echo ""
echo "=== Download Complete ==="
echo "Results saved to: $LOCAL_DIR/"
ls -la "$LOCAL_DIR/"
