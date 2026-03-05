#!/bin/bash
# Merge crawler data from multiple machines into one dataset.
#
# After all machines finish crawling, run this on the machine where you
# want to process and train (e.g., your dorm PC with the GPU).
#
# Usage:
#   bash crawler/merge_machines.sh
#
# Before running, edit the MACHINES array below with your actual SSH hosts
# and the GCE zone if applicable.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PCAP_DIR="$REPO_ROOT/data/crawler-pcap"
TRACES_DIR="$REPO_ROOT/data/crawler-traces"
MERGED_PROGRESS="$TRACES_DIR/progress.csv"

mkdir -p "$PCAP_DIR" "$TRACES_DIR"

# --- Configure these for your machines ---
# Format: "type:host_or_name:zone(optional)"
# type: "ssh" for regular SSH, "gce" for gcloud compute scp
MACHINES=(
    "ssh:ohio:~/cs244c"
    # "gce:wf-crawler:us-central1-a:~/cs244c"
)
# ------------------------------------------

echo "=== Merging crawler data from remote machines ==="
echo ""

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

PROGRESS_FILES=()

# Local progress (if exists)
if [ -f "$MERGED_PROGRESS" ]; then
    echo "[local] Found $(tail -n+2 "$MERGED_PROGRESS" | wc -l) rows in progress.csv"
    PROGRESS_FILES+=("$MERGED_PROGRESS")
fi

for machine in "${MACHINES[@]}"; do
    IFS=':' read -r mtype host zone rpath <<< "$machine"
    rpath="${rpath:-~/cs244c}"
    label="${host}"

    echo ""
    echo "[${label}] Fetching data..."

    remote_progress="$TEMP_DIR/progress_${label}.csv"

    if [ "$mtype" = "ssh" ]; then
        echo "  scp ${host}:${rpath}/data/crawler-traces/progress.csv"
        scp -q "${host}:${rpath}/data/crawler-traces/progress.csv" "$remote_progress" || {
            echo "  WARN: could not fetch progress.csv from ${host}, skipping"
            continue
        }
        echo "  rsync pcaps from ${host}..."
        rsync -az --progress "${host}:${rpath}/data/crawler-pcap/" "$PCAP_DIR/" || {
            echo "  WARN: rsync pcaps failed from ${host}"
        }

    elif [ "$mtype" = "gce" ]; then
        echo "  gcloud scp from ${host} (zone=${zone})"
        gcloud compute scp "${host}:${rpath}/data/crawler-traces/progress.csv" \
            "$remote_progress" --zone="$zone" -q || {
            echo "  WARN: could not fetch progress.csv from ${host}, skipping"
            continue
        }
        echo "  rsync pcaps from ${host}..."
        gcloud compute scp --recurse "${host}:${rpath}/data/crawler-pcap/*" \
            "$PCAP_DIR/" --zone="$zone" -q || {
            echo "  WARN: rsync pcaps failed from ${host}"
        }
    fi

    rows=$(tail -n+2 "$remote_progress" | wc -l)
    echo "  [${label}] ${rows} rows in progress.csv"
    PROGRESS_FILES+=("$remote_progress")
done

echo ""
echo "=== Merging progress CSVs ==="

# Merge: take header from first file, then data rows from all
MERGED_TEMP="$TEMP_DIR/merged_progress.csv"
first=true
for pf in "${PROGRESS_FILES[@]}"; do
    if $first; then
        head -1 "$pf" > "$MERGED_TEMP"
        first=false
    fi
    tail -n+2 "$pf" >> "$MERGED_TEMP"
done

cp "$MERGED_TEMP" "$MERGED_PROGRESS"

total_ok=$(grep -c ',ok,' "$MERGED_PROGRESS" || echo 0)
total_rows=$(tail -n+2 "$MERGED_PROGRESS" | wc -l)
total_pcaps=$(ls "$PCAP_DIR"/*.pcap 2>/dev/null | wc -l)

echo ""
echo "=== Merge complete ==="
echo "  Progress rows : ${total_rows}"
echo "  Successful    : ${total_ok}"
echo "  Pcap files    : ${total_pcaps}"
echo "  Progress CSV  : ${MERGED_PROGRESS}"
echo "  Pcap dir      : ${PCAP_DIR}"
echo ""
echo "Next steps:"
echo "  .venv/bin/python -m crawler.process"
echo "  .venv/bin/python -m crawler.analyze"
echo "  .venv/bin/python src/train_closed_world.py --defense NoDef --epochs 30"
