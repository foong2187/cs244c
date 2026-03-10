#!/usr/bin/env bash
# Generate all defended datasets from raw traces.
#
# End-to-end pipeline:
#   1. (Optional) Clone defense simulator repos
#   2. Apply each defense to raw traces -> defended trace dirs
#   3. Convert defended traces to DF pickle format
#
# Usage:
#   bash scripts/generate_all_datasets.sh [--traces-dir path] [--defenses bro regulator ...]
#
# Examples:
#   # All defenses (BRO, RegulaTor, Tamaraw, BuFLO):
#   bash scripts/generate_all_datasets.sh
#
#   # Only BRO and RegulaTor:
#   bash scripts/generate_all_datasets.sh --defenses bro regulator
#
#   # Custom traces directory:
#   bash scripts/generate_all_datasets.sh --traces-dir /path/to/traces
#
#   # Include WTF-PAD (requires setup_defenses.sh first):
#   bash scripts/setup_defenses.sh
#   bash scripts/generate_all_datasets.sh --defenses bro regulator tamaraw buflo wtfpad

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
TRACES_DIR="$PROJECT_DIR/data/collected/traces"
OUTPUT_BASE="$PROJECT_DIR/data/collected"
DEFENSES=("bro" "regulator" "tamaraw" "buflo")
SEED=42
SEQ_LENGTH=5000

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --traces-dir)
            TRACES_DIR="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --defenses)
            shift
            DEFENSES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DEFENSES+=("$1")
                shift
            done
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --include-wtfpad)
            DEFENSES+=("wtfpad")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Website Fingerprinting Defense Dataset Generator ==="
echo ""
echo "Traces dir:  $TRACES_DIR"
echo "Output base: $OUTPUT_BASE"
echo "Defenses:    ${DEFENSES[*]}"
echo "Seed:        $SEED"
echo ""

# Check traces exist
if [ ! -d "$TRACES_DIR" ]; then
    echo "ERROR: Traces directory not found: $TRACES_DIR"
    echo "Run data collection first or specify --traces-dir"
    exit 1
fi

TRACE_COUNT=$(find "$TRACES_DIR" -maxdepth 1 -type f ! -name '.*' | wc -l | tr -d ' ')
echo "Found $TRACE_COUNT raw trace files"
echo ""

if [ "$TRACE_COUNT" -eq 0 ]; then
    echo "ERROR: No trace files found in $TRACES_DIR"
    exit 1
fi

# Check if WTF-PAD repo is needed
for d in "${DEFENSES[@]}"; do
    if [ "$d" = "wtfpad" ]; then
        if [ ! -d "$PROJECT_DIR/defenses/repos/wtfpad" ]; then
            echo "WTF-PAD selected but repo not cloned. Running setup..."
            bash "$SCRIPT_DIR/setup_defenses.sh"
        fi
    fi
done

# Activate venv if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Run defense simulation + pickle conversion
echo "=== Starting defense simulation ==="
echo ""

python3 "$SCRIPT_DIR/simulate_defenses.py" \
    --traces-dir "$TRACES_DIR" \
    --output-base "$OUTPUT_BASE" \
    --defenses "${DEFENSES[@]}" \
    --pickle \
    --seed "$SEED" \
    --sequence-length "$SEQ_LENGTH"

echo ""
echo "=== Done! ==="
echo ""
echo "Generated datasets:"
echo ""

# Show what was produced
declare -A SUFFIX_MAP
SUFFIX_MAP=( ["bro"]="BRO" ["regulator"]="RegulaTor" ["tamaraw"]="Tamaraw" ["buflo"]="BuFLO" ["wtfpad"]="WTFPAD" )

for d in "${DEFENSES[@]}"; do
    suffix="${SUFFIX_MAP[$d]}"
    trace_dir="$OUTPUT_BASE/traces_${suffix}"
    pickle_dir="$OUTPUT_BASE/pickle_${suffix}"

    if [ -d "$trace_dir" ]; then
        tc=$(find "$trace_dir" -maxdepth 1 -type f ! -name '.*' | wc -l | tr -d ' ')
        echo "  $d:"
        echo "    Defended traces: $trace_dir ($tc files)"
    fi
    if [ -d "$pickle_dir" ]; then
        echo "    Pickle files:    $pickle_dir"
        ls -1 "$pickle_dir"/*.pkl 2>/dev/null | sed 's/^/      /'
    fi
    echo ""
done

echo "You can now train/evaluate the DF model on each defended dataset."
