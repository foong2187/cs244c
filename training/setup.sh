#!/bin/bash
# Setup script for the GPU training VM.
#
# Intended for a GCP instance created with:
#   - Deep Learning VM image (Debian 11, CUDA 12 pre-installed)
#   - Machine type: n1-standard-4 + 1x NVIDIA T4
#   - 50GB disk
#
# Run once after SSHing in:
#   bash setup.sh

set -e

echo "=== Setting up training environment ==="

# Install Python venv support if needed
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip

# Create venv
python3 -m venv .venv
echo "Virtual environment created."

# Install dependencies
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r training/requirements.txt

echo ""
echo "=== Verifying GPU ==="
.venv/bin/python - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  GPU detected: {gpus}")
    print("  TensorFlow will use GPU for training.")
else:
    print("  WARNING: No GPU detected. Training will run on CPU (slow).")
    print("  Make sure you created the VM with a GPU and installed drivers.")
EOF

echo ""
echo "=== Setup complete ==="
echo "Next: copy your dataset pickle files, then run:"
echo "  .venv/bin/python src/train_closed_world.py --defense NoDef --save_model"
