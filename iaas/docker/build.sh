#!/bin/bash
# Build script for WF data collection Docker image

set -e

# Always run from the directory containing this script so relative paths work
cd "$(dirname "$0")"

# Configuration
PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
IMAGE_NAME="wf-data-collection"
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

# Copy the data-collection directory into the build context, clean up on exit
cp -r ../../data-collection .
trap 'rm -rf data-collection/' EXIT

# Build the Docker image for linux/amd64 (GKE nodes are x86_64)
docker buildx build --platform linux/amd64 -t "${FULL_IMAGE_NAME}" --load .

echo "Docker image built successfully: ${FULL_IMAGE_NAME}"

# Optionally push to GCR
if [ "$1" = "push" ]; then
    echo "Pushing image to Google Container Registry..."
    
    # Configure Docker to use gcloud as a credential helper
    gcloud auth configure-docker
    
    # Push the image
    docker push "${FULL_IMAGE_NAME}"
    
    echo "Image pushed successfully to GCR"
    echo "Image URL: ${FULL_IMAGE_NAME}"
fi

echo "Build completed!"
echo ""
echo "Usage examples:"
echo "  Local run: docker run --rm -it ${FULL_IMAGE_NAME} .venv/bin/python collect.py --help"
echo "  Push to GCR: $0 push"