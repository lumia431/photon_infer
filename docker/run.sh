#!/bin/bash

################################################################################
# Run PhotonInfer Docker Container
################################################################################

set -e

IMAGE_NAME="${IMAGE_NAME:-photon_infer:latest}"
MODEL_DIR="${MODEL_DIR:-$HOME/.llama/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

mkdir -p "$RESULTS_DIR"

echo "Running PhotonInfer container..."
echo ""

docker run \
    --rm \
    --gpus all \
    -it \
    -v "$MODEL_DIR:/models:ro" \
    -v "$RESULTS_DIR:/app/results" \
    "$IMAGE_NAME" \
    bash
