#!/bin/bash
set -euxo pipefail

PROJECT_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

source "$PROJECT_DIR/.env"

MODEL_DIR="$PROJECT_DIR/models"
LOCAL_WEIGHTS_DIR="$MODEL_DIR/$MODEL_NAME"

mkdir -p "$LOCAL_WEIGHTS_DIR"

# Download model weights
echo "Downloading $MODEL_NAME..."
huggingface-cli download $MODEL_NAME --local-dir "$LOCAL_WEIGHTS_DIR"

echo "Done!"