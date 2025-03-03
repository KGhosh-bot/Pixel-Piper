#!/bin/bash

set -x  # Enable debug mode

PHYS_DIR="/home/ghosh/MaskedLogic"

# Print received parameters
echo "run_docker.sh received parameters:"
echo "1: $1"
echo "2: $2"
echo "3: $3"
echo "4: $4"

# Split the prompt array string back into an array
IFS='|' read -r -a prompt_array <<< "$4"

# Print the prompt array
echo "Prompt array: ${prompt_array[@]}"

docker run \
    -v "$PHYS_DIR":/workspace \
    --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    mlogic-image \
    "/workspace/train.sh" \
    "$1" "$2" "$3" "${prompt_array[@]}"