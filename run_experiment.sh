#!/usr/bin/env bash
set -euo pipefail

IMAGE="cs536hw2"

echo "[1/3] Building Docker image..."
docker build -t "$IMAGE" .

echo "[2/3] Creating output directories..."
mkdir -p data plots

echo "[3/3] Running experiment..."
docker run --rm \
  --network host \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/plots:/app/plots" \
  "$IMAGE"

echo "Done. Results in ./data and ./plots"
