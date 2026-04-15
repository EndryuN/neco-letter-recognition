#!/usr/bin/env bash
# Download the UCI Letter Recognition dataset into ./data.
# Usage: bash scripts/download_data.sh

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
TARGET="$DATA_DIR/letter-recognition.data"

mkdir -p "$DATA_DIR"

if [[ -f "$TARGET" ]]; then
    echo "Dataset already exists at $TARGET — skipping download."
    exit 0
fi

echo "Downloading UCI Letter Recognition dataset to $TARGET"
if command -v wget >/dev/null 2>&1; then
    wget -O "$TARGET" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$TARGET" "$URL"
else
    echo "Error: neither wget nor curl is available." >&2
    exit 1
fi
echo "Done."
