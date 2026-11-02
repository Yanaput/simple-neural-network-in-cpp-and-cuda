#!/usr/bin/env bash
# Download MNIST into data/ (relative to the repo root) and decompress it.
# Works in Git Bash on Windows and on Linux (Vanda). Uses curl, falls back to wget.
#   bash scripts/get_mnist.sh
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
MIRRORS=(
    "https://ossci-datasets.s3.amazonaws.com/mnist"
    "https://storage.googleapis.com/cvdf-datasets/mnist"
)
# file name -> expected size in bytes after gunzip (header + payload)
FILES=(
    "train-images-idx3-ubyte 47040016"
    "train-labels-idx1-ubyte 60008"
    "t10k-images-idx3-ubyte 7840016"
    "t10k-labels-idx1-ubyte 10008"
)

download() {  # url out
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 2 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        echo "error: need curl or wget" >&2; exit 1
    fi
}

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

for entry in "${FILES[@]}"; do
    read -r name size <<< "$entry"

    if [[ -f "$name" && "$(wc -c < "$name" | tr -d ' ')" == "$size" ]]; then
        echo "ok      $name (already present)"
        continue
    fi

    rm -f "$name" "$name.gz"
    for m in "${MIRRORS[@]}"; do
        if download "$m/$name.gz" "$name.gz"; then break; fi
        echo "warning: $m failed for $name, trying next mirror" >&2
        rm -f "$name.gz"
    done
    if [[ ! -f "$name.gz" ]]; then
        echo "error: could not download $name.gz from any mirror" >&2; exit 1
    fi

    gunzip -f "$name.gz"

    actual="$(wc -c < "$name" | tr -d ' ')"
    if [[ "$actual" != "$size" ]]; then
        echo "error: $name is $actual bytes, expected $size" >&2; exit 1
    fi
    echo "ok      $name ($size bytes)"
done

echo "MNIST ready in $DATA_DIR"
