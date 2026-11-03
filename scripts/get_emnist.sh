#!/usr/bin/env bash
# Download EMNIST Digits (240k train / 40k test, 10 classes) into data/ and decompress it.
# NIST only ships one ~560 MB zip with every split; we extract just the 4 digits files.
# Works in Git Bash on Windows, macOS, and Linux (Vanda: run on the login node).
#   bash scripts/get_emnist.sh
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
ZIP="emnist-gzip.zip"
# file name -> expected size in bytes after gunzip (header + payload)
FILES=(
    "emnist-digits-train-images-idx3-ubyte 188160016"
    "emnist-digits-train-labels-idx1-ubyte 240008"
    "emnist-digits-test-images-idx3-ubyte 31360016"
    "emnist-digits-test-labels-idx1-ubyte 40008"
)

size_ok() {  # name size
    [[ -f "$1" && "$(wc -c < "$1" | tr -d ' ')" == "$2" ]]
}

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

missing=()
for entry in "${FILES[@]}"; do
    read -r name size <<< "$entry"
    if size_ok "$name" "$size"; then
        echo "ok      $name (already present)"
    else
        missing+=("$name")
    fi
done
if [[ ${#missing[@]} -eq 0 ]]; then
    echo "EMNIST digits ready in $DATA_DIR"; exit 0
fi

if [[ ! -f "$ZIP" ]]; then
    echo "downloading $URL (~560 MB)..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 2 -o "$ZIP.part" "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$ZIP.part" "$URL"
    else
        echo "error: need curl or wget" >&2; exit 1
    fi
    mv "$ZIP.part" "$ZIP"   # only a complete download gets the final name
fi

entries=()
for name in "${missing[@]}"; do
    rm -f "$name" "$name.gz"
    entries+=("gzip/$name.gz")
done

# -j / --strip-components drop the gzip/ folder inside the archive
if command -v unzip >/dev/null 2>&1; then
    unzip -j -o -q "$ZIP" "${entries[@]}"
elif tar --version 2>/dev/null | grep -qi bsdtar; then
    tar -xf "$ZIP" --strip-components=1 "${entries[@]}"
else
    echo "error: need unzip or bsdtar to extract $ZIP" >&2; exit 1
fi

for entry in "${FILES[@]}"; do
    read -r name size <<< "$entry"
    [[ -f "$name.gz" ]] && gunzip -f "$name.gz"
    if ! size_ok "$name" "$size"; then
        echo "error: $name is $(wc -c < "$name" | tr -d ' ') bytes, expected $size" >&2; exit 1
    fi
    echo "ok      $name ($size bytes)"
done

rm -f "$ZIP"
echo "EMNIST digits ready in $DATA_DIR"
