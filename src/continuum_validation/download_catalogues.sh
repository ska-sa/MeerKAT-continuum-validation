#!/bin/sh

# Download all reference catalogues from the mirror directory.
MIRROR_URL="http://sdp-services.kat.ac.za/mirror/catalogues/"

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/catalogues"

# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Downloading reference catalogues from $MIRROR_URL to $OUTPUT_DIR..."

wget -c -r -np -nH --cut-dirs=2 -R "index.html*" -P "$OUTPUT_DIR" "$MIRROR_URL"

echo "Catalogues download complete."
