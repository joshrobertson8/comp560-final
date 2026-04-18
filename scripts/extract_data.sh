#!/usr/bin/env bash
# Extract the training and test image tarballs into datasets/dataset_a/.
# Both tarballs start with an 'images/' prefix and do not conflict.
# Safe to re-run: tar will overwrite in place.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Copying train parquet -> datasets/dataset_a/train.parquet"
cp -n or_dataset_a_train.parquet datasets/dataset_a/train.parquet || true

if [ -f datasets/dataset_a/images.tar.gz ]; then
    echo "Extracting test images..."
    tar -xzf datasets/dataset_a/images.tar.gz -C datasets/dataset_a/
fi

if [ -f or_train_images.tar.gz ]; then
    echo "Extracting train images (this may take 20-40 minutes)..."
    tar -xzf or_train_images.tar.gz -C datasets/dataset_a/
fi

echo "Done. Total images:"
find datasets/dataset_a/images -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \) | wc -l
