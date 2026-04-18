#!/usr/bin/env bash
# One-shot training entry.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m src.train \
    --data_root datasets/dataset_a \
    --parquet_file train.parquet \
    --save_dir checkpoints \
    --backbone dinov2_vits14 \
    --embed_dim 256 \
    --freeze_first_n 6 \
    --p 16 --k 4 \
    --epochs 25 --warmup_epochs 2 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --weight_decay 1e-4 \
    --triplet_weight 1.0 --triplet_margin 0.3 \
    --cosface_s 30.0 --cosface_m 0.35 \
    --label_smoothing 0.1 \
    --image_size 224 \
    --num_workers 2 \
    --device mps \
    --eval_every 2 --save_every 5 \
    "$@"
