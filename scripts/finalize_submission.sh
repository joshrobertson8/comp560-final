#!/usr/bin/env bash
# Run AFTER training completes.
#
# Steps:
#   1. Copy best checkpoint into weights/ for model.py.
#   2. Generate dataset_a.csv using the trained StudentModel.
#   3. Run the official evaluate.py on Dataset A.
#   4. Build a zip of the submission package.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STUDENT_ID="${1:-joshrobertson}"
SUBMISSION_ZIP="${STUDENT_ID}_comp560_submission.zip"

if [ ! -f checkpoints/best_model.pth ]; then
    echo "ERROR: checkpoints/best_model.pth not found. Training may not be complete." >&2
    exit 1
fi

echo "=== 1) Copy best checkpoint to weights/ ==="
mkdir -p weights
cp checkpoints/best_model.pth weights/best_model.pth
echo "  weights/best_model.pth ($(du -h weights/best_model.pth | cut -f1))"

echo "=== 2) Self-test model.py ==="
python3 -c "
import torch
from model import StudentModel
m = StudentModel(device='mps')
x = torch.randn(4, 3, 224, 224)
e = m.encode(x)
print(f'  shape={tuple(e.shape)}  norms={e.norm(dim=1).tolist()}  dim={m.embedding_dim}')
"

echo "=== 3) Generate dataset_a predictions CSV ==="
python3 scripts/make_dataset_a_csv.py \
    --dataset_root datasets/dataset_a \
    --output predictions/dataset_a.csv \
    --device mps --batch_size 32 --num_workers 2

echo "=== 4) Evaluate on dataset_a ==="
python3 evaluate.py \
    --student_id "$STUDENT_ID" \
    --prediction predictions/dataset_a.csv \
    --datasets dataset_a

echo "=== 5) Build submission zip ==="
# Exclude large artifacts and local caches.
TMPDIR=$(mktemp -d)
PKG="$TMPDIR/${STUDENT_ID}_comp560_submission"
mkdir -p "$PKG"

cp -r \
    model.py \
    src \
    scripts \
    evaluate.py \
    requirements.txt \
    SUBMISSION_README.md \
    REPORT.md \
    "$PKG"/

mkdir -p "$PKG/weights"
cp weights/best_model.pth "$PKG/weights/"

mkdir -p "$PKG/predictions"
cp predictions/dataset_a.csv "$PKG/predictions/" 2>/dev/null || true

# Include training history (helpful for grader sanity)
if [ -f checkpoints/history.json ]; then
    mkdir -p "$PKG/checkpoints"
    cp checkpoints/history.json "$PKG/checkpoints/"
fi

rm -rf "$ROOT/$SUBMISSION_ZIP"
(cd "$TMPDIR" && zip -r "$ROOT/$SUBMISSION_ZIP" "${STUDENT_ID}_comp560_submission" -x "*__pycache__*" "*.pyc")
rm -rf "$TMPDIR"

echo ""
echo "Submission ready: $ROOT/$SUBMISSION_ZIP"
ls -lh "$ROOT/$SUBMISSION_ZIP"
