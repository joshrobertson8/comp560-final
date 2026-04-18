# COMP 560 — Object Re-Identification

**Student:** Josh Robertson
**Entry point:** `model.py` — `StudentModel` class.

## Quick start (grader)

```bash
pip install -r requirements.txt
python -c "
import torch
from model import StudentModel
m = StudentModel(device='cuda')  # falls back to mps / cpu
x = torch.randn(4, 3, 224, 224)   # ImageNet-normalized input expected
e = m.encode(x)
print(e.shape, e.norm(dim=1))     # (4, 256), unit norms
print('embedding_dim:', m.embedding_dim)
"
```

The model weights live at `weights/best_model.pth` and are loaded relative to
`model.py` — you can import `StudentModel` from any working directory.

## Interface (matches project spec)

```python
class StudentModel:
    def __init__(self, device: str = "cuda"): ...
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) ImageNet-normalized -> (B, 256) L2-normed."""
    @property
    def embedding_dim(self) -> int: ...   # 256
```

- Device resolution: tries the requested device, falls back cuda → mps → cpu.
- Precision: fp16 on CUDA, bf16 on MPS, fp32 on CPU (autocast).
- Outputs are always float32 and L2-normalized.

## Approach in 3 bullets

- **Backbone**: DINOv2 ViT-S/14 (SSL-pretrained on LVD-142M) with the first 6 of 12
  transformer blocks frozen to preserve domain-general features (helps Dataset B).
- **Head**: BNNeck (Linear 384→256 + BatchNorm1d) returning 256-dim L2-normalized
  embeddings at inference (Luo et al., 2019).
- **Training**: CosFace (m=0.35, s=30) + batch-hard triplet (margin 0.3), PK sampling
  with two-level sub-dataset balancing across the 37 OpenAnimals sub-datasets, 25
  epochs with cosine schedule, bf16 autocast on MPS.

## Repository layout

```
.
├── model.py                        # StudentModel entry point
├── weights/
│   └── best_model.pth              # Trained checkpoint
├── src/
│   ├── backbone.py                 # DINOv2 / ConvNeXt-Tiny loaders
│   ├── head.py                     # BNNeck + ReIDModel wrapper
│   ├── losses.py                   # CosFace + TripletHard
│   ├── sampler.py                  # Two-level (sub-dataset, identity) PK sampler
│   ├── dataset.py                  # ReIDDataset + cross-domain holdout split
│   ├── augment.py                  # Train/eval transforms
│   └── train.py                    # Training loop
├── scripts/
│   ├── benchmark_backbone.py       # Throughput benchmark
│   ├── make_dataset_a_csv.py       # Generates ranked CSV for local eval
│   ├── extract_data.sh             # Untars train + test image archives
│   └── train.sh                    # One-shot training entry
├── evaluate.py                     # Provided evaluator (unchanged)
├── requirements.txt
└── SUBMISSION_README.md            # This file
```

## Reproducing results

1. `bash scripts/extract_data.sh` — unpack train+test images into `datasets/dataset_a/`.
2. `bash scripts/train.sh` — trains for 25 epochs and saves `checkpoints/best_model.pth`.
3. `cp checkpoints/best_model.pth weights/best_model.pth`
4. `python scripts/make_dataset_a_csv.py --device mps --output predictions/dataset_a.csv`
5. `python evaluate.py --student_id josh --prediction predictions/dataset_a.csv --datasets dataset_a`

Cross-domain validation: during training the script holds out three
sub-datasets (AAUZebraFish, PolarBearVidID, SMALST) to monitor
generalization, approximating the unseen Dataset B domain gap.
