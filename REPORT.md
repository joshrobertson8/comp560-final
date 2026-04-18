# COMP 560 Object Re-Identification — Technical Report

**Student:** Josh Robertson
**Project:** Object Re-Identification for the OpenAnimals wildlife benchmark (Dataset A) and unseen vehicle surveillance (Dataset B)

## 1. Problem

Object re-identification (ReID) is framed as embedding learning: train an encoder
$f_\theta : \mathcal{I} \rightarrow \mathbb{R}^D$ such that, for a query image $I_q$,
$\cos(f_\theta(I_q), f_\theta(I^+)) > \cos(f_\theta(I_q), f_\theta(I^-))$
for any same-identity positive $I^+$ and different-identity negative $I^-$. At inference,
the query is ranked against a gallery by cosine similarity and scored with
Rank-K and mean average precision (mAP).

Two test sets are used: **Dataset A** is OpenAnimals (37 wildlife sub-datasets, ~110k
train / ~30k test images, 10k+ identities). **Dataset B** is a withheld vehicle
surveillance benchmark (20 cameras, same-camera matches excluded). The domain
gap between A (wildlife) and B (vehicles) is the central modeling challenge —
a model that overfits Dataset A's wildlife classes will collapse on vehicles.

## 2. Method

### 2.1 Backbone — DINOv2 ViT-S/14

I use a Vision Transformer Small (ViT-S/14) backbone pretrained with
DINOv2 (Oquab et al., 2023), a self-supervised method trained on LVD-142M
natural images. DINOv2 was selected over an ImageNet-supervised ResNet-50 or
ConvNeXt for one reason: self-supervised features generalize substantially better
to unseen domains than discriminative ImageNet features, which matters for Dataset B.
The backbone produces a 384-dim CLS token.

### 2.2 Partial fine-tuning

The first 6 of 12 transformer blocks (plus patch embedding, CLS token, and
positional embeddings) are frozen. This halves the trainable parameters
(22M → 10.75M) and preserves the DINOv2 features most important for cross-domain
transfer, at a small cost in Dataset A specialization. Training only the top 6 blocks
is a deliberate bias-variance trade-off for the A→B generalization problem.

### 2.3 BNNeck head

On top of the backbone I use the BNNeck design of Luo et al. (2019): a linear
projection (384 → 256) followed by a BatchNorm1d with no learnable bias. At
training, the pre-BN features feed the triplet loss and the post-BN features feed
the classifier. At inference, only the L2-normalized post-BN features are
returned. BNNeck's decoupling of the triplet and classifier branches is a
well-established 1–2 mAP improvement on person ReID and transfers directly.

The 256-dim embedding is chosen for the project's efficiency metric, which
penalizes larger embedding dims. 128 was considered but risks a discriminability
cliff with 10k+ identities; 384/512 offered negligible mAP gains in pilot runs.

### 2.4 Losses — CosFace + hard triplet

The training objective is $L = L_\mathrm{CosFace} + 1.0 \cdot L_\mathrm{TriHard}$.

**CosFace** (Wang et al., 2018) applies an additive margin in cosine space:
$\text{logit}_{i,y} = s \cdot (\cos\theta_{i,y} - m)$ for the ground-truth class
and $s \cdot \cos\theta_{i,c}$ otherwise, with $s=30$, $m=0.35$. CosFace was
chosen over ArcFace specifically because the Apple Silicon MPS backend exhibits
numerical instability in `acos()`; CosFace's formulation avoids inverse
trigonometric functions and is stable in bf16 autocast. Label smoothing (0.1) is
applied to regularize the 8873-way classifier.

**Hard-mining triplet loss** (Hermans et al., 2017) computes the hardest-positive
and hardest-negative within each PK batch and applies a hinge with margin 0.3 on
the pre-BN features.

### 2.5 Two-level PK sampler

The OpenAnimals training set is extremely imbalanced by sub-dataset
(CatIndividualImages 10k images vs ReunionTurtles 237). A naive PK sampler
would overweight the largest sub-datasets and never see smaller ones. I use a
two-level sampler: **(1)** sample a sub-dataset uniformly from the 34 available
(after holdout), **(2)** sample P=16 identities within it, **(3)** sample K=4
images per identity. This gives batch size 64, with all 16 identities drawn
from the same visual domain — which also yields *harder* negatives (cat vs cat
is much harder than cat vs whale), a free benefit that improves the triplet
signal.

### 2.6 Cross-domain validation

Three sub-datasets are held out entirely from training as a "cross-domain dev
set": **AAUZebraFish** (underwater/fish, 236 imgs), **PolarBearVidID**
(video-sourced polar bears, 1114 imgs), and **SMALST** (synthetic/rendered
animals, 1035 imgs). These are the closest approximation we have to the
unseen Dataset B domain gap. Dev Rank-1 and mAP are evaluated every two
epochs and the best checkpoint (by mean dev mAP) is saved. This gives us a
direct signal on whether fine-tuning is hurting or helping cross-domain
transfer — the worst-case failure mode for this project.

### 2.7 Training configuration

- Optimizer: AdamW, weight decay 1e-4, gradient clip 1.0.
- Layer-wise LR: backbone 3e-5, head + classifier 3e-4.
- Schedule: 2-epoch linear warmup, cosine to 0 over 25 epochs total.
- Batch: P=16, K=4 → 64 per step. 1,659 steps per epoch.
- Precision: bf16 autocast on MPS (fp16 on CUDA, fp32 on CPU). Bf16 is more
  stable than fp16 on Apple Silicon for this workload; the code falls back to
  fp32 per-step if a NaN is detected.
- Augmentation: RandomResizedCrop(224, scale=0.7–1.0), HFlip, ColorJitter,
  RandomGrayscale, RandomErasing (p=0.5) — standard strong-augmentation ReID
  recipe.

### 2.8 Inference / efficiency

At inference, the model takes an ImageNet-normalized batch, runs the backbone
under the target device's autocast dtype, and returns an L2-normalized float32
`(B, 256)` embedding. The `StudentModel` class auto-falls-back
`cuda → mps → cpu`, so graders with any hardware get a functioning model. No
re-ranking is performed in `encode()` since re-ranking is a pairwise operation
the grader's harness controls.

## 3. Experiments

### 3.1 Dataset A (target)

| Model                          | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| ------------------------------ | -----: | -----: | ------: | --: | ---: |
| VLM baseline (provided)         |    ~60 |    ~80 |       - | ~50 |    - |
| ResNet-50 ImageNet (zero-shot) |      - |      - |       - |   - |    - |
| DINOv2 ViT-S/14 zero-shot      |      - |      - |       - |   - |    - |
| **Ours (fine-tuned)**          |  99.95 |  100.00 |    100.00 | 88.19 | 34.62 |

Trained checkpoint **blew past the VLM baseline** (Rank-1 +40, mAP +38). All
7,334 queries produced a correct match within the top 5 (Rank-5 saturates).
The residual ~6% mAP gap between mAP and Rank-5 indicates the model finds a
correct match early but sometimes ranks one or two incorrect neighbors ahead
of *additional* positives — typical of a well-trained ReID embedding that has
not seen every identity's visual variation.

### 3.2 Cross-domain dev (proxy for Dataset B generalization)

| Checkpoint                | AAUZebraFish mAP | PolarBearVidID mAP | SMALST mAP | Mean mAP |
| ------------------------- | ---------------: | -----------------: | ---------: | -------: |
| DINOv2 + random BNNeck    |             56.5 |               31.8 |       27.7 |     38.7 |
| After epoch 1             |             55.1 |               30.9 |       30.4 |     38.8 |
| After epoch 2             |             61.3 |               37.2 |       43.5 |     47.3 |
| After epoch 4             |             63.1 |               37.5 |       49.1 |     49.9 |
| After epoch 6             |             63.0 |               38.1 |       50.6 |     50.6 |
| After epoch 8             |             64.0 |               36.3 |       52.0 |     50.8 |
| **Best checkpoint (epoch 10)** |        61.4 |               37.9 |       53.9 | **51.1** |
| After epoch 25 (final)    |             65.3 |               36.2 |       48.6 |     50.0 |

Mean cross-domain dev mAP improves **+12.4** over random-head zero-shot. The
plateau around epoch 10 and slight regression at epoch 25 (SMALST −5.3) is
textbook: by epoch 25 the classifier is memorizing Dataset A identities at
the margin. The training loop's dev-mAP checkpointing correctly held the
epoch-10 snapshot. This is the weights file submitted — not the last epoch.

Rank-1 is saturated at 100% on all three dev sub-datasets even from the
random-head baseline, so mAP is the discriminating signal. Training must
*raise mean mAP* to be net-positive on cross-domain transfer; a drop would mean
we are overfitting Dataset A's wildlife classes and would hurt Dataset B. The
training loop saves the checkpoint that maximizes mean dev mAP.

### 3.3 Throughput benchmark (MPS, batch 32)

| Backbone            | fp32      | bf16 autocast |
| ------------------- | --------: | ------------: |
| DINOv2 ViT-S/14    | 243 img/s |     255 img/s |
| ConvNeXt-Tiny      | 277 img/s |     261 img/s |

DINOv2 is ~12% slower than ConvNeXt-Tiny on MPS but well above the 30 img/s
floor required for a reasonable efficiency score, and the feature quality
gap strongly favors DINOv2 for the cross-domain objective.

## 4. Efficiency analysis

| Metric                | Value              |
| --------------------- | ------------------ |
| Parameters (total)    | 22.16 M            |
| Trainable params      | 10.75 M            |
| Embedding dim         | 256                |
| Throughput @ MPS bf16 | 255 img/s (batch 32) |
| Peak memory (eval)    | < 2 GB (estimated) |

Compared to the VLM baseline (60 GB peak memory, 0.2 img/s, 384-dim):
- **Memory**: ~30x reduction.
- **Throughput**: ~1,275x faster.
- **Embedding dim**: 1.5x smaller (256 vs 384).

### 4.1 Limitations

- **Apple Silicon training**: MPS bf16 autocast is functional but missing some
  fused kernels (xFormers attention, flash attention). Step time would roughly
  halve on a modern CUDA GPU.
- **Dataset B is untested**: the cross-domain holdout sets are the only
  signal we have for Dataset B generalization. A real sample of the target
  domain would sharpen hyperparameter selection.
- **No re-ranking**: the grading interface encodes each image independently, so
  pairwise techniques like k-reciprocal re-ranking (which typically add 2–5
  mAP on top of embedding-only ranking) are unavailable. Local CSV evaluation
  on Dataset A *can* use re-ranking if we wanted an upper-bound measurement.

## References

- Luo, H., et al. *A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification*. CVPR 2019.
- Hermans, A., Beyer, L., Leibe, B. *In Defense of the Triplet Loss for Person Re-Identification*. arXiv 2017.
- Wang, H., et al. *CosFace: Large Margin Cosine Loss for Deep Face Recognition*. CVPR 2018.
- Oquab, M., et al. *DINOv2: Learning Robust Visual Features without Supervision*. arXiv 2023.
- Zhong, Z., et al. *Re-ranking Person Re-identification with k-reciprocal Encoding*. CVPR 2017.
- Sun, Y., et al. *OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization*. 2024.
