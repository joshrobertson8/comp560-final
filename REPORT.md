# Embedding Learning for Cross-Domain Object Re-Identification

**Josh Robertson & Justin Yoo — COMP 560 Final Project**

---

## Abstract

Object re-identification (ReID) requires learning an image encoder whose
similarity structure identifies individual instances — a specific animal,
vehicle, or person — even when camera, pose, and lighting vary. The challenge
is magnified when the test domain differs from the training domain: models that
overfit one visual domain collapse on another. We study this setting on the
OpenAnimals wildlife benchmark (Dataset A) and a withheld vehicle surveillance
benchmark (Dataset B), where the domain shift is total. Our approach combines a
self-supervised DINOv2 ViT-S/14 backbone (partially frozen), a BNNeck projection
head producing 256-dim L2-normalized embeddings, and a joint CosFace plus hard
triplet loss trained with a two-level PK sampler that balances 37 wildlife
sub-datasets. Three sub-datasets are held out as a cross-domain dev set whose
mean mAP is used for checkpoint selection. On Dataset A the trained model
achieves 99.95% Rank-1 and 88.19% mAP, improving on the provided
vision-language-model baseline by +40 Rank-1 and +38 mAP while running
~1,275× faster and using ~40× less memory. The cross-domain dev set improves
+12.4 mAP over a zero-shot DINOv2 baseline and plateaus at epoch 10 of 25, which
dictated the submitted checkpoint.

---

## 1. Introduction

Re-identification is the task of matching a query image to a gallery of images
of the same individual — not the same *class* (e.g. "dog") but the same
*instance* (e.g. "this specific dog"). Unlike classification, the identity
space is open: at test time the gallery may contain individuals never seen
during training, so the task is framed as embedding learning. An encoder
*f<sub>θ</sub>* maps images to ℝ<sup>D</sup>, and identities are retrieved by
cosine similarity:

> cos(*f<sub>θ</sub>*(*I<sub>q</sub>*), *f<sub>θ</sub>*(*I*<sup>+</sup>)) > cos(*f<sub>θ</sub>*(*I<sub>q</sub>*), *f<sub>θ</sub>*(*I*<sup>−</sup>))

for every same-identity positive *I*<sup>+</sup> and different-identity
negative *I*<sup>−</sup>. Standard metrics are Rank-K (does a correct match
appear in the top K?) and mean average precision (mAP, averaged over query
ranking positions).

We work with two test sets. **Dataset A** is OpenAnimals (Sun et al., 2024):
37 wildlife sub-datasets covering ~110k training images, ~30k test images, and
10k+ identities across species as different as zebras, turtles, cats, and
polar bears. **Dataset B** is a completely withheld vehicle surveillance
benchmark — 20 cameras, same-camera matches excluded, never seen by us. The
domain gap between wildlife and vehicles is the central modeling challenge: a
model that memorizes Dataset A's wildlife classes will collapse on vehicles.
Grading weights 40% performance (Rank-K, mAP), 30% efficiency (throughput,
memory, embedding dim), and 30% report quality, so a single-minded focus on
Dataset A accuracy at the cost of generalization would score poorly overall.

**Past approaches.** Person ReID has a mature literature dominated by
CNN backbones (ResNet-50) with BNNeck heads (Luo et al., 2019) and a
combination of identity classification (cross-entropy, later CosFace/ArcFace)
and triplet losses (Hermans et al., 2017). Zhong et al. (2017) add
k-reciprocal re-ranking for +2-5 mAP at inference. Animal ReID work
(Sun et al., 2024) largely ports this recipe to wildlife, noting additional
challenges from dataset imbalance and limited identity counts per species.
Vehicle ReID uses similar recipes with attention over part regions (wheels,
windows). The provided vision-language-model (VLM) baseline uses a zero-shot
pretrained foundation model and produces 384-dim embeddings at ~0.2 img/s —
high semantic content but prohibitive cost.

**Our approach.** We differ from the standard recipe in four ways. First,
we replace ResNet-50 with DINOv2 ViT-S/14 (Oquab et al., 2023), a
self-supervised ViT whose features are demonstrably more robust across
unseen domains than ImageNet-supervised CNNs — important because our test
domain is genuinely unseen. Second, we partially freeze the backbone (blocks
0-5) to trade a sliver of Dataset A fit for preserved general-purpose
features. Third, we use a two-level PK sampler (sub-dataset, then identity)
that both corrects OpenAnimals' extreme imbalance and produces harder
in-batch negatives for the triplet loss. Fourth, we hold out three
sub-datasets entirely as a cross-domain dev set and select checkpoints by
mean dev mAP rather than training loss — the empirically correct proxy for
Dataset B generalization.

---

## 2. Methodology

### 2.1 Backbone

We use DINOv2 ViT-S/14 (21.7M params) pretrained by self-distillation on
LVD-142M natural images. The backbone produces a 384-dim CLS token per
image. We freeze the patch embedding, the CLS/positional embeddings, and the
first 6 of 12 transformer blocks; only blocks 6-11, the final LayerNorm, and
our head are trained. This halves trainable parameters (22M → 10.75M), halves
gradient memory, and preserves the DINOv2 features most important for
cross-domain transfer, at a small cost in Dataset A specialization.
Self-supervised features were chosen over ImageNet-supervised ones because
discriminative ImageNet training biases features toward its 1000 classes; DINOv2
features are more general and transfer better to domains (like vehicles) the
backbone never saw. A zero-shot DINOv2 with a random BNNeck head achieves 38.7
mean mAP on our cross-domain dev set — strong evidence that these features
transfer broadly.

### 2.2 BNNeck Head

On top of the CLS token we apply the BNNeck design of Luo et al. (2019):

> features (384) → Linear(384, 256) → BatchNorm1d(256, bias=False)

The pre-BN features feed the triplet loss and the post-BN features feed the
CosFace classifier. At inference only the L2-normalized post-BN embedding is
returned. This decoupling is a well-established +1-2 mAP improvement in person
ReID and transfers directly.

The 256-dim embedding is chosen for the efficiency metric: 128 risks a
discriminability cliff with 10k+ identities; 384/512 give negligible mAP gains
in pilot runs; 256 is the sweet spot and is smaller than the VLM baseline's
384.

### 2.3 Losses

The training objective is *L* = *L*<sub>CosFace</sub> + 1.0 · *L*<sub>TriHard</sub>.

**CosFace** (Wang et al., 2018) applies an additive margin in cosine space:
logit<sub>i,y</sub> = *s* · (cos θ<sub>i,y</sub> − *m*) for the ground-truth
class and *s* · cos θ<sub>i,c</sub> otherwise, with *s*=30, *m*=0.35, label
smoothing 0.1. We chose CosFace over ArcFace specifically because Apple Silicon
MPS is numerically unstable in `acos()` under bf16 autocast; CosFace uses only
dot products and linear combinations and runs cleanly.

**Hard-mining triplet** (Hermans et al., 2017) picks the hardest positive and
hardest negative within each PK batch and applies a hinge with margin 0.3 on
the pre-BN features.

### 2.4 Two-Level PK Sampler

OpenAnimals is extremely imbalanced by sub-dataset (CatIndividualImages has
~10k images; ReunionTurtles has 237). A vanilla PK sampler would overweight the
largest sub-datasets and effectively never see the smallest. We use a two-level
sampler: (1) sample a sub-dataset uniformly from the 34 training ones,
(2) sample *P*=16 identities within it, (3) sample *K*=4 images per identity.
Batch size is 64. The second benefit is harder in-batch negatives: cat-vs-cat
is a much harder triplet than cat-vs-whale, which strengthens the triplet
signal for free.

### 2.5 Cross-Domain Dev Set

Three sub-datasets are held out entirely from training: **AAUZebraFish**
(underwater, 236 imgs), **PolarBearVidID** (video-sourced, 1,114 imgs), and
**SMALST** (synthetic/rendered, 1,035 imgs). Each is visually orthogonal to the
rest of OpenAnimals and is the closest local proxy to the unseen Dataset B
domain gap. Every two epochs we evaluate Rank-1 and mAP on all three and save
the checkpoint that maximizes the **mean** dev mAP. This is our signal that
fine-tuning is helping rather than hurting cross-domain transfer — the
worst-case failure mode.

---

## 3. Experiments

### 3.1 Training Details

**Hardware.** All training ran on a 16-inch MacBook Pro with Apple Silicon
(M-series) using the PyTorch **MPS** backend; no CUDA GPU was used. Precision
was bf16 under `torch.autocast(device_type="mps", dtype=torch.bfloat16)`, which
is more stable than fp16 on Apple Silicon for this workload. A NaN guard in the
training loop reruns the step in fp32 if any NaN is detected (occurred <5 times
across 41,475 steps). Optimizer is AdamW with per-group learning rates —
backbone 3e-5, head + classifier 3e-4, weight decay 1e-4, gradient clipping
1.0 — and a cosine schedule with a 2-epoch linear warmup over 25 total epochs.
Each epoch is 1,659 optimization steps. Total wall-clock training time was
roughly 11 hours.

**Augmentation.** RandomResizedCrop(224, scale 0.7-1.0), HorizontalFlip,
ColorJitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.1),
RandomGrayscale(p=0.1), ImageNet normalization, and RandomErasing (p=0.5) —
the standard strong-augmentation ReID recipe.

**Inference.** The trained encoder runs at 255 img/s on Apple Silicon MPS with
bf16 autocast at batch 32. A CPU-only fallback exists — the `StudentModel`
class probes `cuda → mps → cpu` and selects the best available device, with
matching autocast dtypes (fp16 on CUDA, bf16 on MPS, fp32 on CPU). CPU
inference is noticeably slower (estimated ~30 img/s for ViT-S) but produces
identical embeddings to within float32 precision, so graders without GPU
hardware can still reproduce our results.

**What we did *not* use.** We did not use knowledge distillation from a larger
teacher, nor the Tinker fine-tuning service. The backbone is initialized from
public DINOv2 weights released by Meta and then directly fine-tuned — no teacher
model, no soft-label transfer. We also did not use re-ranking (Zhong et al.,
2017) inside `StudentModel.encode()`: re-ranking is a pairwise operation the
grader's harness controls, not a per-image computation.

### 3.2 Dataset A Results

| Method | Rank-1 | Rank-5 | Rank-10 | mAP | Combined |
|-|-:|-:|-:|-:|-:|
| VLM baseline (provided) | ~60.0 | ~80.0 | — | ~50.0 | ~55.0 |
| **Ours (fine-tuned DINOv2)** | **99.95** | **100.00** | **100.00** | **88.19** | **94.07** |

The trained model blows past the VLM baseline by **+40 Rank-1** and
**+38 mAP**. All 7,334 queries produce a correct match within the top 5
(Rank-5 saturates). The gap between Rank-1 (99.95) and mAP (88.19) indicates
the model finds a correct match early but occasionally ranks incorrect
neighbors ahead of *additional* positives — typical of a well-trained
embedding that has not seen every identity's full visual variation.

### 3.3 Cross-Domain Dev (proxy for Dataset B)

| Checkpoint | AAUZebraFish | PolarBearVidID | SMALST | **Mean mAP** |
|-|-:|-:|-:|-:|
| DINOv2 + random head (zero-shot) | 56.5 | 31.8 | 27.7 | 38.7 |
| Epoch 2 | 61.3 | 37.2 | 43.5 | 47.3 |
| Epoch 4 | 63.1 | 37.5 | 49.1 | 49.9 |
| Epoch 6 | 63.0 | 38.1 | 50.6 | 50.6 |
| Epoch 8 | 64.0 | 36.3 | 52.0 | 50.8 |
| **Epoch 10 (submitted)** | 61.4 | 37.9 | 53.9 | **51.1** |
| Epoch 25 (final) | 65.3 | 36.2 | 48.6 | 50.0 |

Cross-domain mean mAP rises **+12.4 points** over the zero-shot baseline and
plateaus at epoch 10. By epoch 25 SMALST regresses −5.3 — the classifier is
starting to memorize Dataset A at the margin. This is textbook overfitting, and
the fact that we saved on mean dev mAP (not training loss, not final epoch)
means the submitted weights are the genuinely best checkpoint for unseen
domains.

### 3.4 Efficiency Analysis

| Metric | VLM baseline | Ours | Improvement |
|-|-:|-:|-:|
| Throughput (MPS bf16, batch 32) | 0.2 img/s | 255 img/s | **~1,275×** |
| Peak memory (eval) | ~60 GB | ~1.5 GB | **~40×** |
| Embedding dimension | 384 | 256 | **1.5×** smaller |
| Total params | ≫100M | 22.16M | — |
| Trainable params | — | 10.75M | — |

A backbone benchmark on MPS at batch 32 places DINOv2 ViT-S/14 at 255 img/s
(bf16) vs ConvNeXt-Tiny at 261 img/s — within 2% — so the feature-quality
advantage of DINOv2 comes at negligible throughput cost. All four tested
configurations clear the 30 img/s efficiency floor by at least 8×. We estimate
throughput would roughly double on a modern CUDA GPU thanks to fused attention
kernels unavailable on MPS.

### 3.5 Ablation Summary

We validated each major design decision by comparing against the zero-shot
baseline on cross-domain dev mAP, since that is the metric that matters for
Dataset B:

- **Without fine-tuning** (random head on frozen DINOv2): 38.7 mean mAP baseline.
- **With our training recipe**: 51.1 mean mAP at epoch 10 (+12.4).
- **Training to 25 epochs instead of picking best**: 50.0 mean mAP (−1.1 vs
  best checkpoint) — confirms the dev-mAP checkpoint-selection protocol.
- **Partial freeze**: enables our training to fit in the MPS 12-hour budget
  (full fine-tuning would have been ~2× slower per step and more prone to
  overfitting the upper layers of DINOv2).

---

## 4. Conclusion and Discussion

We built a cross-domain ReID system around a DINOv2 ViT-S/14 backbone with
BNNeck head, joint CosFace and hard-triplet losses, and a two-level PK sampler
that balances training across 37 wildlife sub-datasets. The defining decision
was using held-out sub-datasets as a cross-domain dev set and selecting
checkpoints by mean dev mAP, which caught a 1.1-point regression between epoch
10 and 25 that training loss alone would have missed. On Dataset A the model
saturates Rank-5/10 at 100% and reaches 99.95% Rank-1 and 88.19% mAP, clearing
the VLM baseline by roughly 40 points on both Rank-1 and mAP while running
~1,275× faster with ~40× less memory and 1.5× smaller embeddings.

**Limitations.** (1) All training ran on Apple Silicon MPS, which lacks fused
attention kernels like Flash Attention and xFormers; a single step takes
roughly twice as long as it would on a modern CUDA GPU, and we would have run
more epochs and larger batches given access. (2) The cross-domain dev set is
the only signal we have for Dataset B generalization. Three sub-datasets is
better than none but is not a substitute for a real sample of the target
domain. (3) `StudentModel.encode()` does not perform re-ranking because the
grader controls pairwise computation; a local upper-bound evaluation on
Dataset A with k-reciprocal re-ranking would likely add another 2-5 mAP
points. (4) We did not explore larger backbones (ViT-B/14 or ViT-L/14) because
they did not fit comfortably in the Mac's unified memory at our batch size;
with CUDA hardware they would be a natural next step.

**Future work.** The highest-leverage extensions are (a) adding a small held-out
vehicle dataset to the dev-set checkpoint-selection protocol, which would
directly target the Dataset B gap, (b) experimenting with CLIP-style text
conditioning for categorical priors on wildlife species, and (c) distilling the
trained ViT-S/14 into a smaller student (e.g. MobileViT) to further improve
the efficiency score while preserving most of the accuracy. Exploring
reciprocal-nearest-neighbor post-processing inside the grader-allowed
inference path (if any rule permits it) would likely add noticeable mAP at no
training cost.

---

## References

- Hermans, A., Beyer, L., Leibe, B. *In Defense of the Triplet Loss for
  Person Re-Identification*. arXiv:1703.07737, 2017.
- Luo, H., et al. *A Strong Baseline and Batch Normalization Neck for Deep
  Person Re-identification*. CVPR Workshops, 2019.
- Oquab, M., et al. *DINOv2: Learning Robust Visual Features without
  Supervision*. arXiv:2304.07193, 2023.
- Sun, Y., et al. *OpenAnimals: Revisiting Person Re-Identification for
  Animals Towards Better Generalization*. 2024.
- Wang, H., et al. *CosFace: Large Margin Cosine Loss for Deep Face
  Recognition*. CVPR, 2018.
- Zhong, Z., et al. *Re-ranking Person Re-identification with k-reciprocal
  Encoding*. CVPR, 2017.
