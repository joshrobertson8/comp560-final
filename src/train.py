"""
COMP560 ReID training loop.

- Backbone: DINOv2 ViT-S/14 (partial freeze: blocks 0-5 frozen by default).
- Head: BNNeck (Linear + BN).
- Losses: CosFace + TripletHard.
- Sampler: two-level PK (sub-dataset balanced).
- Precision: bf16 autocast on MPS, fp16 on CUDA, fp32 on CPU.
- Validation: cross-domain holdout sub-datasets every few epochs.

Run:
    python -m src.train --data_root datasets/dataset_a --epochs 25

Or via script: scripts/train.sh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as `python -m src.train` or `python src/train.py`
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from src.augment import build_train_transform, build_eval_transform
from src.backbone import build_backbone
from src.dataset import ReIDDataset, split_train_dev, DEFAULT_HOLDOUT_SUBS
from src.head import ReIDModel
from src.losses import CosFaceLoss, TripletHardLoss
from src.sampler import TwoLevelPKSampler


# ============================================================================
# Utils
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device in ("cuda", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool = True):
    """Return a torch.autocast context manager for the given device."""
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ============================================================================
# Evaluation on a held-out sub-dataset (cross-domain dev)
# ============================================================================

@torch.inference_mode()
def evaluate_dev(
    model: ReIDModel,
    dataset: ReIDDataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
    max_rank: int = 10,
) -> dict:
    """Build a query/gallery split inside the dataset (2 per id for query, rest
    for gallery + query re-added) and compute Rank-1, Rank-5, mAP."""
    if len(dataset) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "num_query": 0, "num_gallery": 0}

    model.eval()

    # Encode all images
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    embeddings = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast_ctx(device):
            emb = model.encode(images)
        embeddings.append(emb.float().cpu())
    embeddings = torch.cat(embeddings, dim=0).numpy()

    pids = np.array(dataset.identity_labels)
    # Build query / gallery (same protocol as evaluate.py for dataset_a)
    from collections import defaultdict
    idx_by_pid = defaultdict(list)
    for i, p in enumerate(pids):
        idx_by_pid[p].append(i)
    query_idx, gallery_idx = [], []
    for p, idxs in idx_by_pid.items():
        if len(idxs) >= 2:
            query_idx.extend(idxs[:2])
            gallery_idx.extend(idxs[2:])
        else:
            gallery_idx.extend(idxs)
    gallery_idx.extend(query_idx)

    q = embeddings[query_idx]
    g = embeddings[gallery_idx]
    q_pids = pids[query_idx]
    g_pids = pids[gallery_idx]

    if len(q) == 0 or len(g) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "num_query": len(q), "num_gallery": len(g)}

    # Cosine similarity (embeddings are already L2-normalized)
    sim = q @ g.T  # (Q, G)
    order = np.argsort(-sim, axis=1)

    # CMC + mAP
    cmc = np.zeros(max_rank, dtype=np.float64)
    aps = []
    valid = 0
    for qi in range(len(q)):
        g_pid_ordered = g_pids[order[qi]]
        matches = (g_pid_ordered == q_pids[qi]).astype(np.int32)
        if matches.sum() == 0:
            continue
        valid += 1
        c = matches.cumsum()
        c = (c > 0).astype(np.int32)
        cmc[: min(max_rank, len(c))] += c[: min(max_rank, len(c))]

        tmp = matches.cumsum()
        precision_at_k = [x / (i + 1.0) for i, x in enumerate(tmp)]
        ap = (np.asarray(precision_at_k) * matches).sum() / matches.sum()
        aps.append(ap)

    if valid == 0:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "num_query": len(q), "num_gallery": len(g)}
    cmc /= valid
    return {
        "rank1": float(cmc[0]) * 100,
        "rank5": float(cmc[min(4, max_rank - 1)]) * 100,
        "mAP": float(np.mean(aps)) * 100,
        "num_query": len(q),
        "num_gallery": len(g),
    }


# ============================================================================
# Main training loop
# ============================================================================

def train_one_epoch(
    model: ReIDModel,
    cosface: CosFaceLoss,
    triplet: TripletHardLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    triplet_weight: float,
    log_every: int = 50,
) -> dict:
    model.train()
    cosface.train()

    epoch_loss_cos = 0.0
    epoch_loss_tri = 0.0
    n_steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, enabled=use_amp):
            pre_bn, post_bn = model(images)
            loss_cos = cosface(post_bn, labels)
            loss_tri = triplet(pre_bn, labels)
            loss = loss_cos + triplet_weight * loss_tri

        # Guard against NaNs coming from MPS mixed precision
        if not torch.isfinite(loss):
            if use_amp:
                # Retry in fp32 for this step
                with torch.autocast(device_type=device.type, enabled=False):
                    pre_bn, post_bn = model(images)
                    loss_cos = cosface(post_bn, labels)
                    loss_tri = triplet(pre_bn, labels)
                    loss = loss_cos + triplet_weight * loss_tri
            if not torch.isfinite(loss):
                # Skip this step rather than poison grads
                continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(cosface.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss_cos += float(loss_cos.detach())
        epoch_loss_tri += float(loss_tri.detach())
        n_steps += 1

        if step % log_every == 0:
            pbar.set_postfix(
                cos=f"{loss_cos.item():.3f}",
                tri=f"{loss_tri.item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else "n/a",
            )

    return {
        "loss_cos": epoch_loss_cos / max(n_steps, 1),
        "loss_tri": epoch_loss_tri / max(n_steps, 1),
        "n_steps": n_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/dataset_a")
    parser.add_argument("--parquet_file", type=str, default="train.parquet")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "convnext_tiny"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--freeze_first_n", type=int, default=6,
                        help="Freeze first N blocks of backbone (ViT blocks or ConvNeXt stages).")
    parser.add_argument("--p", type=int, default=16, help="identities per batch")
    parser.add_argument("--k", type=int, default=4, help="images per identity")
    parser.add_argument("--num_batches_per_epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lr_backbone", type=float, default=3e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--triplet_weight", type=float, default=1.0)
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--cosface_s", type=float, default=30.0)
    parser.add_argument("--cosface_m", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=2, help="Run dev eval every N epochs.")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device} (AMP: {args.amp})")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_tf = build_train_transform(args.image_size)
    eval_tf = build_eval_transform(args.image_size)

    print("Loading datasets...")
    train_ds, dev_datasets = split_train_dev(
        root=args.data_root,
        parquet_file=args.parquet_file,
        holdout_subs=DEFAULT_HOLDOUT_SUBS,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )
    print(f"Training set: {len(train_ds)} images, {train_ds.num_classes} identities, "
          f"{len(set(train_ds.sub_datasets))} sub-datasets")
    for sub, ds in dev_datasets.items():
        print(f"  Dev ({sub}): {len(ds)} images, {ds.num_classes} identities")

    sampler = TwoLevelPKSampler(
        sub_dataset_labels=train_ds.sub_datasets,
        identity_labels=train_ds.identity_labels,
        p=args.p,
        k=args.k,
        num_batches=args.num_batches_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.p * args.k,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"Batches per epoch: {sampler.num_batches} (batch size {args.p * args.k})")

    # ---- Model ----
    print(f"Building backbone: {args.backbone}")
    backbone = build_backbone(args.backbone)
    backbone.freeze_first_n_blocks(args.freeze_first_n)
    model = ReIDModel(backbone, embed_dim=args.embed_dim).to(device)
    cosface = CosFaceLoss(
        embedding_dim=args.embed_dim,
        num_classes=train_ds.num_classes,
        s=args.cosface_s,
        m=args.cosface_m,
        label_smoothing=args.label_smoothing,
    ).to(device)
    triplet = TripletHardLoss(margin=args.triplet_margin)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M total")

    # ---- Optimizer & Schedule ----
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    loss_params = list(cosface.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
            {"params": loss_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = sampler.num_batches * args.epochs
    warmup_steps = sampler.num_batches * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume ----
    start_epoch = 0
    best_dev_map = -1.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        cosface.load_state_dict(ckpt["cosface_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_dev_map = ckpt.get("best_dev_map", -1.0)

    # ---- Training ----
    history: list[dict] = []
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        stats = train_one_epoch(
            model, cosface, triplet, loader, optimizer, scheduler,
            device, args.amp, epoch + 1, args.triplet_weight,
        )
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: cos={stats['loss_cos']:.3f}  tri={stats['loss_tri']:.3f}  "
              f"time={epoch_time/60:.1f}min")

        record = {"epoch": epoch + 1, "epoch_time_s": epoch_time, **stats, "dev": {}}

        # Dev eval
        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            dev_metrics = {}
            for sub, ds in dev_datasets.items():
                m = evaluate_dev(model, ds, device)
                dev_metrics[sub] = m
                print(f"  Dev[{sub}]: Rank-1={m['rank1']:.1f}  Rank-5={m['rank5']:.1f}  "
                      f"mAP={m['mAP']:.1f}  (Q={m['num_query']}, G={m['num_gallery']})")
            mean_map = float(np.mean([m["mAP"] for m in dev_metrics.values()])) if dev_metrics else 0.0
            mean_r1 = float(np.mean([m["rank1"] for m in dev_metrics.values()])) if dev_metrics else 0.0
            print(f"  Dev mean mAP={mean_map:.2f}  Rank-1={mean_r1:.2f}")
            record["dev"] = dev_metrics
            record["dev_mean_map"] = mean_map
            record["dev_mean_rank1"] = mean_r1

            if mean_map > best_dev_map:
                best_dev_map = mean_map
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "cosface_state_dict": cosface.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_dev_map": best_dev_map,
                        "args": vars(args),
                        "id_to_label": train_ds.id_to_label,
                    },
                    save_dir / "best_model.pth",
                )
                print(f"  -> saved best_model.pth (dev mAP={best_dev_map:.2f})")

        history.append(record)
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "cosface_state_dict": cosface.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dev_map": best_dev_map,
                    "args": vars(args),
                    "id_to_label": train_ds.id_to_label,
                },
                save_dir / f"checkpoint_epoch{epoch+1}.pth",
            )

    # Always save a final "last" checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "cosface_state_dict": cosface.state_dict(),
            "best_dev_map": best_dev_map,
            "args": vars(args),
        },
        save_dir / "last_model.pth",
    )
    print(f"\nDone. Best dev mAP: {best_dev_map:.2f}")


if __name__ == "__main__":
    main()
