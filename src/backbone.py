"""
Backbone loader.

Primary: DINOv2 ViT-S/14 (21M params, 384-dim CLS features).
Loaded via torch.hub with offline fallback: first looks for cached weights in
~/.cache/torch/hub/ and an optional local path under weights/dinov2_vits14.pth.

Fallback: ConvNeXt-Tiny via timm (swap if MPS throughput on DINOv2 is bad).

The backbone forward returns pooled features of shape (B, feat_dim).
Partial-freeze is handled here (freeze patch_embed + first N blocks).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# DINOv2 ViT-S/14
# -----------------------------------------------------------------------------

class DinoV2Backbone(nn.Module):
    """DINOv2 ViT-S/14 wrapper returning CLS token features.

    DINOv2 forward returns a (B, 384) CLS feature. We expose the number of
    transformer blocks to let the training loop freeze the first N.
    """

    FEATURE_DIM = 384
    NUM_BLOCKS = 12

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        # Load via torch.hub (downloads on first use). If `weights_path` is
        # provided and exists, we load the state_dict locally instead — this
        # makes the submission fully offline-reproducible.
        if weights_path and os.path.exists(weights_path):
            # Construct the same architecture without downloading weights
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=False,
                trust_repo=True,
            )
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                trust_repo=True,
            )

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    def freeze_first_n_blocks(self, n: int) -> None:
        """Freeze patch_embed + blocks[0:n]. Rest remains trainable."""
        for p in self.model.patch_embed.parameters():
            p.requires_grad = False
        # DINOv2 also has cls_token and pos_embed parameters at the model level
        if hasattr(self.model, "cls_token"):
            self.model.cls_token.requires_grad = False
        if hasattr(self.model, "pos_embed"):
            self.model.pos_embed.requires_grad = False
        if hasattr(self.model, "mask_token"):
            self.model.mask_token.requires_grad = False
        for i, block in enumerate(self.model.blocks):
            if i < n:
                for p in block.parameters():
                    p.requires_grad = False

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # DINOv2 models return CLS token when called directly
        feats = self.model(images)  # (B, 384)
        return feats


# -----------------------------------------------------------------------------
# Fallback: ConvNeXt-Tiny via timm
# -----------------------------------------------------------------------------

class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-Tiny backbone returning (B, 768) pooled features."""

    FEATURE_DIM = 768
    NUM_STAGES = 4  # 4 downsampling stages

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        import timm
        # num_classes=0 removes the classifier; global_pool='avg' gives pooled features
        self.model = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=(weights_path is None or not os.path.exists(weights_path)),
            num_classes=0,
            global_pool="avg",
        )
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=False)

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    def freeze_first_n_blocks(self, n: int) -> None:
        """Freeze stem + first n of 4 stages."""
        for p in self.model.stem.parameters():
            p.requires_grad = False
        for i, stage in enumerate(self.model.stages):
            if i < n:
                for p in stage.parameters():
                    p.requires_grad = False

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)  # (B, 768)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

BACKBONES = {
    "dinov2_vits14": DinoV2Backbone,
    "convnext_tiny": ConvNeXtTinyBackbone,
}


def build_backbone(name: str, weights_path: str | None = None) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONES)}")
    return BACKBONES[name](weights_path=weights_path)
