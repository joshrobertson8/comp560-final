"""
BNNeck head (Luo et al., 2019).

At training:
  features --Linear--> pre_bn (used for triplet loss) --BN--> post_bn (used for classifier)
At inference:
  we return L2-normalized post-BN features.

The head also exposes a separate classifier weight (CosFace) used only at
training — stored on the head so the training loop can optimize it jointly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeckHead(nn.Module):
    """Linear + BatchNorm head returning two views of the embedding.

    Args:
        in_dim: input feature dim from the backbone.
        embed_dim: output embedding dim (e.g. 256).
    """

    def __init__(self, in_dim: int, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck = nn.Linear(in_dim, embed_dim)
        nn.init.kaiming_normal_(self.bottleneck.weight, mode="fan_out")
        nn.init.constant_(self.bottleneck.bias, 0.0)

        # BN with no affine bias (as in the original BNNeck paper)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pre_bn, post_bn). Both are un-normalized."""
        pre_bn = self.bottleneck(features)
        post_bn = self.bn(pre_bn)
        return pre_bn, post_bn

    @torch.inference_mode()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Inference-only path returning L2-normalized post-BN embeddings."""
        _, post_bn = self.forward(features)
        return F.normalize(post_bn, p=2, dim=1)


class ReIDModel(nn.Module):
    """Backbone + BNNeck head. Convenience wrapper used by train and inference."""

    def __init__(self, backbone: nn.Module, embed_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = BNNeckHead(in_dim=backbone.feature_dim, embed_dim=embed_dim)

    @property
    def embedding_dim(self) -> int:
        return self.head.embed_dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        return self.head(features)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        _, post_bn = self.head(features)
        return F.normalize(post_bn, p=2, dim=1)
