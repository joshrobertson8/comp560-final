"""
Loss functions: CosFace (large-margin cosine) + batch-hard triplet.

CosFace is preferred over ArcFace on MPS because it avoids `acos()` which can
be numerically unstable on Apple Silicon. See Wang et al. CosFace (2018).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    """CosFace: Large Margin Cosine Loss for Deep Face Recognition (2018).

    logit = s * (cos(theta) - m * one_hot(y))

    Args:
        embedding_dim: dimensionality of the post-BN embedding fed in.
        num_classes: number of training identities.
        s: cosine scale factor (temperature^-1). Typical 30.0.
        m: additive cosine margin. Typical 0.35 for ReID.
        label_smoothing: optional label smoothing factor (default 0.1).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.35,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D) — post-BN features (not pre-normalized).
        # CosFace compares L2-normalized weight vs L2-normalized embedding.
        cos = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        # Clamp to valid cosine range. Keeps grads sane on MPS.
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        one_hot = torch.zeros_like(cos).scatter_(1, labels.unsqueeze(1), 1.0)
        logits = self.s * (cos - self.m * one_hot)
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


class TripletHardLoss(nn.Module):
    """Batch-hard triplet loss (Hermans et al., 2017).

    Expects features that have *not* been L2-normalized — internally we compute
    cosine distances after normalizing. PK-sampled batches (P ids, K images
    each) are assumed.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D), labels: (B,)
        emb = F.normalize(embeddings, p=2, dim=1)
        # cosine distance = 1 - cos sim. Shape (B, B).
        dist = 1.0 - emb @ emb.t()

        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)

        # Positive mask: same id, different instance
        pos_mask = same & ~eye
        # Negative mask: different id
        neg_mask = ~same

        # Hardest positive = max distance among positives (replace non-positives with -inf)
        pos_dist = dist.masked_fill(~pos_mask, float("-inf"))
        hardest_pos, _ = pos_dist.max(dim=1)

        # Hardest negative = min distance among negatives
        neg_dist = dist.masked_fill(~neg_mask, float("inf"))
        hardest_neg, _ = neg_dist.min(dim=1)

        # Some rows may have no positive (degenerate). Mask them out.
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return dist.new_zeros(())

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return loss.mean()
