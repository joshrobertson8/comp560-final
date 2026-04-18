"""
Two-level PK sampler.

Each batch:
  1. Sample a sub-dataset uniformly from the K sub-datasets available.
  2. Within that sub-dataset, sample P identities uniformly (with replacement
     if fewer than P identities exist).
  3. For each identity, sample K images uniformly (with replacement if the
     identity has fewer than K instances).

This addresses the extreme sub-dataset imbalance in OpenAnimals (e.g.
CatIndividualImages 10K imgs vs ReunionTurtles 237 imgs). Without it, purely
ID-level PK sampling overweights the largest sub-datasets.

Batch size = P * K. Yields indices into the underlying dataset.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class TwoLevelPKSampler(Sampler[int]):
    """Sample (sub_dataset, identity, image) triples yielding P*K indices per batch.

    Args:
        sub_dataset_labels: list[str] of sub-dataset name per dataset item (len = N).
        identity_labels:    list[int] of identity label per dataset item (len = N).
        p:                  number of identities per batch.
        k:                  number of images per identity per batch.
        num_batches:        number of batches per epoch. Defaults to
                            len(dataset) // (p*k), yielding ~1 epoch of images.
        seed:               random seed.
    """

    def __init__(
        self,
        sub_dataset_labels: list[str],
        identity_labels: list[int],
        p: int = 16,
        k: int = 4,
        num_batches: int | None = None,
        seed: int = 0,
    ):
        assert len(sub_dataset_labels) == len(identity_labels)
        self.p = p
        self.k = k
        self.seed = seed
        self.epoch = 0

        # Build index: sub_dataset -> identity -> list of indices
        self.sub_to_ids: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, (sub, pid) in enumerate(zip(sub_dataset_labels, identity_labels)):
            self.sub_to_ids[sub][pid].append(idx)

        # List of sub-datasets for uniform sampling
        self.sub_datasets = sorted(self.sub_to_ids.keys())
        # Filter out sub-datasets with zero identities (shouldn't happen but safe)
        self.sub_datasets = [s for s in self.sub_datasets if len(self.sub_to_ids[s]) > 0]

        if num_batches is None:
            num_batches = max(1, len(identity_labels) // (p * k))
        self.num_batches = num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches * self.p * self.k

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_batches):
            # Step 1: sample sub-dataset uniformly
            sub = rng.choice(self.sub_datasets)
            pid_to_idxs = self.sub_to_ids[sub]
            pids = list(pid_to_idxs.keys())

            # Step 2: sample P ids (with replacement if needed)
            if len(pids) >= self.p:
                chosen_pids = rng.sample(pids, self.p)
            else:
                chosen_pids = [rng.choice(pids) for _ in range(self.p)]

            # Step 3: for each id, sample K images
            for pid in chosen_pids:
                idxs = pid_to_idxs[pid]
                if len(idxs) >= self.k:
                    chosen = rng.sample(idxs, self.k)
                else:
                    # Sample with replacement
                    chosen = [rng.choice(idxs) for _ in range(self.k)]
                for idx in chosen:
                    yield idx
