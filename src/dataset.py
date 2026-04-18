"""
ReIDTrainDataset with cross-domain holdout.

Loads from a parquet file with columns:
  image_path, identity, camera_id[, split]

Three sub-datasets are held out from training to serve as the cross-domain dev
set (best proxy for the unseen Dataset B). Default holdouts:
  - AAUZebraFish (unique fish domain, 236 imgs)
  - PolarBearVidID (video-source, 1114 imgs)
  - SMALST (synthetic/rendered, 1035 imgs)

These are evaluated separately during training to track generalization.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_HOLDOUT_SUBS = ("AAUZebraFish", "PolarBearVidID", "SMALST")


def _infer_sub_dataset(path: str) -> str:
    """Extract sub-dataset name: 'images/<sub>/...' -> '<sub>'."""
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "images":
        return parts[1]
    return "unknown"


class ReIDDataset(Dataset):
    """Generic ReID dataset reading from a parquet file.

    Args:
        root: dataset root (parent of 'images/').
        parquet_file: name of the parquet file under `root`.
        transform: torchvision transform.
        split_filter: if set, only keep rows whose 'split' column equals this.
        include_subs: if set, only keep rows in these sub-datasets.
        exclude_subs: if set, drop rows in these sub-datasets.
        min_images_per_id: if > 1, drop identities with fewer than this many images.
    """

    def __init__(
        self,
        root: str,
        parquet_file: str = "train.parquet",
        transform=None,
        split_filter: str | None = None,
        include_subs: list[str] | tuple[str, ...] | None = None,
        exclude_subs: list[str] | tuple[str, ...] | None = None,
        min_images_per_id: int = 1,
    ):
        self.root = root
        df = pd.read_parquet(os.path.join(root, parquet_file))

        if split_filter is not None and "split" in df.columns:
            df = df[df["split"] == split_filter].reset_index(drop=True)

        df["sub_dataset"] = df["image_path"].map(_infer_sub_dataset)

        if include_subs is not None:
            df = df[df["sub_dataset"].isin(list(include_subs))].reset_index(drop=True)
        if exclude_subs is not None:
            df = df[~df["sub_dataset"].isin(list(exclude_subs))].reset_index(drop=True)

        if min_images_per_id > 1:
            counts = df.groupby("identity").size()
            keep_ids = counts[counts >= min_images_per_id].index
            df = df[df["identity"].isin(keep_ids)].reset_index(drop=True)

        # Build ID -> contiguous label mapping
        unique_ids = sorted(df["identity"].unique().tolist())
        self.id_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)

        self.image_paths: list[str] = df["image_path"].tolist()
        self.identity_labels: list[int] = [self.id_to_label[p] for p in df["identity"].tolist()]
        self.raw_identities: list[int] = df["identity"].tolist()
        self.sub_datasets: list[str] = df["sub_dataset"].tolist()
        self.camera_ids: list[int] = df["camera_id"].tolist() if "camera_id" in df.columns else [0] * len(df)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.image_paths[idx])
        # Open with a generous catch: corrupted/missing images shouldn't crash an epoch.
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fall back to a black square. The sampler will re-try on next epoch.
            img = Image.new("RGB", (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.identity_labels[idx]


def split_train_dev(
    root: str,
    parquet_file: str = "train.parquet",
    holdout_subs: tuple[str, ...] = DEFAULT_HOLDOUT_SUBS,
    train_transform=None,
    eval_transform=None,
    min_images_per_id_train: int = 2,
) -> tuple[ReIDDataset, dict[str, ReIDDataset]]:
    """Build the training dataset (excluding holdout_subs) and one eval dataset
    per holdout sub-dataset (for cross-domain monitoring).

    Returns:
        (train_dataset, dev_datasets)
        where dev_datasets is a dict: sub_name -> ReIDDataset (eval transform).
    """
    train_ds = ReIDDataset(
        root=root,
        parquet_file=parquet_file,
        transform=train_transform,
        split_filter="train",
        exclude_subs=holdout_subs,
        min_images_per_id=min_images_per_id_train,
    )

    dev_datasets: dict[str, ReIDDataset] = {}
    for sub in holdout_subs:
        ds = ReIDDataset(
            root=root,
            parquet_file=parquet_file,
            transform=eval_transform,
            split_filter="train",
            include_subs=[sub],
            min_images_per_id=2,  # need at least 2 for query/gallery
        )
        if len(ds) > 0:
            dev_datasets[sub] = ds

    return train_ds, dev_datasets
