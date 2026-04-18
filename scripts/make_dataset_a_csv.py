"""
Generate a dataset_a prediction CSV using the submitted StudentModel.

Output format (compatible with evaluate.py):
    query_index,ranked_gallery_indices
    0,"45,12,78,3,..."
    ...

Usage:
    python scripts/make_dataset_a_csv.py \\
        --dataset_root datasets/dataset_a \\
        --output predictions/dataset_a.csv \\
        --device mps --batch_size 32

Adds optional k-reciprocal re-ranking (off by default; toggle with --rerank)
for local upper-bound validation. Re-ranking is NOT part of the submitted
model (graders control ranking), so leave it off if you want to validate the
number the grader will actually see.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from model import StudentModel  # noqa: E402


class ImageOnlyDataset(Dataset):
    def __init__(self, root: str, image_paths: list[str], image_size: int = 224):
        self.root = root
        self.image_paths = image_paths
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tf(img), idx


def load_dataset_a_paths(root: str) -> tuple[list[str], list[str]]:
    df = pd.read_parquet(os.path.join(root, "test.parquet"))
    query_paths, gallery_paths = [], []
    for pid, group in df.groupby("identity"):
        paths = group["image_path"].values.tolist()
        if len(paths) >= 2:
            query_paths.extend(paths[:2])
            gallery_paths.extend(paths[2:])
        else:
            gallery_paths.extend(paths)
    gallery_paths.extend(query_paths)  # standard ReID protocol
    return query_paths, gallery_paths


def encode_all(model: StudentModel, ds: Dataset, batch_size: int, num_workers: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    out = np.zeros((len(ds), model.embedding_dim), dtype=np.float32)
    t0 = time.time()
    for images, indices in tqdm(loader, desc="Encoding"):
        emb = model.encode(images).cpu().numpy()
        out[indices.numpy()] = emb
    print(f"  encoded {len(ds)} images in {time.time()-t0:.1f}s "
          f"({len(ds)/(time.time()-t0):.1f} img/s)")
    return out


def tta_encode_all(model: StudentModel, ds: Dataset, batch_size: int, num_workers: int) -> np.ndarray:
    """Horizontal-flip TTA for CSV generation."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    out = np.zeros((len(ds), model.embedding_dim), dtype=np.float32)
    for images, indices in tqdm(loader, desc="Encoding (TTA)"):
        e1 = model.encode(images).cpu().numpy()
        e2 = model.encode(torch.flip(images, dims=[3])).cpu().numpy()
        emb = e1 + e2
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        out[indices.numpy()] = emb
    return out


def k_reciprocal_rerank(
    q: np.ndarray, g: np.ndarray, k1: int = 20, k2: int = 6, lambda_value: float = 0.3
) -> np.ndarray:
    """k-reciprocal re-ranking (Zhong et al. 2017).

    Returns a new (Q, G) distance matrix. Higher cost is worse (lower
    similarity).
    """
    q_num = q.shape[0]
    g_num = g.shape[0]
    all_emb = np.vstack([q, g]).astype(np.float32)
    all_num = all_emb.shape[0]

    sim = all_emb @ all_emb.T
    original_dist = 2.0 - 2.0 * sim  # cosine distance in [0, 2]
    del sim

    initial_rank = np.argsort(original_dist).astype(np.int32)
    V = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(all_num):
        forward_k = initial_rank[i, : k1 + 1]
        back_k = initial_rank[forward_k, : k1 + 1]
        fi = np.where(back_k == i)[0]
        k_reciprocal = forward_k[fi]

        # Expand
        k_expand = k_reciprocal
        for candidate in k_reciprocal:
            c_forward = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            c_back = initial_rank[c_forward, : int(np.around(k1 / 2)) + 1]
            cfi = np.where(c_back == candidate)[0]
            c_reciprocal = c_forward[cfi]
            if len(np.intersect1d(c_reciprocal, k_reciprocal)) > 2. / 3. * len(c_reciprocal):
                k_expand = np.append(k_expand, c_reciprocal)

        k_expand = np.unique(k_expand)
        weight = np.exp(-original_dist[i, k_expand])
        V[i, k_expand] = weight / weight.sum()

    # Local query expansion
    if k2 > 0:
        V_qe = np.zeros_like(V)
        for i in range(all_num):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe
        del V_qe

    # Jaccard distance
    jaccard_dist = np.zeros_like(original_dist)
    indNonZero = [np.where(V[i] != 0)[0] for i in range(all_num)]
    indImages = [np.where(V[:, j] != 0)[0] for j in range(all_num)]

    for i in range(all_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        indNonZero_i = indNonZero[i]
        for j in indNonZero_i:
            indImages_j = indImages[j]
            temp_min[indImages_j] += np.minimum(V[i, j], V[indImages_j, j])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:q_num, q_num:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="datasets/dataset_a")
    parser.add_argument("--output", default="predictions/dataset_a.csv")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--tta", action="store_true", help="Horizontal-flip TTA")
    parser.add_argument("--rerank", action="store_true", help="k-reciprocal re-ranking (upper bound)")
    args = parser.parse_args()

    query_paths, gallery_paths = load_dataset_a_paths(args.dataset_root)
    print(f"Queries: {len(query_paths)}  Gallery: {len(gallery_paths)}")

    model = StudentModel(device=args.device)
    print(f"StudentModel device: {model.device}  embedding_dim: {model.embedding_dim}")

    q_ds = ImageOnlyDataset(args.dataset_root, query_paths)
    g_ds = ImageOnlyDataset(args.dataset_root, gallery_paths)

    encode_fn = tta_encode_all if args.tta else encode_all
    q_emb = encode_fn(model, q_ds, args.batch_size, args.num_workers)
    g_emb = encode_fn(model, g_ds, args.batch_size, args.num_workers)

    print("Ranking...")
    if args.rerank:
        dist = k_reciprocal_rerank(q_emb, g_emb)
        rankings = np.argsort(dist, axis=1)[:, : args.top_k]
    else:
        # Cosine sim; embeddings are L2-normalized
        sim = q_emb @ g_emb.T
        rankings = np.argsort(-sim, axis=1)[:, : args.top_k]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for qi in range(len(query_paths)):
        ranked = ",".join(str(int(x)) for x in rankings[qi])
        rows.append({"query_index": qi, "ranked_gallery_indices": ranked})
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
