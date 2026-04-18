"""
Day-1 throughput benchmark on the configured device (default: MPS).

Measures forward-pass throughput (images/sec) and peak memory for the two
candidate backbones (DINOv2 ViT-S/14 and ConvNeXt-Tiny), in fp32 and bf16.

Usage: python scripts/benchmark_backbone.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.backbone import build_backbone
from src.head import ReIDModel


def bench(
    backbone_name: str,
    device: torch.device,
    dtype: torch.dtype,
    embed_dim: int = 256,
    batch_size: int = 32,
    iterations: int = 30,
    warmup: int = 5,
) -> dict:
    torch.manual_seed(0)
    bb = build_backbone(backbone_name)
    model = ReIDModel(bb, embed_dim=embed_dim).to(device).eval()

    x = torch.randn(batch_size, 3, 224, 224, device=device)

    use_autocast = dtype in (torch.float16, torch.bfloat16)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _ = model.encode(x)
            else:
                _ = model.encode(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.inference_mode():
        for _ in range(iterations):
            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _ = model.encode(x)
            else:
                _ = model.encode(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
    elapsed = time.time() - t0

    imgs = iterations * batch_size
    throughput = imgs / elapsed

    peak_mem_mb = -1.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    del model, bb, x
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "backbone": backbone_name,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "throughput_img_s": throughput,
        "peak_mem_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtypes = [torch.float32]
    if device.type == "mps":
        dtypes.append(torch.bfloat16)
    if device.type == "cuda":
        dtypes.append(torch.float16)

    results = []
    for bb_name in ["dinov2_vits14", "convnext_tiny"]:
        for dtype in dtypes:
            try:
                r = bench(bb_name, device, dtype, batch_size=args.batch_size,
                          iterations=args.iterations)
                print(f"{r['backbone']:20s}  {r['dtype']:22s}  "
                      f"{r['throughput_img_s']:6.1f} img/s  "
                      f"peak={r['peak_mem_mb']:.1f} MB")
                results.append(r)
            except Exception as e:
                print(f"{bb_name} / {dtype}: FAILED — {e}")

    print("\nSummary:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
