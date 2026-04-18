"""
Generate presentation-quality charts from training history + final metrics.

Outputs PNG files to charts/ for use in a slide deck.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ---- Style ----
plt.rcParams.update({
    "figure.figsize": (10, 5.6),
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "font.family": "sans-serif",
})

COLORS = {
    "primary": "#2E86AB",
    "accent": "#E63946",
    "green": "#06A77D",
    "orange": "#F18F01",
    "gray": "#6C757D",
    "light": "#E8EDF2",
}

history = json.loads((ROOT / "checkpoints" / "history.json").read_text())
epochs = [h["epoch"] for h in history]


# ============================================================================
# 1) Training loss curves
# ============================================================================
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
cos = [h["loss_cos"] for h in history]
tri = [h["loss_tri"] for h in history]

l1 = ax1.plot(epochs, cos, color=COLORS["primary"], linewidth=2.5,
              marker="o", markersize=5, label="CosFace loss")
l2 = ax2.plot(epochs, tri, color=COLORS["accent"], linewidth=2.5,
              marker="s", markersize=5, label="Triplet loss")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("CosFace loss", color=COLORS["primary"])
ax2.set_ylabel("Triplet loss", color=COLORS["accent"])
ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
ax2.spines["top"].set_visible(False)

ax1.set_title("Training losses — monotonic decrease over 25 epochs")
ax1.grid(True, alpha=0.25, axis="y")

lines = l1 + l2
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "01_training_losses.png")
plt.close()
print(f"Wrote {CHARTS_DIR / '01_training_losses.png'}")


# ============================================================================
# 2) Cross-domain dev mAP over epochs (3 held-out sub-datasets + mean)
# ============================================================================
eval_epochs, zf, pb, sm, mean_map = [], [], [], [], []
for h in history:
    if h.get("dev"):
        eval_epochs.append(h["epoch"])
        zf.append(h["dev"]["AAUZebraFish"]["mAP"])
        pb.append(h["dev"]["PolarBearVidID"]["mAP"])
        sm.append(h["dev"]["SMALST"]["mAP"])
        mean_map.append(h["dev_mean_map"])

fig, ax = plt.subplots()

# Zero-shot baseline line (DINOv2 + random head)
zero_shot_mean = 38.7
ax.axhline(y=zero_shot_mean, color=COLORS["gray"], linestyle=":", linewidth=1.5,
           alpha=0.6)
ax.text(24.5, zero_shot_mean + 0.4, "Zero-shot baseline (38.7)", fontsize=9,
        color=COLORS["gray"], ha="right")

ax.plot(eval_epochs, zf, color=COLORS["primary"], linewidth=1.8,
        marker="o", markersize=5, alpha=0.75, label="AAUZebraFish (fish)")
ax.plot(eval_epochs, pb, color=COLORS["orange"], linewidth=1.8,
        marker="s", markersize=5, alpha=0.75, label="PolarBearVidID (video)")
ax.plot(eval_epochs, sm, color=COLORS["green"], linewidth=1.8,
        marker="^", markersize=5, alpha=0.75, label="SMALST (synthetic)")
ax.plot(eval_epochs, mean_map, color=COLORS["accent"], linewidth=3,
        marker="D", markersize=7, label="Mean (tracked for early-stop)")

# Highlight best
best_i = int(np.argmax(mean_map))
ax.scatter([eval_epochs[best_i]], [mean_map[best_i]], s=280,
           facecolors="none", edgecolors=COLORS["accent"], linewidths=3, zorder=5)
ax.annotate(f"Best: epoch {eval_epochs[best_i]}\nmean mAP = {mean_map[best_i]:.2f}",
            xy=(eval_epochs[best_i], mean_map[best_i]),
            xytext=(eval_epochs[best_i] + 2, mean_map[best_i] - 8),
            fontsize=10, color=COLORS["accent"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["accent"]))

ax.set_xlabel("Epoch")
ax.set_ylabel("mAP (%)")
ax.set_title("Cross-domain dev mAP — proxy for Dataset B generalization")
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right")
ax.set_xticks(eval_epochs)
ax.set_ylim(25, 70)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "02_cross_domain_dev_map.png")
plt.close()
print(f"Wrote {CHARTS_DIR / '02_cross_domain_dev_map.png'}")


# ============================================================================
# 3) Dataset A final metrics vs VLM baseline (bar chart)
# ============================================================================
metrics = ["Rank-1", "Rank-5", "Rank-10", "Rank-20", "mAP", "Combined"]
ours = [99.95, 100.00, 100.00, 100.00, 88.19, 94.07]
vlm = [60, 80, None, None, 50, 55]  # Rank-10/20 unreported for VLM

fig, ax = plt.subplots()
x = np.arange(len(metrics))
width = 0.38

ours_bars = ax.bar(x - width/2, ours, width, label="Ours (DINOv2 ViT-S/14)",
                   color=COLORS["primary"], edgecolor="white", linewidth=1.5)
vlm_vals = [v if v is not None else 0 for v in vlm]
vlm_bars = ax.bar(x + width/2, vlm_vals, width, label="VLM baseline (provided)",
                  color=COLORS["gray"], edgecolor="white", linewidth=1.5,
                  alpha=0.7)

# Annotate our bars
for b, v in zip(ours_bars, ours):
    label = f"{v:.2f}" if v < 100 else "100.00"
    ax.text(b.get_x() + b.get_width()/2, v + 1, label,
            ha="center", va="bottom", fontsize=10,
            color=COLORS["primary"], fontweight="bold")

# Annotate VLM bars (hatch out the missing ones)
for b, v in zip(vlm_bars, vlm):
    if v is None:
        b.set_hatch("//")
        b.set_alpha(0.3)
        ax.text(b.get_x() + b.get_width()/2, 3, "n/a",
                ha="center", va="bottom", fontsize=9, color=COLORS["gray"])
    else:
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"~{v}",
                ha="center", va="bottom", fontsize=10, color=COLORS["gray"])

ax.set_ylabel("Metric value (%)")
ax.set_title("Dataset A performance — fine-tuned model vs provided VLM baseline")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(True, alpha=0.25, axis="y")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "03_dataset_a_vs_baseline.png")
plt.close()
print(f"Wrote {CHARTS_DIR / '03_dataset_a_vs_baseline.png'}")


# ============================================================================
# 4) Efficiency: throughput + memory + embed dim on log scale
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# Throughput (img/s, log scale because VLM is 1000x slower)
ax = axes[0]
names = ["VLM\nbaseline", "Ours\n(MPS bf16)", "Ours\n(projected CUDA)"]
tp = [0.2, 255, 700]  # last is estimate for CUDA
colors = [COLORS["gray"], COLORS["primary"], COLORS["green"]]
bars = ax.bar(names, tp, color=colors, edgecolor="white", linewidth=1.5)
ax.set_yscale("log")
ax.set_ylabel("Throughput (images / second)")
ax.set_title("Throughput (log scale)")
for b, v in zip(bars, tp):
    ax.text(b.get_x() + b.get_width()/2, v * 1.15, f"{v:g}",
            ha="center", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.25, axis="y", which="both")

# Memory
ax = axes[1]
names = ["VLM\nbaseline", "Ours"]
mem = [60 * 1024, 1500]  # MB
colors = [COLORS["gray"], COLORS["primary"]]
bars = ax.bar(names, mem, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Peak memory (MB)")
ax.set_title("Peak memory")
for b, v in zip(bars, mem):
    label = f"{v/1024:.0f} GB" if v >= 1024 else f"{v:.0f} MB"
    ax.text(b.get_x() + b.get_width()/2, v + 1500, label,
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(mem) * 1.18)
ax.grid(True, alpha=0.25, axis="y")

# Embedding dim
ax = axes[2]
names = ["VLM\nbaseline", "Ours"]
dims = [384, 256]
colors = [COLORS["gray"], COLORS["primary"]]
bars = ax.bar(names, dims, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Embedding dimension")
ax.set_title("Embedding size")
for b, v in zip(bars, dims):
    ax.text(b.get_x() + b.get_width()/2, v + 8, str(v),
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 450)
ax.grid(True, alpha=0.25, axis="y")

fig.suptitle("Efficiency — 1,275× faster, 40× less memory, 1.5× smaller embeddings",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "04_efficiency.png")
plt.close()
print(f"Wrote {CHARTS_DIR / '04_efficiency.png'}")


# ============================================================================
# 5) Backbone benchmark on MPS (from day-1 run)
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 4.8))
configs = [
    ("DINOv2 ViT-S/14\nfp32", 243, COLORS["primary"]),
    ("DINOv2 ViT-S/14\nbf16", 255, COLORS["primary"]),
    ("ConvNeXt-Tiny\nfp32", 277, COLORS["gray"]),
    ("ConvNeXt-Tiny\nbf16", 261, COLORS["gray"]),
]
names = [c[0] for c in configs]
vals = [c[1] for c in configs]
cols = [c[2] for c in configs]

bars = ax.bar(names, vals, color=cols, edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 4, f"{v} img/s",
            ha="center", fontsize=11, fontweight="bold")

ax.axhline(y=30, color=COLORS["accent"], linestyle="--", linewidth=1.5)
ax.text(3.5, 35, "30 img/s floor",
        color=COLORS["accent"], fontsize=10, ha="right")

ax.set_ylabel("Throughput (images / second)")
ax.set_title("Backbone benchmark on Apple Silicon MPS (batch 32)")
ax.set_ylim(0, 320)
ax.grid(True, alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "05_backbone_benchmark.png")
plt.close()
print(f"Wrote {CHARTS_DIR / '05_backbone_benchmark.png'}")


# ============================================================================
# 6) Dataset composition — image counts across 37 OpenAnimals sub-datasets
# ============================================================================
import pandas as pd
df = pd.read_parquet(ROOT / "datasets" / "dataset_a" / "train.parquet")
df["sub_dataset"] = df["image_path"].str.split("/").str[1]
counts = df["sub_dataset"].value_counts().sort_values(ascending=True)

holdouts = {"AAUZebraFish", "PolarBearVidID", "SMALST"}
bar_colors = [COLORS["accent"] if s in holdouts else COLORS["primary"]
              for s in counts.index]

fig, ax = plt.subplots(figsize=(10, 9))
y = np.arange(len(counts))
bars = ax.barh(y, counts.values, color=bar_colors, edgecolor="white")
ax.set_yticks(y)
ax.set_yticklabels(counts.index, fontsize=9)
ax.set_xlabel("Training images")
ax.set_title("OpenAnimals — 37 sub-datasets (red = held out as cross-domain dev)")

for b, v in zip(bars, counts.values):
    ax.text(v + 100, b.get_y() + b.get_height()/2, f"{v:,}",
            va="center", fontsize=8)

ax.grid(True, alpha=0.25, axis="x")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "06_dataset_composition.png")
plt.close()
print(f"Wrote {CHARTS_DIR / '06_dataset_composition.png'}")


print(f"\nAll charts generated in {CHARTS_DIR}/")
