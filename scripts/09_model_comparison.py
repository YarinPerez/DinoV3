"""
09_model_comparison.py -- ViT-S vs ViT-B vs ViT-L, head to head.

Bigger Vision Transformers cost more parameters and more time per image,
but produce richer features. This script measures all four trade-offs --
parameter count, embedding width, forward latency and k-NN accuracy --
and plots them side by side.

    uv run python scripts/09_model_comparison.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from backbone_loader import load_backbone  # noqa: E402
from device import get_device  # noqa: E402
from eval_features import get_eval_features  # noqa: E402
from knn_eval import knn_accuracy  # noqa: E402
from plot_style import PALETTE, save_figure  # noqa: E402


def measure(key, device):
    """Collect capacity, speed and accuracy stats for one backbone."""
    model, spec = load_backbone(key)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                        device=device)
    with torch.no_grad():
        for _ in range(3):                       # warm-up
            model.forward_features(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            model.forward_features(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency = (time.time() - start) / 10 * 1000   # ms / image
    train_x, train_y, test_x, test_y = get_eval_features(key)
    knn = knn_accuracy(train_x, train_y, test_x, test_y,
                       config.EVAL.knn_k, config.TRAIN.num_classes)
    return {"label": spec.label, "params": params, "dim": spec.embed_dim,
            "latency": latency, "knn": knn}


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("evaluation")
    device = get_device()
    stats = {k: measure(k, device) for k in config.COMPARISON_KEYS}
    labels = [stats[k]["label"] for k in stats]

    panels = [("parameters (millions)", "params", "{:.1f}"),
              ("embedding dimension", "dim", "{:.0f}"),
              ("forward latency (ms/image)", "latency", "{:.1f}"),
              ("k-NN accuracy (STL-10)", "knn", "{:.3f}")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (title, field, fmt) in zip(axes, panels):
        values = [stats[k][field] for k in stats]
        ax.bar(labels, values, color=PALETTE[:len(stats)])
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.18)
        for i, v in enumerate(values):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    fig.suptitle("Scaling the backbone: capacity and cost vs feature "
                 "quality", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out / "model_comparison.png",
                {"group": "evaluation", "script": "09", "stats": stats})
    print("stats:", {k: (round(stats[k]["params"], 1),
                         round(stats[k]["knn"], 3)) for k in stats})
    print(f"model comparison -> {out}")


if __name__ == "__main__":
    main()
