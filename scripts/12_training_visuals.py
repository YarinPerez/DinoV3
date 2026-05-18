"""
12_training_visuals.py -- Visualize the mini DINO training run.

Reads the history and checkpoints written by script 11 and produces:
  * training_curves   -- loss, k-NN accuracy and EMA momentum,
  * feature_emergence -- patch-feature PCA of one image at several
    epochs, so you literally watch structure appear out of noise.

Run script 11 first.

    uv run python scripts/12_training_visuals.py
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "training"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from datasets import stl10  # noqa: E402
from decomposition import pca_rgb  # noqa: E402
from device import get_device  # noqa: E402
from image_io import preprocess, to_displayable  # noqa: E402
from plot_style import PALETTE, save_figure  # noqa: E402
from tiny_vit import TinyViT  # noqa: E402


def curves_figure(history):
    """Loss, k-NN-accuracy and EMA-momentum curves, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].plot(history["loss"], color=PALETTE[3], lw=2)
    axes[0].set_title("DINO loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss  (lower = student tracks teacher)")
    axes[1].plot(history["knn_epoch"], history["knn"], "o-",
                 color=PALETTE[2], lw=2)
    axes[1].axhline(1.0 / config.TRAIN.num_classes, ls="--", color="gray",
                    label="chance (10%)")
    axes[1].set_title("teacher k-NN accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy  (labels used only to measure)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[2].plot(history["momentum"], color=PALETTE[5], lw=2)
    axes[2].set_title("EMA teacher momentum")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("momentum  (cosine schedule)")
    fig.suptitle("Mini DINO training -- the learning signal in three views",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def emergence_figure(device):
    """Patch-feature PCA of one image at increasing training epochs."""
    cfg = config.TRAIN
    pil = stl10("test")[0][0]
    tensor = preprocess(pil, 96, cfg.patch_size).to(device)
    epochs = list(cfg.pca_track_epochs)
    fig, axes = plt.subplots(1, len(epochs) + 1,
                             figsize=(3.0 * (len(epochs) + 1), 3.4))
    axes[0].imshow(to_displayable(tensor))
    axes[0].set_title("input image")
    axes[0].axis("off")
    grid_side = 96 // cfg.patch_size
    for i, epoch in enumerate(epochs):
        model = TinyViT(cfg.patch_size, cfg.embed_dim, cfg.depth,
                        cfg.num_heads,
                        base_grid=cfg.global_crop // cfg.patch_size).to(device)
        state = torch.load(paths.CHECKPOINTS / f"teacher_epoch{epoch}.pt",
                           map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            _, patches = model.forward_tokens(tensor.unsqueeze(0))
        grid = patches[0].reshape(grid_side, grid_side, -1)
        axes[i + 1].imshow(pca_rgb(grid))
        axes[i + 1].set_title(f"epoch {epoch}")
        axes[i + 1].axis("off")
    fig.suptitle("Patch-feature structure emerging from noise during "
                 "training", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("training")
    device = get_device()
    history_path = paths.CHECKPOINTS / "history.json"
    if not history_path.exists():
        raise SystemExit("No training history found -- run script 11 first.")
    history = json.loads(history_path.read_text())

    meta = {"group": "training", "script": "12"}
    save_figure(curves_figure(history), out / "training_curves.png", meta)
    save_figure(emergence_figure(device), out / "feature_emergence.png", meta)
    print(f"training figures -> {out}")


if __name__ == "__main__":
    main()
