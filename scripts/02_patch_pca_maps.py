"""
02_patch_pca_maps.py -- Visualize dense patch features with PCA.

Every image patch becomes a high-dimensional feature vector. Projecting
those vectors onto their top 3 principal components and showing the
result as RGB reveals that DINO groups patches by *meaning*: an object
is painted one set of colours, the background another. Thresholding the
first principal component alone yields a free foreground segmentation.

    uv run python scripts/02_patch_pca_maps.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from backbone_loader import load_backbone  # noqa: E402
from datasets import sample_images  # noqa: E402
from decomposition import foreground_mask, pca_rgb  # noqa: E402
from device import get_device  # noqa: E402
from features import layer_grids, patch_grid  # noqa: E402
from image_io import to_displayable  # noqa: E402
from plot_style import save_figure  # noqa: E402


def _overlay(disp: np.ndarray, mask: np.ndarray, patch: int) -> np.ndarray:
    """Dim the background by an upsampled foreground mask."""
    big = np.kron(mask.astype(float), np.ones((patch, patch)))
    big = big[:disp.shape[0], :disp.shape[1], None]
    return disp * (0.30 + 0.70 * big)


def pca_maps_figure(model, spec, samples, device):
    """One row per image: input | PCA-RGB feature map | foreground."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(9.5, 3.0 * len(samples)))
    for r, (_, _, tensor) in enumerate(samples):
        grid = patch_grid(model, tensor.to(device), spec.patch_size, l2=True)
        disp = to_displayable(tensor)
        axes[r, 0].imshow(disp)
        axes[r, 1].imshow(pca_rgb(grid))
        axes[r, 2].imshow(_overlay(disp, foreground_mask(grid), spec.patch_size))
        for c in range(3):
            axes[r, c].axis("off")
    for c, title in enumerate(["input", "feature PCA (RGB)",
                               "foreground mask"]):
        axes[0, c].set_title(title, fontsize=11)
    fig.suptitle(f"Dense patch features group by meaning -- {spec.label}",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def layerwise_figure(model, spec, sample, device):
    """PCA-RGB feature maps taken at increasing transformer depth."""
    layers = [2, 5, 8, spec.depth - 1]
    grids = layer_grids(model, sample[2].to(device), layers)
    fig, axes = plt.subplots(1, len(layers) + 1,
                             figsize=(3 * (len(layers) + 1), 3.3))
    axes[0].imshow(to_displayable(sample[2]))
    axes[0].set_title("input")
    axes[0].axis("off")
    for i, (layer, grid) in enumerate(zip(layers, grids)):
        axes[i + 1].imshow(pca_rgb(grid))
        axes[i + 1].set_title(f"after block {layer + 1}")
        axes[i + 1].axis("off")
    fig.suptitle("Patch features sharpen with depth -- early blocks look "
                 f"like texture, late blocks like parts ({spec.label})",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("features")
    device = get_device()
    model, spec = load_backbone(config.DEFAULT_BACKBONE)
    samples = sample_images(5, spec.patch_size)

    meta = {"group": "features", "script": "02", "model": spec.label}
    save_figure(pca_maps_figure(model, spec, samples, device),
                out / "pca_feature_maps.png", meta)
    save_figure(layerwise_figure(model, spec, samples[0], device),
                out / "layerwise_pca.png", meta)
    print(f"feature PCA figures -> {out}")


if __name__ == "__main__":
    main()
