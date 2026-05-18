"""
04_patch_similarity.py -- Patch-to-patch similarity ("click a patch").

Because patch features are L2-normalised, the dot product between two
patches is their cosine similarity. Pick one patch and score every other
patch against it: patches showing the same kind of content light up,
even far away in the image. The full similarity matrix shows the same
thing globally -- a block structure that mirrors the scene's regions.

    uv run python scripts/04_patch_similarity.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from backbone_loader import load_backbone  # noqa: E402
from datasets import sample_images  # noqa: E402
from device import get_device  # noqa: E402
from features import patch_grid  # noqa: E402
from image_io import to_displayable  # noqa: E402
from plot_style import DIVERGE_CMAP, PALETTE, save_figure  # noqa: E402
from similarity import self_similarity, similarity_matrix  # noqa: E402


def click_figure(model, spec, sample, device):
    """Input with 3 query patches marked, then a similarity map for each."""
    grid = patch_grid(model, sample[2].to(device), spec.patch_size, l2=True)
    h, w, _ = grid.shape
    queries = [(h // 3, w // 2), (h // 6, w // 6), (3 * h // 4, w // 2)]
    disp = to_displayable(sample[2])
    patch = spec.patch_size
    fig, axes = plt.subplots(1, 1 + len(queries),
                             figsize=(3.4 * (1 + len(queries)), 3.7))
    axes[0].imshow(disp)
    axes[0].set_title("query patches")
    for i, (r, c) in enumerate(queries):
        axes[0].add_patch(plt.Rectangle((c * patch, r * patch), patch, patch,
                          fill=False, edgecolor=PALETTE[i], lw=3))
    for i, (r, c) in enumerate(queries):
        sim = self_similarity(grid, r, c).cpu().numpy()
        image = axes[i + 1].imshow(sim, cmap=DIVERGE_CMAP, vmin=-1, vmax=1,
                                   interpolation="bilinear")
        axes[i + 1].scatter([c], [r], s=90, edgecolor="black",
                            facecolor=PALETTE[i], zorder=3)
        axes[i + 1].set_title(f"cosine similarity to query {i + 1}")
        fig.colorbar(image, ax=axes[i + 1], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Click a patch -> every similar patch lights up "
                 f"({spec.label})", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def matrix_figure(model, spec, sample, device):
    """The full patch-to-patch cosine-similarity matrix."""
    grid = patch_grid(model, sample[2].to(device), spec.patch_size, l2=True)
    matrix = similarity_matrix(grid).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.grid(False)
    image = ax.imshow(matrix, cmap=DIVERGE_CMAP, vmin=-1, vmax=1)
    ax.set_title(f"Patch-to-patch similarity matrix "
                 f"({matrix.shape[0]} patches)")
    ax.set_xlabel("patch index")
    ax.set_ylabel("patch index")
    fig.colorbar(image, ax=ax, fraction=0.046, label="cosine similarity")
    fig.tight_layout()
    return fig


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("similarity")
    device = get_device()
    model, spec = load_backbone(config.DEFAULT_BACKBONE)
    sample = sample_images(2, spec.patch_size)[0]
    meta = {"group": "similarity", "script": "04", "model": spec.label}

    save_figure(click_figure(model, spec, sample, device),
                out / "click_a_patch.png", meta)
    save_figure(matrix_figure(model, spec, sample, device),
                out / "similarity_matrix.png", meta)
    print(f"similarity figures -> {out}")


if __name__ == "__main__":
    main()
