"""
06_embedding_projection.py -- Project CLS embeddings to 2-D.

The CLS token is a single vector summarising a whole image. t-SNE and
UMAP squeeze those 768-D vectors down to 2-D so we can plot them. If the
self-supervised backbone learned meaningful features, images of the same
class land near each other -- forming clusters that DINO was never told
about, because training used no labels at all.

    uv run python scripts/06_embedding_projection.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from datasets import STL10_CLASSES  # noqa: E402
from decomposition import tsne, umap_proj  # noqa: E402
from eval_features import get_eval_features  # noqa: E402
from plot_style import PALETTE, save_figure  # noqa: E402


def scatter(ax, coords, labels, title):
    """Scatter 2-D points coloured by class label."""
    for c, name in enumerate(STL10_CLASSES):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=9, color=PALETTE[c],
                   label=name, alpha=0.7, edgecolors="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("embeddings")
    _, _, test_x, test_y = get_eval_features(config.DEFAULT_BACKBONE)

    n = min(1500, len(test_x))
    features, labels = test_x[:n], test_y[:n].numpy()
    coords_tsne = tsne(features)
    coords_umap = umap_proj(features)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    scatter(axes[0], coords_tsne, labels, "t-SNE of CLS embeddings")
    scatter(axes[1], coords_umap, labels, "UMAP of CLS embeddings")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8, markerscale=1.6)
    fig.suptitle("Frozen DINO features cluster by class -- although no "
                 "labels were used in training", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    save_figure(fig, out / "embedding_projection.png",
                {"group": "embeddings", "script": "06",
                 "model": config.DEFAULT_BACKBONE, "n_points": n})
    print(f"embedding projections -> {out}")


if __name__ == "__main__":
    main()
