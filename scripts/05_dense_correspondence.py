"""
05_dense_correspondence.py -- Match patches across two images.

For every patch of image A, find the patch in image B whose feature
vector is most similar. Drawing a line for each match shows that DINO
features are *instance-invariant*: an ear matches an ear, a paw a paw,
across two different photos of the same kind of animal -- with no
supervision and no fine-tuning.

    uv run python scripts/05_dense_correspondence.py
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
from datasets import similar_pair  # noqa: E402
from device import get_device  # noqa: E402
from features import patch_grid  # noqa: E402
from image_io import to_displayable  # noqa: E402
from plot_style import save_figure  # noqa: E402
from similarity import correspondence  # noqa: E402


def correspondence_figure(model, spec, pair, device, step: int = 3):
    """Two images side by side; lines link best-matching patches."""
    grid_a = patch_grid(model, pair[0][2].to(device), spec.patch_size, l2=True)
    grid_b = patch_grid(model, pair[1][2].to(device), spec.patch_size, l2=True)
    index, score = correspondence(grid_a, grid_b)
    index, score = index.cpu().numpy(), score.cpu().numpy()
    h, w, _ = grid_a.shape
    patch = spec.patch_size
    disp_a, disp_b = to_displayable(pair[0][2]), to_displayable(pair[1][2])
    side = disp_a.shape[0]
    gap = int(0.12 * side)

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.imshow(disp_a, extent=(0, side, side, 0))
    ax.imshow(disp_b, extent=(side + gap, 2 * side + gap, side, 0))
    ax.set_xlim(0, 2 * side + gap)
    ax.set_ylim(side, 0)
    ax.axis("off")

    sampled = [r * w + c for r in range(1, h, step) for c in range(1, w, step)]
    keep_above = np.quantile(score[sampled], 0.4)   # show the top ~60%
    cmap = plt.cm.viridis
    for flat in sampled:
        if score[flat] < keep_above:
            continue
        ar, ac = divmod(flat, w)
        br, bc = divmod(int(index[flat]), w)
        xa, ya = ac * patch + patch / 2, ar * patch + patch / 2
        xb, yb = side + gap + bc * patch + patch / 2, br * patch + patch / 2
        colour = cmap(float(score[flat]))
        ax.plot([xa, xb], [ya, yb], "-", color=colour, lw=1.0, alpha=0.7)
        ax.scatter([xa, xb], [ya, yb], s=9, color=colour, zorder=3)

    scalar = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(keep_above, 1.0))
    fig.colorbar(scalar, ax=ax, fraction=0.03, label="match cosine similarity")
    fig.suptitle("Dense correspondence: matching parts across two images "
                 f"({spec.label})", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("correspondence")
    device = get_device()
    model, spec = load_backbone(config.DEFAULT_BACKBONE)
    pair = similar_pair(spec.patch_size)

    save_figure(correspondence_figure(model, spec, pair, device),
                out / "dense_correspondence.png",
                {"group": "correspondence", "script": "05",
                 "model": spec.label})
    print(f"correspondence figure -> {out}")


if __name__ == "__main__":
    main()
