"""
08_resolution_sweep.py -- Feature granularity vs input resolution.

A ViT tiles its input into fixed-size patches, so a larger input image
yields more patches and therefore a finer feature grid. This sweep feeds
the same photo at four resolutions and shows how the patch-feature map
gets denser. Handling such rescaling cleanly is exactly what DINOv3's
rotary position embeddings (RoPE) are designed for.

    uv run python scripts/08_resolution_sweep.py
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
from decomposition import pca_rgb  # noqa: E402
from device import get_device  # noqa: E402
from features import patch_grid  # noqa: E402
from image_io import preprocess, to_displayable  # noqa: E402
from plot_style import save_figure  # noqa: E402


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("resolution")
    device = get_device()
    model, spec = load_backbone(config.DEFAULT_BACKBONE)
    _, pil, _ = sample_images(1, spec.patch_size)[0]

    resolutions = config.RESOLUTION_SWEEP
    fig, axes = plt.subplots(2, len(resolutions),
                             figsize=(3.0 * len(resolutions), 6.3))
    for col, requested in enumerate(resolutions):
        tensor = preprocess(pil, requested, spec.patch_size)
        grid = patch_grid(model, tensor.to(device), spec.patch_size, l2=True)
        h, w, _ = grid.shape
        axes[0, col].imshow(to_displayable(tensor))
        axes[0, col].set_title(f"input @ {tensor.shape[-1]} px")
        axes[1, col].imshow(pca_rgb(grid))
        axes[1, col].set_title(f"{h}x{w} = {h * w} patch tokens")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle("Higher resolution -> a finer patch-feature grid  "
                 f"({spec.label}; RoPE lets DINOv3 rescale seamlessly)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out / "resolution_feature_maps.png",
                {"group": "resolution", "script": "08",
                 "model": spec.label, "resolutions": resolutions})
    print(f"resolution figure -> {out}")


if __name__ == "__main__":
    main()
