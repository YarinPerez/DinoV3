"""
03_attention_maps.py -- Visualize self-attention.

Self-attention is how every token decides which other tokens to listen
to. Reading off the CLS token's attention over the image patches shows
*where the model looks*. Three figures:
  * per-head maps   -- different heads specialise on different regions,
  * attention overlay -- attention concentrates on the salient object,
  * register comparison -- extra "register" tokens soak up the bright
    high-norm artefacts that otherwise pollute the attention map.

    uv run python scripts/03_attention_maps.py
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
from attention_hook import capture_attention  # noqa: E402
from backbone_loader import load_backbone  # noqa: E402
from datasets import sample_images  # noqa: E402
from device import get_device  # noqa: E402
from features import grid_shape  # noqa: E402
from image_io import to_displayable  # noqa: E402
from plot_style import HEATMAP_CMAP, save_figure  # noqa: E402


def cls_attention(model, image, n_registers, patch):
    """Per-head attention of the CLS token over the patch grid (H,h,w)."""
    with capture_attention(model, -1) as store:
        model.forward_features(image.unsqueeze(0))
    attn = store[0][0]                              # (heads, N, N)
    h, w = grid_shape(image.shape[-2:], patch)
    maps = attn[:, 0, 1 + n_registers:].reshape(-1, h, w)
    return maps.float().cpu().numpy()


def _norm(x):
    """
    Normalise a map to [0, 1] using the 1st/99th percentiles.

    Plain min-max would let a single high-norm artefact patch (a known
    DINOv2-without-registers quirk, visible as a bright corner spot)
    crush all the real structure into near-zero. Clipping at percentiles
    keeps the artefact saturated while spreading the useful range.
    """
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def heads_figure(model, spec, sample, device):
    """Input + every attention head + the head-averaged map."""
    maps = cls_attention(model, sample[2].to(device), spec.num_registers,
                         spec.patch_size)
    n, cols = maps.shape[0], 5
    rows = -(-(n + 2) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.ravel()
    axes[0].imshow(to_displayable(sample[2]))
    axes[0].set_title("input")
    for i in range(n):
        axes[i + 1].imshow(_norm(maps[i]), cmap=HEATMAP_CMAP,
                           interpolation="bilinear")
        axes[i + 1].set_title(f"head {i + 1}", fontsize=9)
    axes[n + 1].imshow(_norm(maps.mean(0)), cmap=HEATMAP_CMAP,
                       interpolation="bilinear")
    axes[n + 1].set_title("mean of heads", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"CLS-token attention (last block) -- {spec.label}\n"
                 "each head specialises on a different part of the scene",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def overlay_figure(model, spec, samples, device):
    """Top row: images. Bottom row: mean attention overlaid on them."""
    fig, axes = plt.subplots(2, len(samples),
                             figsize=(2.7 * len(samples), 5.6))
    for c, (_, _, tensor) in enumerate(samples):
        disp = to_displayable(tensor)
        mean = cls_attention(model, tensor.to(device), spec.num_registers,
                             spec.patch_size).mean(0)
        axes[0, c].imshow(disp)
        axes[1, c].imshow(disp)
        axes[1, c].imshow(_norm(mean), cmap=HEATMAP_CMAP, alpha=0.6,
                          extent=(0, disp.shape[1], disp.shape[0], 0),
                          interpolation="bilinear")
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    fig.suptitle(f"Attention concentrates on the salient object "
                 f"({spec.label})", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def register_figure(sample, device):
    """ViT-S attention with vs without register tokens."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    axes[0].imshow(to_displayable(sample[2]))
    axes[0].set_title("input")
    for ax, key, title in [(axes[1], "vits14", "ViT-S/14 (no registers)"),
                           (axes[2], "vits14_reg", "ViT-S/14 + 4 registers")]:
        model, spec = load_backbone(key)
        mean = cls_attention(model, sample[2].to(device), spec.num_registers,
                             spec.patch_size).mean(0)
        ax.imshow(_norm(mean), cmap=HEATMAP_CMAP, interpolation="bilinear")
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Register tokens absorb the high-norm attention artefacts",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("attention")
    device = get_device()
    model, spec = load_backbone(config.DEFAULT_BACKBONE)
    samples = sample_images(4, spec.patch_size)
    meta = {"group": "attention", "script": "03", "model": spec.label}

    save_figure(heads_figure(model, spec, samples[0], device),
                out / "attention_heads.png", meta)
    save_figure(overlay_figure(model, spec, samples, device),
                out / "attention_overlay.png", meta)
    save_figure(register_figure(samples[0], device),
                out / "register_comparison.png",
                {"group": "attention", "script": "03"})
    print(f"attention figures -> {out}")


if __name__ == "__main__":
    main()
