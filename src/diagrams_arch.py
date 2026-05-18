"""
diagrams_arch.py -- The ViT backbone, drawn as concept diagrams.

  * draw_vit_pipeline -- the end-to-end flow: image -> tokens -> features.
  * draw_patchify     -- how a real photo is cut into patches and each
                         patch is flattened + linearly embedded.
  * draw_token_layout -- the token sequence [CLS | registers | patches].
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from diagram_utils import arrow, blank_canvas, box, caption, token_strip
from plot_style import PALETTE


def draw_vit_pipeline():
    """End-to-end: how a ViT turns an image into token features."""
    fig, ax = blank_canvas(12.5, 3.6, (0, 25), (0, 8))
    stages = [
        ("Input\nimage", PALETTE[7]),
        ("Patchify +\nlinear embed", PALETTE[0]),
        ("Prepend CLS,\nadd positions", PALETTE[1]),
        ("Transformer\nblocks x L", PALETTE[2]),
        ("Final\nLayerNorm", PALETTE[4]),
        ("CLS vector +\npatch tokens", PALETTE[3]),
    ]
    w, h, gap = 3.3, 3.8, 0.65
    for i, (text, color) in enumerate(stages):
        x = 0.4 + i * (w + gap)
        box(ax, x, 2.7, w, h, text, color=color, fontsize=9)
        if i:
            arrow(ax, (x - gap - 0.02, 4.6), (x + 0.02, 4.6))
    caption(ax, 12.5, 1.5,
            "Self-attention lets every patch token exchange information with "
            "every other; the CLS token aggregates the whole image.")
    ax.set_title("How a Vision Transformer processes an image",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def draw_patchify(image_rgb, n_side: int = 7):
    """Cut a real photo into patches; flatten one into a token vector."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [1, 0.62, 1]})
    side = image_rgb.shape[0]
    step = side / n_side
    ax = axes[0]
    ax.imshow(image_rgb)
    for i in range(1, n_side):
        ax.axhline(i * step, color="white", lw=1.0)
        ax.axvline(i * step, color="white", lw=1.0)
    pr, pc = 2, 3
    ax.add_patch(plt.Rectangle((pc * step, pr * step), step, step,
                               fill=False, edgecolor=PALETTE[3], lw=3.5))
    ax.set_title(f"1. Split into a {n_side}x{n_side} patch grid")
    ax.axis("off")

    ax = axes[1]
    patch = image_rgb[int(pr * step):int((pr + 1) * step),
                      int(pc * step):int((pc + 1) * step)]
    ax.imshow(patch)
    ax.set_title("2. One patch")
    ax.axis("off")

    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    ax.set_title("3. Flatten + linear layer")
    values = np.random.default_rng(0).random(9)
    for i, value in enumerate(values):
        box(ax, 4.4, 0.7 + i * 0.92, 1.7, 0.84, f"{value:.2f}",
            color=plt.cm.viridis(value), text_color="black", fontsize=7)
    arrow(ax, (1.4, 5.0), (4.2, 5.0))
    caption(ax, 5.2, 9.5, "token vector (D = 384 / 768 / 1024)")
    caption(ax, 1.6, 5.9, "flatten")
    fig.suptitle("Patchify: an image becomes a sequence of token vectors",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def draw_token_layout():
    """The token sequence the transformer sees: CLS, registers, patches."""
    fig, ax = blank_canvas(12, 3.6, (0, 20), (0, 7))
    size = 1.12
    labels = ["CLS"] + [f"R{i}" for i in range(1, 5)] + \
             [f"P{i}" for i in range(1, 11)] + ["..."]
    colors = [PALETTE[3]] + [PALETTE[1]] * 4 + [PALETTE[0]] * 10 + ["#c8c8c8"]
    token_strip(ax, 0.5, 3.6, size, labels, colors, fontsize=7)
    caption(ax, 0.5 + size * 0.5, 2.9, "global\nsummary", 8)
    caption(ax, 0.5 + size * 3.0, 2.9,
            "register tokens -- scratch space that\nabsorbs high-norm "
            "attention artefacts", 8)
    caption(ax, 0.5 + size * 10.5, 2.9,
            "patch tokens -- one per image patch", 8)
    caption(ax, 10, 1.2,
            "The CLS token feeds image-level tasks (classification, "
            "retrieval); the patch tokens feed dense tasks "
            "(segmentation, depth, matching).")
    ax.set_title("The token sequence:  [ CLS | registers | patches ]",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig
