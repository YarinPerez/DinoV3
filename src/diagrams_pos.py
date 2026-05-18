"""
diagrams_pos.py -- Positional encoding: learned embeddings vs RoPE.

A transformer is permutation-invariant: without positional information it
cannot tell a patch's location. This figure contrasts the two schemes:

  * DINOv2 -- a *learned* vector per grid position, added to each token.
  * DINOv3 -- *rotary* position embeddings (RoPE): query/key vectors are
    rotated by an angle proportional to position, so attention depends on
    *relative* position and the model generalises to new resolutions.
"""
from __future__ import annotations

import numpy as np

from diagram_utils import arrow, box, caption
from plot_style import PALETTE, plt


def draw_positional_encoding():
    """Side-by-side comparison of learned vs rotary positional encoding."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- Panel A: learned positional embeddings (DINOv2) --------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    ax.set_title("DINOv2: learned positional embeddings")
    for i in range(4):
        x = 0.7 + i * 2.3
        box(ax, x, 6.2, 1.7, 1.4, f"patch\n{i + 1}", color=PALETTE[0],
            fontsize=8)
        ax.text(x + 0.85, 5.6, "+", ha="center", fontsize=15, weight="bold")
        box(ax, x, 3.8, 1.7, 1.4, f"pos {i + 1}", color=PALETTE[1],
            fontsize=8)
    caption(ax, 5, 2.4,
            "One learned vector per grid position is added to each patch "
            "token. Simple, but the grid size is fixed at training time -- "
            "new resolutions need interpolation.")

    # --- Panel B: rotary positional embeddings / RoPE (DINOv3) -------
    ax = axes[1]
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.grid(False)
    ax.set_title("DINOv3: rotary embeddings (RoPE)")
    for i, pos in enumerate([0, 1, 2, 3]):
        angle = pos * np.pi / 6
        ax.annotate("", xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE[i],
                                    lw=2.4))
        ax.text(1.12 * np.cos(angle), 1.12 * np.sin(angle), f"pos {pos}",
                color=PALETTE[i], fontsize=8, weight="bold",
                ha="center", va="center")
    ax.text(0, -1.28,
            "The query/key vector is rotated by an angle proportional to\n"
            "position. A dot product then depends only on the *relative*\n"
            "angle -- so RoPE extends cleanly to unseen image sizes.",
            ha="center", va="center", fontsize=8.5, style="italic")
    fig.suptitle("Telling the transformer *where* each patch is",
                 fontweight="bold")
    fig.tight_layout()
    return fig
