"""
diagrams_loss.py -- The DINO loss and DINOv3's Gram anchoring.

  * draw_dino_loss      -- how the student is trained to match the teacher
    distribution, with centering + sharpening to prevent collapse.
  * draw_gram_anchoring -- DINOv3's headline fix: why dense features
    degrade over long training and how anchoring the Gram matrix to an
    early checkpoint repairs them. (Curves are schematic illustrations.)
"""
from __future__ import annotations

import numpy as np

from diagram_utils import arrow, blank_canvas, box, caption
from plot_style import PALETTE, plt


def draw_dino_loss():
    """The teacher/student distribution-matching loss."""
    fig, ax = blank_canvas(12.5, 5.0, (0, 27), (0, 12))
    teacher = [("teacher\noutput", PALETTE[2]), ("- center", PALETTE[5]),
               ("softmax\n/ tau_t  (sharp)", PALETTE[2]),
               ("P_teacher\n(peaky)", PALETTE[2])]
    student = [("student\noutput", PALETTE[0]),
               ("softmax\n/ tau_s", PALETTE[0]),
               ("P_student", PALETTE[0])]
    for i, (text, color) in enumerate(teacher):
        x = 0.5 + i * 5.0
        box(ax, x, 8.2, 4.0, 2.6, text, color=color, fontsize=8)
        if i:
            arrow(ax, (x - 1.05, 9.5), (x - 0.05, 9.5))
    for i, (text, color) in enumerate(student):
        x = 0.5 + i * 5.0
        box(ax, x, 1.0, 4.0, 2.6, text, color=color, fontsize=8)
        if i:
            arrow(ax, (x - 1.05, 2.3), (x - 0.05, 2.3))
    box(ax, 21.0, 4.6, 5.2, 2.8,
        "cross-entropy\nH(P_teacher, P_student)", color=PALETTE[3],
        fontsize=8.5)
    arrow(ax, (16.5, 8.2), (22.0, 7.4))
    arrow(ax, (12.5, 3.6), (22.0, 5.2))
    caption(ax, 13.5, 0.2,
            "Centering (subtract a running mean) and sharpening (a small "
            "teacher temperature tau_t) stop the network from collapsing "
            "to one constant output.")
    ax.set_title("The DINO loss: the student matches a sharpened, "
                 "centred teacher", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def draw_gram_anchoring():
    """Why dense features degrade, and how Gram anchoring repairs them."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Panel A: schematic quality-vs-training curves ---------------
    ax = axes[0]
    t = np.linspace(0, 1, 200)
    global_q = 1 - np.exp(-4.5 * t)
    dense_unanchored = np.exp(-((t - 0.4) ** 2) / 0.05) * 0.95
    dense_unanchored[:80] = global_q[:80] * 0.95
    dense_anchored = 1 - np.exp(-4.0 * t)
    ax.plot(t, global_q, color=PALETTE[0], lw=2.5,
            label="global / CLS quality")
    ax.plot(t, dense_unanchored, color=PALETTE[3], lw=2.5,
            label="dense quality - no anchoring")
    ax.plot(t, dense_anchored, color=PALETTE[2], lw=2.5, ls="--",
            label="dense quality - Gram anchoring")
    ax.axvspan(0.55, 1.0, color=PALETTE[2], alpha=0.08)
    ax.text(0.77, 0.52, "anchoring\nphase", ha="center", fontsize=8,
            color=PALETTE[2], style="italic")
    ax.set_xlabel("training progress")
    ax.set_ylabel("feature quality (schematic)")
    ax.set_title("Long training degrades dense features")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, 1.08)

    # --- Panel B: Gram-matrix alignment schematic -------------------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    box(ax, 0.4, 6.6, 4.3, 2.6, "current model\npatch features",
        color=PALETTE[0], fontsize=8)
    box(ax, 0.4, 1.0, 4.3, 2.6, "Gram teacher\n(frozen early\ncheckpoint)",
        color=PALETTE[2], fontsize=8)
    box(ax, 5.6, 6.6, 4.0, 2.6, "Gram matrix\nG_student", color=PALETTE[0],
        fontsize=8)
    box(ax, 5.6, 1.0, 4.0, 2.6, "Gram matrix\nG_teacher", color=PALETTE[2],
        fontsize=8)
    arrow(ax, (4.7, 7.9), (5.6, 7.9))
    arrow(ax, (4.7, 2.3), (5.6, 2.3))
    arrow(ax, (7.6, 6.6), (7.6, 3.6), color=PALETTE[3])
    ax.text(8.2, 5.1, "align:\n|| G_s - G_t ||", color=PALETTE[3],
            fontsize=8, weight="bold", va="center")
    caption(ax, 5, 0.1,
            "The Gram matrix is patch-to-patch similarity. Anchoring it "
            "fixes the *relational* structure without freezing features.")
    fig.suptitle("Gram anchoring: DINOv3's fix for dense-feature decay",
                 fontweight="bold")
    fig.tight_layout()
    return fig
