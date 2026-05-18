"""
diagrams_train.py -- The DINO self-distillation training loop.

  * draw_student_teacher -- two networks: a student trained by gradient
    descent and a teacher that is an EMA copy of the student.
  * draw_multicrop       -- the multi-crop strategy: 2 global + several
    local crops; the teacher only ever sees the global crops.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from diagram_utils import arrow, blank_canvas, box, caption
from plot_style import PALETTE


def draw_student_teacher():
    """The student/teacher setup with EMA update and stop-gradient."""
    fig, ax = blank_canvas(12.5, 5.4, (0, 25), (0, 12))
    box(ax, 0.5, 4.0, 3.6, 4.0, "two augmented\nviews of the\nsame image",
        color=PALETTE[7], fontsize=8.5)
    box(ax, 6.2, 7.4, 5.6, 3.2, "TEACHER\n(EMA of student;\nstop-gradient)",
        color=PALETTE[2], fontsize=8.5)
    box(ax, 6.2, 1.4, 5.6, 3.2, "STUDENT\n(updated by\ngradient descent)",
        color=PALETTE[0], fontsize=8.5)
    box(ax, 15.0, 4.6, 5.2, 3.0, "cross-entropy:\nstudent matches\nteacher  "
        "(DINO loss)", color=PALETTE[3], fontsize=8.5)
    arrow(ax, (4.1, 6.4), (6.2, 8.6))
    arrow(ax, (4.1, 5.6), (6.2, 3.2))
    arrow(ax, (11.8, 8.6), (15.0, 6.6))
    arrow(ax, (11.8, 3.0), (15.0, 5.4))
    # gradient flows back to the student only (L-shaped, red)
    arrow(ax, (17.6, 4.6), (17.6, 0.7), color=PALETTE[3])
    arrow(ax, (17.6, 0.7), (9.2, 1.4), color=PALETTE[3])
    caption(ax, 18.8, 0.3, "gradient updates the STUDENT only",
            8.5, color=PALETTE[3], weight="bold")
    # EMA copies student weights into the teacher (dashed)
    arrow(ax, (9.0, 4.6), (9.0, 7.4), color=PALETTE[5], ls="--")
    caption(ax, 10.3, 6.0, "EMA copy", 8.5, color=PALETTE[5], weight="bold")
    caption(ax, 7.0, 0.3,
            "EMA update:  teacher <- m * teacher + (1-m) * student   "
            "(m ~ 0.996)", 8.5, color=PALETTE[5])
    ax.set_title("DINO self-distillation: a student learns from an "
                 "EMA teacher", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def draw_multicrop(image_rgb):
    """Multi-crop augmentation: 2 global crops + several local crops."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.imshow(image_rgb)
    side = image_rgb.shape[0]
    globals_ = [(0.05, 0.05, 0.8), (0.18, 0.12, 0.78)]
    locals_ = [(0.10, 0.55, 0.3), (0.45, 0.10, 0.32), (0.62, 0.55, 0.3),
               (0.40, 0.45, 0.28), (0.05, 0.30, 0.26), (0.70, 0.18, 0.27)]
    for (x, y, s) in globals_:
        ax.add_patch(plt.Rectangle((x * side, y * side), s * side, s * side,
                     fill=False, edgecolor=PALETTE[2], lw=3))
    for (x, y, s) in locals_:
        ax.add_patch(plt.Rectangle((x * side, y * side), s * side, s * side,
                     fill=False, edgecolor=PALETTE[1], lw=2, ls="--"))
    ax.set_title("2 global crops (solid) + 6 local crops (dashed)")
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    box(ax, 0.5, 6.6, 4.0, 2.2, "global\ncrops", color=PALETTE[2],
        fontsize=9)
    box(ax, 0.5, 1.4, 4.0, 2.2, "local\ncrops", color=PALETTE[1],
        fontsize=9)
    box(ax, 6.4, 6.6, 3.1, 2.2, "TEACHER", color=PALETTE[2], fontsize=9)
    box(ax, 6.4, 1.4, 3.1, 2.2, "STUDENT", color=PALETTE[0], fontsize=9)
    arrow(ax, (4.5, 7.7), (6.4, 7.7))
    arrow(ax, (4.5, 7.0), (6.4, 3.0))
    arrow(ax, (4.5, 2.5), (6.4, 2.5))
    caption(ax, 5, 0.4,
            "The teacher sees only global crops; the student sees all of "
            "them. Matching a local crop to a global one forces a "
            "'part implies whole' representation.")
    fig.suptitle("Multi-crop: many views of one image",
                 fontweight="bold")
    fig.tight_layout()
    return fig
