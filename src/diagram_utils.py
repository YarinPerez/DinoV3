"""
diagram_utils.py -- Small matplotlib primitives for concept diagrams.

The architecture and training diagrams are drawn from a few shared
building blocks -- labelled boxes, arrows, token strips -- so each diagram
module can stay focused on layout rather than low-level patch drawing.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import PALETTE


def blank_canvas(width: float, height: float, xlim, ylim):
    """Create a figure + axis with no ticks/spines -- a blank drawing area."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def box(ax, x, y, w, h, text="", color=PALETTE[0], text_color="white",
        fontsize=9, alpha=1.0):
    """Draw a rounded, labelled box; (x, y) is the bottom-left corner."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor="black", linewidth=1.1, alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, weight="bold")


def arrow(ax, start, end, color="#222222", style="-|>", lw=1.7, ls="-"):
    """Draw an arrow from `start` to `end` (each an (x, y) tuple)."""
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=15, linestyle=ls,
        color=color, lw=lw, shrinkA=2, shrinkB=2))


def token_strip(ax, x, y, size, labels, colors, fontsize=8):
    """Draw a horizontal strip of square tokens, each labelled inside."""
    for i, (label, color) in enumerate(zip(labels, colors)):
        box(ax, x + i * size, y, size * 0.9, size * 0.9, label,
            color=color, fontsize=fontsize)


def caption(ax, x, y, text, fontsize=9, color="#333333", weight="normal"):
    """Place a short caption / annotation at (x, y)."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic", weight=weight)
