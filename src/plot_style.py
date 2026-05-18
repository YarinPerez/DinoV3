"""
plot_style.py -- Shared matplotlib styling and the figure-saving helper.

A single consistent look is applied to every figure in the project. Each
figure is saved as a PNG *plus* a JSON "sidecar" recording the parameters
and metrics that produced it -- so any result can be inspected and
reproduced later, as CLAUDE.md requires.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")          # headless backend: no display server needed
import matplotlib.pyplot as plt  # noqa: E402

# A clean, consistent look applied to every figure.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Colour-blind-friendly categorical palette (used for class colours etc.).
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

# Perceptually-uniform colormaps for heatmaps.
HEATMAP_CMAP = "magma"      # attention / similarity intensity
DIVERGE_CMAP = "RdBu_r"     # signed quantities (e.g. cosine similarity)


def save_figure(fig, path, metadata: Optional[dict] = None) -> Path:
    """
    Save `fig` to `path` as PNG and write a `<path>.json` sidecar.

    The sidecar stores `metadata` (e.g. model name, image, measured
    numbers) so every PNG carries the recipe that produced it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    sidecar = path.with_suffix(path.suffix + ".json")
    payload = {"figure": path.name, **(metadata or {})}
    sidecar.write_text(json.dumps(payload, indent=2, default=str))
    return path


def grid_figure(n_cols: int, n_rows: int, panel: float = 3.0):
    """Create a (n_rows x n_cols) figure of square `panel`-inch axes."""
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel * n_cols, panel * n_rows))
    return fig, axes
