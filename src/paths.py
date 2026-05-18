"""
paths.py -- Resolves and creates the project's output directories.

All generated artefacts live under assets/<group>/ ; datasets under data/ ;
training checkpoints under checkpoints/. Every one of these is git-ignored
and fully regenerable by re-running the scripts.
"""
from __future__ import annotations

from pathlib import Path

# Project root = parent of the src/ directory that contains this file.
ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS: Path = ROOT / "assets"
DATA: Path = ROOT / "data"
CHECKPOINTS: Path = ROOT / "checkpoints"
DOCS: Path = ROOT / "docs"

# One sub-folder per visualization group (see docs/VISUALIZATION_GUIDE.md).
ASSET_GROUPS = [
    "diagrams",        # 01 -- architecture & concept diagrams
    "features",        # 02 -- PCA feature maps
    "attention",       # 03 -- attention maps
    "similarity",      # 04 -- patch similarity heatmaps
    "correspondence",  # 05 -- cross-image matching
    "embeddings",      # 06 -- t-SNE / UMAP projections
    "evaluation",      # 07, 09 -- k-NN, linear probe, model comparison
    "resolution",      # 08 -- resolution sweep
    "retrieval",       # 10 -- nearest-neighbour retrieval
    "training",        # 12 -- mini-training curves & feature emergence
]


def group_dir(name: str) -> Path:
    """Return assets/<name>/, creating it on demand."""
    d = ASSETS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs() -> None:
    """Create every output directory up-front (idempotent)."""
    for base in (ASSETS, DATA, CHECKPOINTS):
        base.mkdir(parents=True, exist_ok=True)
    for group in ASSET_GROUPS:
        (ASSETS / group).mkdir(parents=True, exist_ok=True)
