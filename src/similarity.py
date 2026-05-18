"""
similarity.py -- Patch-to-patch similarity and dense correspondence.

DINO patch features carry meaning in their *direction*: two patches that
depict the same kind of content point the same way. After L2-normalising,
a dot product between two patch vectors is exactly their cosine
similarity. From that single operation we get:

  * self-similarity  -- pick one patch, score every patch by similarity
    to it (the interactive "click-a-patch" demo),
  * dense correspondence -- for each patch of image A, find its most
    similar patch in image B (matching parts across images).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _flat_normalized(grid: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """``(h, w, D)`` grid -> (L2-normalised ``(h*w, D)``, ``(h, w)``)."""
    h, w, d = grid.shape
    flat = F.normalize(grid.reshape(h * w, d), dim=-1)
    return flat, (h, w)


def self_similarity(grid: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """
    Cosine similarity of the patch at ``(row, col)`` to every patch,
    returned as an ``(h, w)`` map with values in [-1, 1].
    """
    flat, (h, w) = _flat_normalized(grid)
    query = flat[row * w + col]
    return (flat @ query).reshape(h, w)


def similarity_matrix(grid: torch.Tensor) -> torch.Tensor:
    """Full ``(h*w, h*w)`` patch-to-patch cosine-similarity matrix."""
    flat, _ = _flat_normalized(grid)
    return flat @ flat.t()


def correspondence(grid_a: torch.Tensor,
                   grid_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For every patch in grid A, find its best-matching patch in grid B.

    Returns ``(match_index, match_score)``: a flat index into B's patch
    grid and the cosine similarity of that best match, one entry per
    patch of A.
    """
    flat_a, _ = _flat_normalized(grid_a)
    flat_b, _ = _flat_normalized(grid_b)
    similarity = flat_a @ flat_b.t()        # (patches_A, patches_B)
    score, index = similarity.max(dim=1)
    return index, score
