"""
decomposition.py -- Dimensionality reduction for feature visualization.

A patch feature is a high-dimensional vector (384-1024 numbers). To *see*
it we must project to something a human can read: 3 dimensions mapped to
RGB, or 2 dimensions for a scatter plot. PCA, t-SNE and UMAP are the three
projectors used across the project.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config


def _to_numpy(x) -> np.ndarray:
    """Accept a torch tensor or array-like; return a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def pca_components(grid: torch.Tensor, n: int = 3) -> np.ndarray:
    """
    First `n` PCA components of a patch grid ``(h, w, D)``, each rescaled
    to [0, 1] and returned as ``(h, w, n)``.
    """
    arr = _to_numpy(grid)
    h, w, d = arr.shape
    comps = PCA(n_components=n, random_state=config.SEED).fit_transform(
        arr.reshape(h * w, d))
    lo, hi = comps.min(0), comps.max(0)
    scaled = (comps - lo) / (hi - lo + 1e-8)
    return scaled.reshape(h, w, n)


def pca_rgb(grid: torch.Tensor) -> np.ndarray:
    """
    Map a patch grid ``(h, w, D)`` to an ``(h, w, 3)`` RGB image: each of
    the top 3 principal components drives one colour channel, so patches
    with similar features get similar colours and semantic regions emerge.
    """
    return pca_components(grid, n=3)


def foreground_mask(grid: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Boolean ``(h, w)`` foreground mask from the 1st PCA component.

    The leading component of self-supervised patch features reliably
    separates object from background -- thresholding it gives a free,
    label-free segmentation. The mask is oriented so the foreground is
    the minority region (objects are usually smaller than the background).
    """
    first = pca_components(grid, n=1)[:, :, 0]
    mask = first > threshold
    if mask.mean() > 0.5:           # threshold caught the background -> flip
        mask = ~mask
    return mask


def tsne(features, perplexity: float | None = None) -> np.ndarray:
    """2-D t-SNE projection of ``(N, D)`` embeddings."""
    perplexity = perplexity or config.EVAL.tsne_perplexity
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=config.SEED).fit_transform(_to_numpy(features))


def umap_proj(features, n_neighbors: int = 15) -> np.ndarray:
    """2-D UMAP projection of ``(N, D)`` embeddings."""
    import umap                       # lazy import: pulls in numba/llvmlite
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        random_state=config.SEED)
    return reducer.fit_transform(_to_numpy(features))
