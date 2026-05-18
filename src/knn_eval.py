"""
knn_eval.py -- k-Nearest-Neighbour evaluation of frozen features.

The cleanest evidence that a self-supervised backbone learned something
useful: take its features *frozen* -- no training, no fine-tuning -- and
classify a test image by the labels of its nearest training neighbours.
A high k-NN accuracy means the representation already groups images by
semantic content. This is the standard label-free probe for DINO models.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch


@torch.no_grad()
def knn_predict(train_x: torch.Tensor, train_y: torch.Tensor,
                test_x: torch.Tensor, k: int = 20,
                num_classes: int = 10) -> torch.Tensor:
    """
    Cosine-similarity k-NN classifier.

    Inputs are L2-normalised CLS features, so ``test_x @ train_x.T`` is
    the cosine similarity. Each test point votes among its `k` nearest
    neighbours, weighted by similarity (closer neighbours count more).
    """
    sims = test_x @ train_x.t()                       # (n_test, n_train)
    top_sim, top_idx = sims.topk(k, dim=1)
    neighbour_labels = train_y[top_idx]               # (n_test, k)
    weights = top_sim.clamp(min=0)
    votes = torch.zeros(test_x.size(0), num_classes)
    for c in range(num_classes):
        votes[:, c] = (weights * (neighbour_labels == c)).sum(dim=1)
    return votes.argmax(dim=1)


@torch.no_grad()
def knn_accuracy(train_x, train_y, test_x, test_y, k: int = 20,
                 num_classes: int = 10) -> float:
    """Top-1 k-NN accuracy in [0, 1]."""
    pred = knn_predict(train_x, train_y, test_x, k, num_classes)
    return (pred == test_y).float().mean().item()


@torch.no_grad()
def knn_sweep(train_x, train_y, test_x, test_y, ks: Sequence[int],
              num_classes: int = 10) -> Dict[int, float]:
    """k-NN accuracy for several values of `k`, as ``{k: accuracy}``."""
    return {int(k): knn_accuracy(train_x, train_y, test_x, test_y,
                                 int(k), num_classes) for k in ks}
