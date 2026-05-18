"""
eval_features.py -- Cached frozen-feature extraction for evaluation.

Scripts 06, 07, 09 and 10 all need the same thing: L2-normalised CLS
features for the STL-10 train/test splits, per backbone. Extracting them
is the slow step, so the result is cached to ``data/feature_cache/`` and
every later script reads it back instantly.
"""
from __future__ import annotations

from typing import Tuple

import torch

import config
import paths
from backbone_loader import load_backbone
from datasets import eval_loaders
from device import get_device
from features import dataset_features

Tensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def get_eval_features(model_key: str) -> Tensors:
    """
    Return ``(train_x, train_y, test_x, test_y)`` for one backbone.

    Features are computed once and cached on disk; subsequent calls (and
    subsequent scripts) load the cache in a fraction of a second.
    """
    cache = paths.DATA / "feature_cache" / f"{model_key}.pt"
    if cache.exists():
        data = torch.load(cache)
        return data["train_x"], data["train_y"], data["test_x"], data["test_y"]

    model, spec = load_backbone(model_key)
    device = get_device()
    train_loader, test_loader = eval_loaders(spec.patch_size)
    train_x, train_y = dataset_features(model, train_loader, device,
                                        config.EVAL.train_samples)
    test_x, test_y = dataset_features(model, test_loader, device,
                                      config.EVAL.test_samples)

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_x": train_x, "train_y": train_y,
                "test_x": test_x, "test_y": test_y}, cache)
    return train_x, train_y, test_x, test_y
