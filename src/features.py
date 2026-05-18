"""
features.py -- Extract features from a (frozen) DINOv2 backbone.

Three kinds of feature are used throughout the project:
  * the CLS token       -- one vector summarising the whole image,
  * the patch tokens    -- one vector per image patch (a spatial grid),
  * intermediate layers -- the patch grid as it looks after each block.

Every function here runs under ``torch.no_grad()``: the backbones are
frozen feature extractors, never trained.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def forward_tokens(model, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run the backbone; return CLS / register / patch tokens."""
    out = model.forward_features(batch)
    return {
        "cls": out["x_norm_clstoken"],          # (B, D)
        "registers": out["x_norm_regtokens"],   # (B, R, D)
        "patches": out["x_norm_patchtokens"],   # (B, P, D)
    }


@torch.no_grad()
def cls_features(model, batch: torch.Tensor, l2: bool = True) -> torch.Tensor:
    """Return CLS embeddings ``(B, D)``, L2-normalised by default."""
    cls = model.forward_features(batch)["x_norm_clstoken"]
    return F.normalize(cls, dim=-1) if l2 else cls


def grid_shape(image_hw: Tuple[int, int], patch_size: int) -> Tuple[int, int]:
    """Map an ``(H, W)`` image size to its ``(h, w)`` patch-grid size."""
    return image_hw[0] // patch_size, image_hw[1] // patch_size


@torch.no_grad()
def patch_grid(model, image: torch.Tensor, patch_size: int,
               l2: bool = False) -> torch.Tensor:
    """
    Patch tokens of a single image, reshaped to a spatial grid ``(h, w, D)``.

    `image` is a ``(3, H, W)`` tensor; the batch dimension is added here.
    """
    patches = model.forward_features(image.unsqueeze(0))
    patches = patches["x_norm_patchtokens"][0]            # (P, D)
    h, w = grid_shape(image.shape[-2:], patch_size)
    grid = patches.reshape(h, w, -1)
    return F.normalize(grid, dim=-1) if l2 else grid


@torch.no_grad()
def layer_grids(model, image: torch.Tensor,
                layers: Sequence[int]) -> List[torch.Tensor]:
    """
    Patch-token grids ``(h, w, D)`` taken from several transformer blocks,
    via DINOv2's ``get_intermediate_layers`` helper. Lets us watch the
    representation sharpen with depth.
    """
    feats = model.get_intermediate_layers(
        image.unsqueeze(0), n=list(layers), reshape=True, norm=True)
    return [f[0].permute(1, 2, 0) for f in feats]   # (1,D,h,w) -> (h,w,D)


@torch.no_grad()
def dataset_features(model, loader, device,
                     limit: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract L2-normalised CLS features for every image in `loader`.

    Returns ``(X, y)`` with X of shape ``(N, D)`` on the CPU and y the
    integer labels -- the input to k-NN, the linear probe and t-SNE.
    """
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    seen = 0
    for images, targets in loader:
        feats.append(cls_features(model, images.to(device), l2=True).cpu())
        labels.append(targets)
        seen += len(targets)
        if limit is not None and seen >= limit:
            break
    x = torch.cat(feats)
    y = torch.cat(labels)
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y
