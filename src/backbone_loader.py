"""
backbone_loader.py -- Load pretrained DINOv2 ViT backbones from torch.hub.

DINOv2 is openly downloadable and mechanically near-identical to DINOv3
(same Vision Transformer, same DINO + iBOT + KoLeo training, optional
register tokens), so it powers every live visualization in this project.
See docs/PRD.md section 4.1 for why we use DINOv2 rather than the gated
DINOv3 weights.
"""
from __future__ import annotations

import os

# Disable xFormers *before* the hub code is imported: PyTorch's native
# attention is used instead, which keeps the attention maths interceptable
# (see src/attention_hook.py). This also silences noisy warnings.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from functools import lru_cache  # noqa: E402

import torch  # noqa: E402

import config  # noqa: E402
from device import get_device, maybe_compile  # noqa: E402

_HUB_VERBOSE = False


@lru_cache(maxsize=None)
def load_backbone(key: str, compile_model: bool = False):
    """
    Return (model, spec) for the DINOv2 backbone identified by `key`.

    The weights are downloaded once (torch.hub caches them), moved to the
    best available device and set to eval() mode -- we never train these
    backbones, we only read their frozen features.

    Set `compile_model=True` for batched feature extraction (k-NN, probe)
    where torch.compile() speeds things up. Leave it False whenever you
    attach an attention hook: compilation would freeze the original,
    un-hooked forward pass and the hook would never fire.
    """
    if key not in config.DINOV2_BACKBONES:
        raise KeyError(f"unknown backbone '{key}'; "
                       f"choices: {list(config.DINOV2_BACKBONES)}")
    spec = config.DINOV2_BACKBONES[key]
    model = torch.hub.load(config.DINOV2_HUB_REPO, spec.hub_name,
                           verbose=_HUB_VERBOSE)
    model = model.to(get_device()).eval()
    for param in model.parameters():        # frozen: no gradients needed
        param.requires_grad_(False)
    if compile_model:
        model = maybe_compile(model)
    return model, spec


def describe(spec) -> str:
    """One-line human-readable summary of a backbone spec."""
    return (f"{spec.label}: {spec.depth} blocks, {spec.num_heads} heads, "
            f"embed dim {spec.embed_dim}, patch {spec.patch_size}, "
            f"{spec.num_registers} register tokens")
