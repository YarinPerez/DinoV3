"""
attention_hook.py -- Capture self-attention maps from a DINOv2 backbone.

Why this module exists
----------------------
DINOv2's attention block calls ``torch.nn.functional.scaled_dot_product_
attention`` -- a *fused* kernel that never materialises the softmax(QK^T)
matrix. A normal forward hook therefore cannot observe the attention
weights at all.

The trick used here: temporarily replace one attention module's
``forward`` with a mathematically identical re-implementation that *also*
records the attention matrix. A context manager restores the original
forward on exit, so the model is left untouched.

This module is the linchpin for every attention figure (script 03).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List

import torch


def attention_modules(model) -> list:
    """
    Return every attention sub-module of `model`, in depth order.

    Detection is duck-typed (a module with ``qkv``, ``proj`` and
    ``num_heads``) so it works whether or not the ViT uses block chunking.
    """
    mods = [m for m in model.modules()
            if hasattr(m, "qkv") and hasattr(m, "proj")
            and hasattr(m, "num_heads")]
    if not mods:
        raise RuntimeError("no attention modules found in this model")
    return mods


def _capturing_forward(module, x: torch.Tensor, store: List[torch.Tensor]):
    """
    Re-implementation of DINOv2 ``Attention.forward`` that records the
    softmax attention matrix into `store`.

    It computes exactly what scaled_dot_product_attention computes at
    eval time (no mask, no dropout):  softmax(Q Kᵀ / sqrt(d)) @ V.
    """
    B, N, C = x.shape
    head_dim = C // module.num_heads
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
    q, k, v = torch.unbind(qkv, 2)                      # each (B, N, H, d)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # -> (B, H, N, d)
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)                         # (B, H, N, N)
    store.append(attn.detach())
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    return module.proj_drop(module.proj(out))


@contextmanager
def capture_attention(model, block_index: int = -1):
    """
    Context manager yielding a list that, after a forward pass, holds the
    attention tensor ``(batch, heads, tokens, tokens)`` of one block.

    Token order along both axes is ``[CLS, registers..., patches...]`` --
    so ``attn[0, head, 0, 1 + R:]`` is the CLS token's attention over the
    image patches, ready to reshape into a spatial map.

    `block_index` indexes into the depth-ordered attention modules
    (``-1`` = last block, the one most studies visualise).
    """
    module = attention_modules(model)[block_index]
    store: List[torch.Tensor] = []
    had_own_forward = "forward" in module.__dict__
    original = module.__dict__.get("forward")
    module.forward = lambda x, *a, **kw: _capturing_forward(module, x, store)
    try:
        yield store
    finally:                                  # always restore the model
        if had_own_forward:
            module.forward = original
        else:
            module.__dict__.pop("forward", None)
