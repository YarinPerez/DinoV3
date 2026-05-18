"""
tiny_vit.py -- A small Vision Transformer for the from-scratch demo.

This is a miniature of the same architecture DINOv2/DINOv3 use: a conv
patch embedding, a learned CLS token, learned positional embeddings (with
bicubic interpolation so one model handles the different crop sizes), and
a stack of pre-norm transformer blocks. It is deliberately tiny (~3-6M
parameters) so it trains in minutes on one GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Conv-based patch embedding: an image becomes a sequence of tokens."""

    def __init__(self, patch_size: int, in_chans: int = 3,
                 embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, h*w, D)


class Block(nn.Module):
    """Pre-norm transformer block: self-attention + MLP, both residual."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """Compact ViT: patch embed + CLS token + positional embed + blocks."""

    def __init__(self, patch_size: int = 8, embed_dim: int = 192,
                 depth: int = 6, heads: int = 6, base_grid: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.base_grid = base_grid
        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, base_grid * base_grid + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _positions(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Positional embeddings, bicubically resized to (grid_h, grid_w)."""
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        if grid_h == self.base_grid and grid_w == self.base_grid:
            return self.pos_embed
        dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, self.base_grid, self.base_grid, dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos, size=(grid_h, grid_w), mode="bicubic",
            align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_tokens(self, x):
        """Return ``(cls_token, patch_tokens)`` after the final LayerNorm."""
        b, _, height, width = x.shape
        tokens = self.patch_embed(x)
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self._positions(grid_h, grid_w)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x):
        """Return just the CLS token -- the image-level embedding."""
        return self.forward_tokens(x)[0]
