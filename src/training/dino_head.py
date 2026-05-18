"""
dino_head.py -- The DINO projection head.

The backbone produces an embedding; the head turns it into a score over
``out_dim`` learned "prototypes". An MLP narrows the embedding to a
bottleneck, L2-normalises it (this normalisation is important for
stability), then a final linear layer compares it to every prototype.
A softmax over those scores is the distribution the DINO loss matches.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """MLP -> L2-normalised bottleneck -> linear layer over prototypes."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512,
                 bottleneck_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # The original DINO weight-normalises this layer; a plain linear
        # layer is enough at the small scale of this teaching demo.
        self.prototypes = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.prototypes(x)              # (B, out_dim) logits
