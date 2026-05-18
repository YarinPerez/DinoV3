"""
linear_probe.py -- Linear-probe evaluation of frozen features.

A linear probe trains *only* a single linear layer (a matrix multiply) on
top of frozen backbone features. Because the backbone itself is never
touched, the probe's accuracy measures how *linearly separable* the
representation already is -- the standard yardstick for comparing
self-supervised models. We also emit a per-class report and a confusion
matrix, as CLAUDE.md requires for machine-learning work.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

import config


def train_probe(train_x: torch.Tensor, train_y: torch.Tensor,
                num_classes: int, device, epochs: int | None = None,
                lr: float | None = None) -> nn.Linear:
    """
    Fit a linear classifier on frozen features with full-batch AdamW.

    Frozen features are cheap to store, so the whole training set fits in
    one batch -- the optimisation is fast and deterministic.
    """
    epochs = epochs or config.EVAL.probe_epochs
    lr = lr or config.EVAL.probe_lr
    probe = nn.Linear(train_x.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr,
                                  weight_decay=1e-4)
    x, y = train_x.to(device), train_y.to(device)
    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(x), y)
        loss.backward()
        optimizer.step()
    return probe.eval()


@torch.no_grad()
def evaluate_probe(probe: nn.Linear, test_x: torch.Tensor,
                   test_y: torch.Tensor, class_names: List[str],
                   device) -> Dict:
    """
    Evaluate a trained probe: overall accuracy, a per-class
    precision/recall/F1 report, and the confusion matrix.
    """
    pred = probe(test_x.to(device)).argmax(dim=1).cpu()
    true = test_y.cpu()
    accuracy = (pred == true).float().mean().item()
    report = classification_report(
        true.numpy(), pred.numpy(), target_names=class_names,
        output_dict=True, zero_division=0)
    matrix = confusion_matrix(true.numpy(), pred.numpy())
    return {"accuracy": accuracy, "report": report,
            "confusion_matrix": matrix}
