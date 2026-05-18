"""
dino_loss.py -- The (simplified) DINO self-distillation loss.

The student is trained so that, for two different crops of one image,
its output distribution matches the teacher's. Two safeguards stop the
networks cheating by emitting one constant vector ("collapse"):

  * sharpening -- a low teacher temperature makes the teacher's target
    distribution peaky and confident;
  * centering  -- a running mean is subtracted from the teacher logits,
    so no single prototype can dominate every image.

This is the DINO loss; the full DINOv2/DINOv3 objective adds the iBOT
masked-patch term and KoLeo regularisation, omitted here for clarity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Cross-entropy of the student against a centred, sharpened teacher."""

    def __init__(self, out_dim: int, student_temp: float,
                 center_momentum: float):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs, teacher_temp):
        """
        `student_outputs` : list of (B, out_dim) logits, one per crop.
        `teacher_outputs` : list of (B, out_dim) logits, global crops only.
        Crops at the same index are the same view and are skipped.
        """
        teacher_probs = [
            F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
            for t in teacher_outputs]
        student_logs = [F.log_softmax(s / self.student_temp, dim=-1)
                        for s in student_outputs]
        total, n_terms = 0.0, 0
        for ti, t_prob in enumerate(teacher_probs):
            for si, s_log in enumerate(student_logs):
                if si == ti:                       # same crop -> skip
                    continue
                total = total - (t_prob * s_log).sum(dim=-1).mean()
                n_terms += 1
        self._update_center(teacher_outputs)
        return total / max(n_terms, 1)

    @torch.no_grad()
    def _update_center(self, teacher_outputs):
        """EMA-update the centering vector from the latest teacher logits."""
        batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum)
        self.center.add_(batch_center * (1 - self.center_momentum))
