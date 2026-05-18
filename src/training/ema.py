"""
ema.py -- The EMA teacher: weight averaging and its momentum schedule.

The teacher is never trained directly. After every student step its
weights are nudged a little towards the student's:

    teacher <- m * teacher + (1 - m) * student

With m close to 1 the teacher is a slow, smoothed average of the
student's whole recent history -- a more stable target than the student
itself. `m` follows a cosine schedule, rising towards 1 so the teacher
settles down as training progresses.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def ema_update(student: torch.nn.Module, teacher: torch.nn.Module,
               momentum: float) -> None:
    """Move every teacher parameter a step towards the student's."""
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.mul_(momentum).add_(s_param.detach(), alpha=1.0 - momentum)
    # Buffers (e.g. LayerNorm running stats) are copied outright.
    for s_buf, t_buf in zip(student.buffers(), teacher.buffers()):
        t_buf.copy_(s_buf)


def cosine_momentum(step: int, total_steps: int, base: float,
                    final: float) -> float:
    """Cosine schedule for the EMA momentum, rising from `base` to `final`."""
    progress = min(step / max(total_steps, 1), 1.0)
    return final - (final - base) * (math.cos(math.pi * progress) + 1) / 2
