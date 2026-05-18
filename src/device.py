"""
device.py -- Hardware selection and the torch.compile() wrapper.

CLAUDE.md asks for "compiled" code to improve runtime performance. For a
PyTorch project the idiomatic tool is torch.compile(), which JIT-compiles
the model graph (via TorchInductor) into fused, optimised kernels. We wrap
it so that a compilation failure on any host silently and safely falls
back to ordinary eager execution.
"""
from __future__ import annotations

import warnings

import torch


def get_device() -> torch.device:
    """Pick the best available compute device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable one-line description of `device`."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"CUDA | {name} | {total_gb:.1f} GB VRAM"
    if device.type == "mps":
        return "Apple MPS (Metal GPU)"
    return "CPU"


def maybe_compile(model, enabled: bool = True):
    """
    Return torch.compile(model) when possible, else the model unchanged.

    A try/except guard means an unsupported host (old GPU, missing
    compiler toolchain, ...) never breaks a run -- it just loses the
    speed-up. The warning makes the fallback visible.
    """
    if not enabled:
        return model
    try:
        return torch.compile(model)
    except Exception as exc:  # pragma: no cover -- host-dependent
        warnings.warn(f"torch.compile() unavailable, using eager mode: {exc}")
        return model
