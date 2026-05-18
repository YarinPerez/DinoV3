"""
dinov3_loader.py -- Optional Hugging Face DINOv3 path.

The real DINOv3 weights are *gated*: they require a Hugging Face account,
acceptance of Meta's licence and an authentication token. This module
loads them when they are available and fails cleanly (raising
``Dinov3Unavailable``) when they are not -- so the project always runs on
the open DINOv2 backbones regardless of access.

To enable this path:
    uv run huggingface-cli login        # paste a token
    # then accept the licence at huggingface.co/facebook/dinov3-vitb16-...
"""
from __future__ import annotations

import config
from device import get_device


class Dinov3Unavailable(RuntimeError):
    """Raised when the gated DINOv3 weights cannot be loaded."""


def probe_dinov3() -> str:
    """Return a one-line availability status string. Never raises."""
    try:
        from huggingface_hub import model_info
        model_info(config.DINOV3_HF_ID)
        return f"available -- {config.DINOV3_HF_ID} is reachable"
    except Exception as exc:
        return (f"gated/unreachable ({type(exc).__name__}) -- "
                f"the project will use DINOv2 only")


def load_dinov3():
    """
    Return ``(model, image_processor)`` for DINOv3.

    Raises ``Dinov3Unavailable`` if transformers is missing or the gated
    weights cannot be downloaded; callers should catch it and continue.
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise Dinov3Unavailable(f"transformers unavailable: {exc}")
    try:
        model = AutoModel.from_pretrained(config.DINOV3_HF_ID)
        processor = AutoImageProcessor.from_pretrained(config.DINOV3_HF_ID)
    except Exception as exc:
        raise Dinov3Unavailable(
            f"cannot load gated weights {config.DINOV3_HF_ID}: {exc}")
    return model.to(get_device()).eval(), processor
