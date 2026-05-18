"""
image_io.py -- Load and preprocess images for the DINOv2 backbones.

A Vision Transformer can only tile an image whose height and width are
exact multiples of its patch size, so every image is resized to such a
square. Pixels are then normalised with the ImageNet statistics the
backbones were trained on. `denormalize` inverts that step for display.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import config


def round_to_patch(size: int, patch: int) -> int:
    """Largest multiple of `patch` not exceeding `size` (at least `patch`)."""
    return max(patch, (size // patch) * patch)


def preprocess(img: Image.Image, image_size: int = config.IMAGE_SIZE,
                patch: int = 14) -> torch.Tensor:
    """
    Convert a PIL image to a normalised ``(3, S, S)`` tensor, where S is a
    multiple of `patch`. The shorter side is resized, then a centre crop
    keeps the aspect ratio undistorted.
    """
    side = round_to_patch(image_size, patch)
    pipeline = transforms.Compose([
        transforms.Resize(side, antialias=True),
        transforms.CenterCrop(side),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return pipeline(img.convert("RGB"))


def eval_transform(image_size: int = config.EVAL.image_size, patch: int = 14):
    """A torchvision transform (PIL -> tensor) for the evaluation datasets."""
    return lambda img: preprocess(img, image_size, patch)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation -> values back in [0, 1], ready to plot."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (tensor.detach().cpu() * std + mean).clamp(0, 1)


def to_displayable(tensor: torch.Tensor):
    """De-normalised ``(H, W, 3)`` NumPy array for ``imshow``."""
    return denormalize(tensor).permute(1, 2, 0).numpy()


def load_image(path, image_size: int = config.IMAGE_SIZE,
               patch: int = 14) -> torch.Tensor:
    """Load an image file straight to a preprocessed tensor."""
    return preprocess(Image.open(Path(path)), image_size, patch)
