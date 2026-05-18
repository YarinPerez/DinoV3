"""
multicrop.py -- Multi-crop data augmentation for DINO.

DINO never compares an image to itself directly. It makes many augmented
*crops* of one image -- two large "global" crops and several small
"local" crops -- and trains the networks to produce a consistent output
across all of them. The teacher sees only the global crops; the student
sees all of them, so it must learn that a small local patch and the
whole object should map to the same place.

Implementation note: augmentation runs entirely on tensors via
torchvision.transforms.v2 (never on PIL images). PIL inside forked/spawned
data-loader workers proved unstable here; the tensor path is rock solid.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import config


def _augment(crop_size: int, scale):
    """A tensor-only random-crop + photometric augmentation pipeline."""
    return v2.Compose([
        v2.RandomResizedCrop(crop_size, scale=scale, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),       # uint8 -> float [0,1]
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class MultiCrop:
    """Turn one (C, H, W) uint8 image tensor into 2 global + N local crops."""

    def __init__(self, cfg=config.TRAIN):
        self.n_local = cfg.n_local_crops
        # global crops cover most of the image; local crops a small part
        self.global_tf = _augment(cfg.global_crop, (0.4, 1.0))
        self.local_tf = _augment(cfg.local_crop, (0.05, 0.4))

    def __call__(self, image):
        crops = [self.global_tf(image), self.global_tf(image)]
        crops += [self.local_tf(image) for _ in range(self.n_local)]
        return crops


class CropDataset(Dataset):
    """Wrap a uint8 image array; each item is a list of multi-crop tensors."""

    def __init__(self, images, multicrop: MultiCrop):
        self.images = images                # uint8 ndarray (N, C, H, W)
        self.multicrop = multicrop

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        return self.multicrop(image), -1     # label is unused (no labels!)


def collate_crops(batch):
    """
    Collate a batch of crop-lists into a list of batched tensors.

    Element i of the result is crop i stacked across the whole batch,
    shape ``(B, 3, size, size)``. Crops 0-1 are global, the rest local.
    """
    per_crop = zip(*[crops for crops, _label in batch])
    return [torch.stack(group) for group in per_crop]
