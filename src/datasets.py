"""
datasets.py -- Dataset access for the project.

Two torchvision datasets are used:
  * Oxford-IIIT Pet -- real, high-resolution cat/dog photos; the sample
    images for every single-image visualization (PCA maps, attention,
    similarity, correspondence, retrieval).
  * STL-10 -- 10-class 96px photos; its labelled splits drive the k-NN,
    linear-probe and t-SNE evaluations, and its 100k-image unlabelled
    split powers the from-scratch training demo (see src/training/).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import config
import paths
from image_io import eval_transform, preprocess

STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]

SampleList = List[Tuple[str, Image.Image, torch.Tensor]]


def stl10(split: str, transform=None):
    """torchvision STL-10 dataset ('train' | 'test' | 'unlabeled')."""
    return datasets.STL10(root=str(paths.DATA), split=split,
                          transform=transform, download=True)


def oxford_pet(split: str = "test", transform=None):
    """torchvision Oxford-IIIT Pet dataset ('trainval' | 'test')."""
    return datasets.OxfordIIITPet(root=str(paths.DATA), split=split,
                                  transform=transform, download=True)


def _curate(items, patch: int) -> SampleList:
    """Preprocess a list of (name, PIL) pairs into the sample-list format."""
    return [(name, img, preprocess(img, config.IMAGE_SIZE, patch))
            for name, img in items]


def sample_images(n: int = 8, patch: int = 14) -> SampleList:
    """
    Up to `n` curated sample images as (name, PIL, tensor) tuples.

    Drawn from Oxford-IIIT Pet (high-resolution cat/dog photos); falls
    back to STL-10 if Oxford-IIIT Pet cannot be downloaded.
    """
    try:
        ds = oxford_pet("test")
        idx = torch.linspace(0, len(ds) - 1, n).round().long().tolist()
        items = [(f"pet_{j:02d}", ds[i][0]) for j, i in enumerate(idx)]
    except Exception:
        ds = stl10("test")
        idx = torch.linspace(0, len(ds) - 1, n).round().long().tolist()
        items = [(f"{STL10_CLASSES[ds[i][1]]}_{j:02d}", ds[i][0])
                 for j, i in enumerate(idx)]
    return _curate(items, patch)


def similar_pair(patch: int = 14) -> SampleList:
    """
    Two different photos that share a category -- the input to the dense
    correspondence demo (script 05).
    """
    try:
        ds = oxford_pet("test")
        label0 = ds[0][1]
        second = next(i for i in range(1, min(len(ds), 300))
                      if ds[i][1] == label0)
    except Exception:
        ds = stl10("test")
        label0 = ds[0][1]
        second = next(i for i in range(1, len(ds)) if ds[i][1] == label0)
    items = [("pair_a", ds[0][0]), ("pair_b", ds[second][0])]
    return _curate(items, patch)


def eval_loaders(patch: int = 14):
    """
    DataLoaders over the STL-10 train/test splits, preprocessed for the
    backbone. Feeds k-NN, the linear probe, t-SNE and retrieval.
    """
    tf = eval_transform(config.EVAL.image_size, patch)
    train = stl10("train", tf)
    test = Subset(stl10("test", tf), range(config.EVAL.test_samples))
    # num_workers=0: the lambda transform is not picklable, and STL-10's
    # tiny 96px images decode fast enough single-process.
    train_loader = DataLoader(train, batch_size=64, num_workers=0)
    test_loader = DataLoader(test, batch_size=64, num_workers=0)
    return train_loader, test_loader
