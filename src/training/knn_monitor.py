"""
knn_monitor.py -- A periodic k-NN probe to watch features improve.

DINO uses no labels, so the loss going down does not, by itself, prove
the features became *useful*. Every few epochs we therefore borrow the
labelled STL-10 splits, extract the teacher's CLS features and run a
quick k-NN classification. A rising k-NN accuracy is the real evidence
that the representation is learning structure.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config
from datasets import stl10
from image_io import preprocess
from knn_eval import knn_accuracy


def build_probe_loaders(n_train: int = 2000, n_test: int = 1000):
    """Small labelled STL-10 train/test loaders for the k-NN monitor."""
    def transform(image):
        return preprocess(image, 96, config.TRAIN.patch_size)

    train = Subset(stl10("train", transform), range(n_train))
    test = Subset(stl10("test", transform), range(n_test))
    return (DataLoader(train, batch_size=128),
            DataLoader(test, batch_size=128))


@torch.no_grad()
def _cls_features(model, loader, device):
    """L2-normalised CLS features + labels for everything in `loader`."""
    feats, labels = [], []
    for images, targets in loader:
        cls = model(images.to(device))
        feats.append(F.normalize(cls, dim=-1).cpu())
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_probe(model, loaders, device, k: int = 20) -> float:
    """k-NN accuracy of `model`'s CLS features on labelled STL-10."""
    was_training = model.training
    model.eval()
    train_loader, test_loader = loaders
    train_x, train_y = _cls_features(model, train_loader, device)
    test_x, test_y = _cls_features(model, test_loader, device)
    if was_training:
        model.train()
    return knn_accuracy(train_x, train_y, test_x, test_y, k,
                        config.TRAIN.num_classes)
