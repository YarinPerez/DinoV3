"""
config.py -- Central configuration: the single source of truth.

Every other module reads its settings from here, so experiments are
reproducible and easy to tweak in one place. This module deliberately
imports nothing else from the project (it sits at the bottom of the
dependency layering described in docs/PLANNING.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Pretrained backbone registry (DINOv2 -- served openly via torch.hub)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BackboneSpec:
    """Static description of one pretrained Vision Transformer backbone."""
    key: str            # short id used in our code and output filenames
    hub_name: str       # torch.hub entry-point name
    embed_dim: int      # width of every token vector
    depth: int          # number of transformer blocks
    num_heads: int      # attention heads per block
    patch_size: int = 14
    num_registers: int = 0
    label: str = ""     # human-readable name for plot titles


DINOV2_HUB_REPO: str = "facebookresearch/dinov2"

DINOV2_BACKBONES = {
    "vits14": BackboneSpec("vits14", "dinov2_vits14", 384, 12, 6, label="ViT-S/14"),
    "vitb14": BackboneSpec("vitb14", "dinov2_vitb14", 768, 12, 12, label="ViT-B/14"),
    "vitl14": BackboneSpec("vitl14", "dinov2_vitl14", 1024, 24, 16, label="ViT-L/14"),
    "vits14_reg": BackboneSpec("vits14_reg", "dinov2_vits14_reg", 384, 12, 6,
                               num_registers=4, label="ViT-S/14 +reg"),
    "vitb14_reg": BackboneSpec("vitb14_reg", "dinov2_vitb14_reg", 768, 12, 12,
                               num_registers=4, label="ViT-B/14 +reg"),
    "vitl14_reg": BackboneSpec("vitl14_reg", "dinov2_vitl14_reg", 1024, 24, 16,
                               num_registers=4, label="ViT-L/14 +reg"),
}

# Models used for the size-comparison figures, and the project default.
COMPARISON_KEYS: List[str] = ["vits14", "vitb14", "vitl14"]
DEFAULT_BACKBONE: str = "vitb14"

# Optional gated DINOv3 backbone (Hugging Face). Used only if authenticated.
DINOV3_HF_ID: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
IMAGE_SIZE: int = 448           # divisible by patch sizes 14 and 16
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Input sizes for the resolution-sweep experiment (all multiples of 14 and 16).
RESOLUTION_SWEEP: List[int] = [224, 336, 448, 560]


# --------------------------------------------------------------------------
# Evaluation (k-NN / linear probe / projections)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    dataset: str = "stl10"          # 10-class photos; labelled splits
    image_size: int = 224          # backbone input size for evaluation
    train_samples: int = 5000       # frozen-feature train set (STL-10 train)
    test_samples: int = 2000        # subset of the STL-10 test split
    knn_k: int = 20
    probe_epochs: int = 150
    probe_lr: float = 1e-3
    tsne_perplexity: float = 30.0


EVAL = EvalConfig()


# --------------------------------------------------------------------------
# Mini DINO self-distillation training demo
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "stl10"
    subset_size: int = 12000        # cap on unlabeled images (keeps it fast)
    # multi-crop geometry
    global_crop: int = 64
    local_crop: int = 32
    n_local_crops: int = 6
    # tiny ViT student / teacher
    patch_size: int = 8
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 6
    out_dim: int = 2048             # projection-head prototypes
    # optimisation
    batch_size: int = 128
    epochs: int = 30
    warmup_epochs: int = 5
    base_lr: float = 5e-4
    weight_decay: float = 0.04
    # DINO loss temperatures (teacher sharper than student; gentle warmup)
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.04
    teacher_temp_warmup_epochs: int = 10
    center_momentum: float = 0.9
    # EMA teacher momentum schedule (cosine ema_base -> ema_final)
    ema_base: float = 0.996
    ema_final: float = 1.0
    # monitoring
    knn_every: int = 4
    knn_k: int = 20
    pca_track_epochs: Tuple[int, ...] = (0, 8, 18, 29)
    num_classes: int = 10


TRAIN = TrainConfig()
