"""
train_dino.py -- The mini DINO self-distillation training loop.

Ties the pieces together: a TinyViT + DINO head as the student, an EMA
copy as the teacher, multi-crop data, the DINO loss, and the k-NN
monitor. It trains on the STL-10 *unlabelled* split -- no labels at all --
and logs everything script 12 needs to visualise the run.
"""
from __future__ import annotations

import copy
import json
import time

import torch
from torch.utils.data import DataLoader

import config
import paths
from datasets import stl10
from device import device_info, get_device, maybe_compile
from dino_head import DINOHead
from dino_loss import DINOLoss
from ema import cosine_momentum, ema_update
from knn_monitor import build_probe_loaders, knn_probe
from multicrop import CropDataset, MultiCrop, collate_crops
from tiny_vit import TinyViT


def build_model(cfg, device):
    """A TinyViT backbone followed by a DINO projection head."""
    backbone = TinyViT(cfg.patch_size, cfg.embed_dim, cfg.depth, cfg.num_heads,
                       base_grid=cfg.global_crop // cfg.patch_size)
    head = DINOHead(cfg.embed_dim, cfg.out_dim)
    return torch.nn.Sequential(backbone, head).to(device)


def teacher_temperature(epoch: int, cfg) -> float:
    """Teacher temperature, linearly warmed over the first few epochs."""
    warm = cfg.teacher_temp_warmup_epochs
    if epoch >= warm:
        return cfg.teacher_temp
    span = cfg.teacher_temp - cfg.warmup_teacher_temp
    return cfg.warmup_teacher_temp + span * epoch / max(warm, 1)


def run_training(cfg=config.TRAIN):
    """Train the student/teacher pair; return the logged history dict."""
    config.set_seeds()
    paths.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device_info(device)}")

    # Take the raw STL-10 unlabelled image array and keep only the subset
    # we train on. Slicing here also keeps the dataset object small, so the
    # "spawn" workers need not pickle the full 2.6 GB array.
    raw_images = stl10("unlabeled").data
    n_images = min(cfg.subset_size, len(raw_images))
    dataset = CropDataset(raw_images[:n_images].copy(), MultiCrop(cfg))
    # num_workers=0: augmentation runs in this process. Subprocess workers
    # proved unstable here (fork and spawn both crashed); single-process
    # loading is a touch slower but completely reliable.
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_crops,
                        drop_last=True)

    student = build_model(cfg, device)
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad_(False)
    student_compiled = maybe_compile(student)        # torch.compile (NFR3)

    criterion = DINOLoss(cfg.out_dim, cfg.student_temp,
                         cfg.center_momentum).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.base_lr,
                                  weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           total_steps)
    probe_loaders = build_probe_loaders()

    history = {"loss": [], "knn": [], "knn_epoch": [], "momentum": []}
    step, start = 0, time.time()
    for epoch in range(cfg.epochs):
        t_temp = teacher_temperature(epoch, cfg)
        epoch_loss = 0.0
        for crops in loader:
            crops = [c.to(device, non_blocking=True) for c in crops]
            with torch.no_grad():
                teacher_out = [teacher(g) for g in crops[:2]]
            student_out = [student_compiled(c) for c in crops]
            loss = criterion(student_out, teacher_out, t_temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            momentum = cosine_momentum(step, total_steps,
                                       cfg.ema_base, cfg.ema_final)
            ema_update(student, teacher, momentum)
            history["momentum"].append(momentum)
            epoch_loss += loss.item()
            step += 1

        history["loss"].append(epoch_loss / len(loader))
        if epoch in cfg.pca_track_epochs:
            torch.save(teacher[0].state_dict(),
                       paths.CHECKPOINTS / f"teacher_epoch{epoch}.pt")
        if epoch % cfg.knn_every == 0 or epoch == cfg.epochs - 1:
            accuracy = knn_probe(teacher[0], probe_loaders, device, cfg.knn_k)
            history["knn"].append(accuracy)
            history["knn_epoch"].append(epoch)
            print(f"  epoch {epoch:2d}  loss {history['loss'][-1]:.4f}  "
                  f"knn {accuracy:.3f}  (momentum {momentum:.4f})")
        else:
            print(f"  epoch {epoch:2d}  loss {history['loss'][-1]:.4f}")

    history["minutes"] = (time.time() - start) / 60.0
    torch.save(teacher[0].state_dict(), paths.CHECKPOINTS / "teacher_final.pt")
    (paths.CHECKPOINTS / "history.json").write_text(json.dumps(history))
    return history
