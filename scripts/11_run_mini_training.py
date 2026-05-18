"""
11_run_mini_training.py -- Train a DINO model from scratch.

This runs the real self-distillation loop on the STL-10 unlabelled split
(no labels). It is small on purpose -- a few-million-parameter ViT, a
capped subset, a short schedule -- so the whole run finishes in minutes
on one GPU while still showing the mechanism work: the loss falls and the
(monitoring-only) k-NN accuracy climbs well above chance.

Checkpoints, the loss/k-NN history and per-epoch snapshots are written to
checkpoints/ for script 12 to visualise.

    uv run python scripts/11_run_mini_training.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "training"))
warnings.filterwarnings("ignore")

import config  # noqa: E402
import paths  # noqa: E402
from train_dino import run_training  # noqa: E402


def main() -> None:
    paths.ensure_dirs()
    cfg = config.TRAIN
    print("Mini DINO training -- STL-10 unlabelled split (no labels used)")
    print(f"  {cfg.epochs} epochs | {cfg.subset_size} images | "
          f"batch {cfg.batch_size} | 2 global + {cfg.n_local_crops} local crops")
    print(f"  TinyViT: dim {cfg.embed_dim}, depth {cfg.depth}, "
          f"patch {cfg.patch_size}\n")

    history = run_training(cfg)

    chance = 1.0 / cfg.num_classes
    print(f"\nfinished in {history['minutes']:.1f} min")
    print(f"  loss: {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")
    print(f"  k-NN accuracy: {history['knn'][0]:.3f} -> {history['knn'][-1]:.3f}"
          f"  (chance = {chance:.2f})")
    print(f"  checkpoints + history -> {paths.CHECKPOINTS}")


if __name__ == "__main__":
    main()
