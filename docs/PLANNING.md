# PLANNING — Architecture & Design

> **Status:** Draft — awaiting user approval
> **Companion to:** `docs/PRD.md`, `docs/TASKS.md`
> **Date:** 2026-05-18

This document describes *how* the DINOv3 learning resource is built: the system
architecture, the data flow, the module breakdown, the key design decisions and their
trade-offs, and how progress is tracked.

---

## 1. Architecture Overview

The project is a **pipeline of independent stages**. Each stage is a numbered script in
`scripts/` that imports small, single-purpose modules from `src/` and writes figures to
`assets/`. No stage depends on another stage's *in-memory* state — only on files. This
makes every stage independently runnable, re-runnable, and debuggable.

```
                         ┌──────────────────────────────────────┐
                         │            src/  (library)            │
   pretrained            │  config · paths · device              │
   DINOv2  ──torch.hub──▶ │  backbone_loader · attention_hook     │
   (DINOv3 optional, HF)  │  features · image_io · datasets       │ ──┐
                          │  decomposition · similarity           │   │
   STL-10 / CIFAR-10 ───▶ │  knn_eval · linear_probe · plot_style │   │
   (torchvision)          │  diagrams_arch · diagrams_train       │   │
                          │  training/ (tiny_vit, dino_loss, ...) │   │
                          └──────────────────────────────────────┘   │
                                                                      ▼
   scripts/00..13  ──imports src/──▶  compute  ──▶  assets/<group>/*.png + *.json
                                                                      │
                                                                      ▼
                          docs/*.md  +  README.md   embed figures from assets/
```

**Layering rule:** `scripts/` may import `src/`; `src/` modules may import each other
but never `scripts/`. `config.py` is imported by everything and imports nothing from the
project. This keeps the dependency graph acyclic.

---

## 2. Data Flow

A typical visualization script follows the same five steps:

1. **Configure** — read model name, device, image set, and hyperparameters from
   `src/config.py`.
2. **Load** — `backbone_loader.py` fetches a pretrained backbone (cached after first
   download); `image_io.py` / `datasets.py` load and normalize inputs.
3. **Extract** — `features.py` (and `attention_hook.py`) run the forward pass and return
   CLS tokens, patch-token grids, intermediate layers, or attention maps.
4. **Analyze** — `decomposition.py`, `similarity.py`, `knn_eval.py`, or
   `linear_probe.py` turn raw features into something interpretable.
5. **Render & persist** — `plot_style.py` saves a PNG plus a sidecar JSON recording the
   parameters and any metrics, into the correct `assets/<group>/` folder.

The **mini-training** stage is the one exception — it is a real training loop
(`src/training/train_dino.py`) that periodically checkpoints and logs, then a separate
script (`scripts/12`) reads those logs/checkpoints to render its figures. Training and
plotting are deliberately decoupled so plots can be regenerated without retraining.

---

## 3. Module Breakdown

All files obey the **150-line hard limit**. Approximate sizes in parentheses.

### 3.1 `src/` — foundation

| Module | Responsibility |
|--------|----------------|
| `config.py` (~90) | Single source of truth: dataclasses for the model list, device prefs, image/patch sizes, training hyperparameters, RNG seeds, and asset paths. |
| `paths.py` (~40) | Resolves and creates `assets/`, `data/`, `assets/<group>/` directories. |
| `device.py` (~50) | Picks CUDA/MPS/CPU; sets dtype/AMP; wraps `torch.compile()` with a try/except fallback to eager mode. |
| `plot_style.py` (~70) | Shared matplotlib theme, colormaps, and the `save_figure()` helper that writes PNG + JSON sidecar. |

### 3.2 `src/` — model access & feature extraction

| Module | Responsibility |
|--------|----------------|
| `backbone_loader.py` (~120) | Loads DINOv2 ViT-S/B/L (+ `_reg`) via `torch.hub`; returns the model plus metadata (embed dim, patch size, depth, #heads, #registers). |
| `dinov3_loader.py` (~110) | Optional DINOv3 path via HF `transformers`; detects missing auth/gating and raises a clean, catchable `Dinov3Unavailable` so callers skip it. |
| `attention_hook.py` (~110) | **Linchpin.** Registers a forward hook on the last block's attention to capture the `softmax(QK^T)` tensor `(heads, tokens, tokens)`. Built and tested first. |
| `features.py` (~140) | Unified extraction: CLS token, patch-token grid reshaped to `(H/p, W/p, D)`, and intermediate layers; optional L2-normalization. |
| `image_io.py` (~90) | Loads/resizes/normalizes images (ImageNet stats); guarantees dimensions divisible by patch size; provides the sample-image set. |
| `datasets.py` (~130) | `torchvision` wrappers for STL-10 and CIFAR-10 with label metadata, used by k-NN / probe / t-SNE / mini-training. |

### 3.3 `src/` — analysis

| Module | Responsibility |
|--------|----------------|
| `decomposition.py` (~120) | PCA (3-component → RGB map; 1st component → fg/bg mask), t-SNE, UMAP. |
| `similarity.py` (~110) | Patch-patch cosine similarity; click-a-patch heatmap; cross-image dense correspondence. |
| `knn_eval.py` (~120) | k-NN classification accuracy on frozen CLS features. |
| `linear_probe.py` (~140) | Trains one linear layer on frozen features; returns accuracy, per-class report, confusion matrix; writes a metrics JSON. |
| `diagrams_arch.py` (~140) | Matplotlib architecture diagrams: ViT pipeline, patchify, CLS/registers, RoPE. |
| `diagrams_train.py` (~140) | Matplotlib mechanism diagrams: EMA loop, multi-crop, DINO/iBOT loss, Gram anchoring. |

### 3.4 `src/training/` — mini DINO demo

| Module | Responsibility |
|--------|----------------|
| `tiny_vit.py` (~140) | Small ViT (patch embed, CLS token, transformer blocks) used as both student and teacher. |
| `dino_head.py` (~80) | Projection head: MLP + weight-normalized last layer. |
| `multicrop.py` (~120) | Multi-crop augmentation: 2 global + 6 local crops. |
| `dino_loss.py` (~120) | Simplified DINO loss: temperature softmax, EMA centering, cross-entropy across crop pairs; optional iBOT term. |
| `ema.py` (~60) | EMA teacher update and cosine momentum schedule. |
| `knn_monitor.py` (~90) | Periodic k-NN probe on a labeled STL-10 subset during training. |
| `train_dino.py` (~150) | Training loop: AdamW, cosine LR + warmup, logging, checkpointing. |

### 3.5 `scripts/` — entry points

`00_setup_check` · `01_concept_diagrams` · `02_patch_pca_maps` ·
`03_attention_maps` · `04_patch_similarity` · `05_dense_correspondence` ·
`06_embedding_projection` · `07_knn_and_probe` · `08_resolution_sweep` ·
`09_model_comparison` · `10_image_retrieval` · `11_run_mini_training` ·
`12_training_visuals` · `13_build_readme_assets`.

Each is a thin orchestrator (~80–130 lines): configure → call `src/` → save.

---

## 4. Key Design Decisions & Trade-offs

| # | Decision | Rationale / Trade-off |
|---|----------|-----------------------|
| D1 | **DINOv2 for figures, DINOv3 optional.** | DINOv3 weights are gated; DINOv2 is open and mechanically near-identical. Trade-off: figures aren't *literally* DINOv3, but the project runs for everyone. The DINOv3 path is wired in for authenticated users; docs carry the DINOv3-specific theory. |
| D2 | **File-based stage decoupling.** | Each script reads/writes files only. Slower than one big in-memory run, but every stage is independently runnable and debuggable — essential for a teaching repo. |
| D3 | **`torch.compile()` for "compilation."** | CLAUDE.md mandates compiled code. For PyTorch, `torch.compile()` is the idiomatic choice over Cython/Numba. Wrapped in try/except so a compile failure never breaks the run. |
| D4 | **STL-10 unlabeled split for mini-training.** | Designed for SSL; 96×96 makes multi-crop meaningful (CIFAR's 32×32 does not). ~2.5 GB download accepted by the user. |
| D5 | **Simplified DINO loss in the demo.** | Full DINOv2 loss (Sinkhorn, full iBOT, KoLeo, sharded heads) is too heavy to teach in ~150 lines. We implement centering+sharpening DINO loss + optional light iBOT; documented as a simplification. |
| D6 | **Attention via forward hook.** | DINOv2's hub model has no attention output. A hook on the last block is non-invasive and robust. Risk: internal module names — mitigated by building `attention_hook.py` first against a real model. |
| D7 | **PNG + JSON sidecars.** | Every figure ships with the exact parameters/metrics that produced it — reproducibility and inspectability per CLAUDE.md. |
| D8 | **Many small modules over few large ones.** | The 150-line limit forces single-responsibility files, which read as teaching units. Trade-off: more files to navigate — mitigated by this document and clear naming. |

---

## 5. Environment & Reproducibility

- **`uv`** manages the virtual environment; `pyproject.toml` declares dependencies and
  `uv.lock` pins exact versions. Python 3.11.
- Dependencies: `torch>=2.4`, `torchvision`, `transformers>=4.53`, `huggingface-hub`,
  `scikit-learn>=1.4`, `umap-learn>=0.5`, `matplotlib>=3.8`, `numpy>=1.26`, `pillow`,
  `tqdm`.
- A single `set_seeds()` in `config.py` seeds `random`, `numpy`, and `torch` (CPU+CUDA).
- t-SNE/UMAP carry residual nondeterminism even when seeded — documented where used.
- Downloaded weights/datasets live in `data/` and the `torch.hub` cache; `assets/` and
  `data/` are git-ignored.

---

## 6. Error Handling & Graceful Degradation

| Situation | Behavior |
|-----------|----------|
| No CUDA GPU | `device.py` falls back to CPU; small models still run (slower); a warning is logged. |
| `torch.compile()` fails | Caught; model runs eager; warning logged. |
| DINOv3 gated/unauthenticated | `dinov3_loader.py` raises `Dinov3Unavailable`; scripts catch it, log "DINOv3 path skipped," and continue with DINOv2. |
| Dataset download fails | Script exits with a clear message naming the dataset and target path. |
| GPU out of memory | Batch size / image count are config knobs; the README documents lowering them. |

---

## 7. Progress Tracking

`docs/TASKS.md` is the live checklist. Each task has a checkbox; tasks are marked
`[x]` **as they are completed during implementation**, and a one-line note may be added
(e.g. a measured accuracy). The same tasks are mirrored in the harness task list for
this session. A task is only ticked when its acceptance check passes — partial work
stays unticked.

---

## 8. Verification Strategy

End-to-end verification (detailed in `docs/TASKS.md`, Phase 8):

1. `uv sync` → `scripts/00_setup_check.py` green.
2. Run `scripts/01`–`12` in order; each exits 0 and writes its figures.
3. `scripts/13_build_readme_assets.py` reports **0 missing figures**.
4. Mini-training acceptance: smoothed loss decreases; final k-NN accuracy ≫ 10%;
   feature-PCA emergence panel shows visible structure.
5. Linear probe emits accuracy + per-class report + confusion-matrix PNG.
6. Code audit: `wc -l` over `src/` and `scripts/` → no file > 150 lines.
7. README renders with every figure embedded and captioned.

---

## 9. Approval

Per `CLAUDE.md`, **implementation begins only after this document and `docs/TASKS.md`
are approved.**

**→ Please review `docs/PLANNING.md` and `docs/TASKS.md` together and approve, or
request changes.**
