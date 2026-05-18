# Product Requirements Document — DINOv3: An Extensive, Visual Learning Resource

> **Status:** Draft — awaiting user approval
> **Author:** Claude Code
> **Date:** 2026-05-18
> **Project:** `AI_RamatGan/L59/DinoV3`

---

## 1. Purpose & Vision

This project is a **teaching artifact**. Its goal is not to ship a product but to let a
reader *understand, in extensive detail, how DINOv3 works* — the self-supervised vision
foundation model released by Meta AI in August 2025.

The deliverable is a small, well-documented Python codebase plus a set of Markdown
documents. Running the code produces **~30+ figures** that are then embedded in the
documentation. Every abstract concept is paired with a concrete, generated picture.

The guiding principle (from `CLAUDE.md`): **teach and help the reader understand, not
just complete a task.**

### 1.1 Who is this for?

A reader who knows basic deep learning (what a neural network and a loss function are)
but has *not* studied vision transformers or self-supervised learning. By the end they
should be able to explain: what problem DINOv3 solves, how a Vision Transformer turns an
image into tokens, how the student/teacher self-distillation loop learns *without
labels*, and what DINOv3 specifically changed relative to DINOv2.

---

## 2. Background — What is DINOv3?

### 2.1 The problem: labels are expensive

Classic image models (e.g. a ResNet trained on ImageNet) learn from **(image, label)**
pairs. Labels are costly, slow to collect, biased toward the label vocabulary, and cap
the dataset size. A model trained to predict "1 of 1000 classes" also tends to *throw
away* everything not needed for that decision.

**Self-supervised learning (SSL)** removes labels entirely. The model learns from the
*structure of images themselves*. The result is a **foundation model**: a frozen
feature extractor whose outputs are reused, untouched, for many downstream tasks
(classification, segmentation, depth, retrieval).

### 2.2 The DINO family

| Model | Year | One-line idea |
|-------|------|---------------|
| DINO | 2021 | Self-**di**stillation with **no** labels: a student network is trained to match a teacher that is an exponential moving average (EMA) of the student. Attention maps emerge that segment objects for free. |
| DINOv2 | 2023 | Scale it up: curated 142M-image dataset, add the **iBOT** masked-patch objective, **KoLeo** regularization, **register tokens**. Produces strong *dense* (per-patch) features. |
| **DINOv3** | 2025 | Scale it *much* further (7B-parameter ViT, 1.7B images) and fix a newly-identified failure mode of long training with **Gram anchoring**. Adds RoPE positional encoding, high-resolution adaptation, a distilled model family, and text alignment. |

### 2.3 Why DINOv3 matters

DINOv3 is the first SSL model to **beat weakly-supervised models on dense prediction
tasks without any fine-tuning**. A single frozen DINOv3 backbone feeds lightweight heads
for segmentation, depth, 3D correspondence, and more. Understanding it is understanding
the current state of the art in general-purpose visual representations.

### 2.4 The core mechanisms (taught in full in `docs/THEORY.md`)

1. **Vision Transformer (ViT) backbone.** The image is cut into a grid of fixed-size
   patches; each patch is linearly embedded into a vector ("token"). A learned **CLS
   token** is prepended to summarize the whole image; **register tokens** are extra
   scratch tokens. Tokens pass through transformer blocks (multi-head self-attention +
   MLP). Output: one vector per patch + one CLS vector.

2. **Self-distillation (the DINO loss).** Two networks, *student* and *teacher*, share
   the architecture. The teacher's weights are an **EMA** of the student's — it is never
   trained by gradient descent. Both see different **crops** of the same image
   (multi-crop: 2 large "global" crops + several small "local" crops). The student is
   trained so its output distribution matches the teacher's. Collapse (all outputs
   identical) is prevented by **centering** + **sharpening** the teacher.

3. **iBOT masked-patch loss.** Some student patches are masked; the student must predict
   the teacher's patch outputs at those positions. This is what makes *dense* features
   sharp.

4. **KoLeo regularization.** Encourages embeddings in a batch to spread out, improving
   nearest-neighbor retrieval.

5. **Gram anchoring (DINOv3's headline contribution).** During very long training,
   global metrics keep improving but *dense* patch features slowly degrade — similarity
   maps get noisy. DINOv3 keeps a frozen early checkpoint (the "Gram teacher") and adds
   a loss that aligns the **Gram matrix** (the patch-to-patch similarity structure) of
   the current model to the Gram teacher's. This regularizes only the *relational*
   structure of features, restoring clean dense maps.

---

## 3. Goals & Non-Goals

### 3.1 Goals

- **G1** — Explain DINOv3 end to end, concisely and coherently, in `docs/THEORY.md`
  and a condensed `README.md`.
- **G2** — Generate **~30+ figures** (architecture diagrams + real model outputs) and
  embed every one of them in the documentation with a caption explaining the takeaway.
- **G3** — Provide **worked examples** on real images: PCA feature maps, attention
  maps, patch similarity, dense correspondence, retrieval.
- **G4** — Include a **runnable mini self-distillation training demo** so the reader
  watches the learning mechanism work live (loss curve, EMA schedule, k-NN accuracy
  rising, features emerging from noise).
- **G5** — Quantitatively evaluate frozen features: k-NN classification, linear
  probing (with **confusion matrix** and per-class metrics), model-size comparison.
- **G6** — Be fully reproducible: `uv`-managed environment, fixed seeds, every result
  saved to a file.

### 3.2 Non-Goals

- **NG1** — Not reproducing DINOv3 pretraining at scale (7B params / 1.7B images is
  infeasible and unnecessary for teaching).
- **NG2** — Not fine-tuning or building production downstream heads.
- **NG3** — Not a from-scratch reimplementation of the official DINOv2/DINOv3
  codebase; we *use* pretrained backbones for visualization.

---

## 4. Scope & Approach

### 4.1 Which model do we actually run?

DINOv3 weights are **gated** on Hugging Face (require accepting Meta's license and an
authentication token). To keep the project runnable for everyone:

- **Live visualizations run on DINOv2**, which is openly downloadable via `torch.hub`
  (`facebookresearch/dinov2`) and is *mechanically near-identical* — same ViT backbone,
  same DINO + iBOT + KoLeo training, and it also offers **register-token variants**.
- An **optional DINOv3 code path** (`src/dinov3_loader.py`, via Hugging Face
  `transformers`) is wired in. If the user is authenticated it is used for side-by-side
  comparison figures; if not, the project **degrades gracefully** and simply notes it.
- The **documentation explains DINOv3 specifically** — Gram anchoring, RoPE, scaling,
  the distilled family, `dino.txt` text alignment — even where the figures use DINOv2.
  The `docs/DINOV3_VS_DINOV2.md` file is dedicated to the differences.

This is a deliberate, disclosed trade-off: maximum reproducibility, with full DINOv3
conceptual coverage.

### 4.2 Models compared

ViT-Small, ViT-Base, ViT-Large (DINOv2, patch size 14), plus their register-token
variants. Target hardware is a single **NVIDIA CUDA GPU**.

### 4.3 The mini-training demo

A small ViT (~3–6M parameters) is trained from scratch with a simplified DINO loss on
the **STL-10 unlabeled split** (100k images at 96×96 — designed for SSL research). The
labeled split provides a k-NN probe to monitor progress. Target runtime: well under
~30 minutes on one GPU. This demo is *illustrative*, not state-of-the-art.

---

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | Load DINOv2 ViT-S/B/L backbones (and `_reg` variants) via `torch.hub`. |
| FR2 | Optionally load a DINOv3 backbone via `transformers`; skip cleanly if gated/unauthenticated. |
| FR3 | Extract CLS tokens, per-patch token grids, and intermediate-layer features. |
| FR4 | Capture last-block self-attention maps (via forward hook — no native accessor). |
| FR5 | Compute PCA / t-SNE / UMAP projections of features. |
| FR6 | Compute patch-to-patch cosine similarity and cross-image dense correspondence. |
| FR7 | Evaluate frozen features with k-NN and a linear probe; emit a confusion matrix. |
| FR8 | Render architecture & training-mechanism diagrams with matplotlib. |
| FR9 | Train the mini DINO demo and log loss, EMA momentum, and k-NN accuracy per epoch. |
| FR10 | Save every figure as PNG + a sidecar JSON of its parameters/metrics into `assets/`. |
| FR11 | A verification script confirms every expected figure exists. |

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | All Python runs inside a `uv`-managed virtual environment. |
| NFR2 | **Max 150 lines per source file** — larger modules are split. |
| NFR3 | Model forward passes use `torch.compile()` for performance, with a graceful fallback if compilation fails on the host. (This is our reading of CLAUDE.md's "code must be compiled" requirement for a PyTorch project.) |
| NFR4 | Fixed random seeds; results are reproducible run-to-run (modulo documented t-SNE/UMAP nondeterminism). |
| NFR5 | Heavy, educational code comments — the source is itself a teaching text. |
| NFR6 | Device auto-detection; CPU fallback works (slower) if no GPU is present. |
| NFR7 | Every result is persisted to disk for later inspection. |

---

## 7. Deliverables

```
README.md                  Condensed, figure-rich walkthrough — the front door.
docs/PRD.md                This document.
docs/PLANNING.md           Architecture, data flow, design decisions.
docs/TASKS.md              Implementation checklist (ticked off during the build).
docs/THEORY.md             The full deep-dive explanation of DINOv3.
docs/DINOV3_VS_DINOV2.md   Focused comparison + the Gram-anchoring explainer.
docs/VISUALIZATION_GUIDE.md  Catalog: every figure — what it shows, how, the takeaway.
src/ + src/training/       ~25 small Python modules (<=150 lines each).
scripts/00..13             Entry points; each generates one group of figures.
assets/                    Generated PNG + JSON figures (gitignored).
pyproject.toml / uv.lock   Reproducible environment.
```

---

## 8. The Visualization Catalog (G2)

Grouped; full detail in `docs/VISUALIZATION_GUIDE.md`. Target ≥30 figures.

1. **Concept diagrams** — ViT pipeline; patchify; CLS/register token layout; RoPE vs
   learned positional embeddings; student/teacher + EMA loop; multi-crop;
   DINO + iBOT loss; Gram-anchoring degradation curve.
2. **Dense patch features** — PCA-RGB feature maps; foreground/background mask from the
   first PCA component (segmentation for free); layer-by-layer PCA.
3. **Attention** — per-head CLS attention; attention-rollout saliency;
   **register vs no-register** artifact comparison.
4. **Similarity & correspondence** — click-a-patch cosine-similarity heatmaps;
   cross-image patch matching with drawn correspondence lines.
5. **Global embeddings & evaluation** — t-SNE & UMAP of CLS embeddings colored by class;
   k-NN accuracy by model size; linear-probe accuracy, per-class report, and
   **confusion matrix**; accuracy/dimension/latency comparison; retrieval gallery.
6. **Resolution behavior** — feature maps and accuracy across input resolutions
   (motivates DINOv3's high-resolution adaptation).
7. **Mini-training** — loss curve; EMA momentum schedule; teacher/student agreement;
   k-NN accuracy over epochs; **feature-PCA emergence** (one image at epoch 0 / mid /
   final, structure appearing out of noise).

---

## 9. Success Criteria & Acceptance

The project is **done** when:

- **AC1** — `uv sync` succeeds and `scripts/00_setup_check.py` confirms the environment,
  GPU, and a working DINOv2 forward pass.
- **AC2** — `scripts/01`–`12` all run to completion and write their figures to `assets/`.
- **AC3** — `scripts/13_build_readme_assets.py` reports **0 missing figures**; every
  figure referenced by the docs exists.
- **AC4** — Mini-training: smoothed loss **decreases**; k-NN accuracy ends **clearly
  above the 10% chance level**; the per-epoch feature-PCA panel shows visibly emerging
  structure.
- **AC5** — The linear probe produces an accuracy number, a per-class report, and a
  saved confusion-matrix figure.
- **AC6** — Code audit: **no source file exceeds 150 lines**; `torch.compile()` is
  applied; seeds are fixed.
- **AC7** — `README.md` reads as a coherent, concise walkthrough with every figure
  embedded and captioned.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| DINOv2 exposes **no attention accessor** | `src/attention_hook.py` captures attention via a forward hook; built and tested first as the linchpin module. |
| DINOv3 weights gated → can't download | Project runs entirely on DINOv2; DINOv3 path is optional and skipped cleanly. Disclosed in §4.1. |
| `torch.compile()` fails on the host | Wrapped in try/except; falls back to eager mode with a logged warning. |
| Mini-training doesn't converge in the time budget | Small model, capped subset, simplified loss, tuned schedule; acceptance is "above chance," not SOTA. |
| GPU memory limits ViT-L | Batch sizes and image counts are configurable in `src/config.py`. |
| t-SNE/UMAP nondeterminism | Seeded; residual nondeterminism documented. |

---

## 11. Out of Scope for v1 (possible follow-ups)

Downstream segmentation/depth heads; the `dino.txt` zero-shot text path; ConvNeXt
DINOv3 variants; the 7B model; video features.

---

## 12. Approval

Per `CLAUDE.md`, **no implementation code is written until this PRD is approved**, after
which `docs/TASKS.md` and `docs/PLANNING.md` are written and also approved.

**→ Please review and approve this PRD, or request changes.**
