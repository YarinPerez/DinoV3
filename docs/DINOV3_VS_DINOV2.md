# DINOv3 vs DINOv2 — What Actually Changed

DINOv2 (2023) and DINOv3 (2025) share one recipe: a Vision Transformer
trained by **self-distillation** (DINO loss) + **masked image modelling**
(iBOT) + **KoLeo** regularisation. DINOv3 keeps all of that and changes
four things. This file explains each, and goes deep on the most important
one — **Gram anchoring**.

> The live figures in this project run on **DINOv2** (it is openly
> downloadable; DINOv3's weights are gated). DINOv2 is mechanically
> near-identical, so every mechanism you see in the figures is the same
> one DINOv3 uses. What this file adds is the DINOv3-specific *deltas*.

---

## 1. Side-by-side

| Aspect | DINOv2 (2023) | DINOv3 (2025) |
|---|---|---|
| Largest model | ViT-g, ~1.1 B params | Custom ViT, **~7 B params** |
| Training images | ~142 M (curated, LVD-142M) | **~1.7 B** (curated, LVD-1689M) |
| Positional encoding | Learned, interpolated | **Axial RoPE** + position jittering |
| Register tokens | Added mid-life (DINOv2-reg) | Built in (4 tokens) |
| Patch size (ViT) | 14 | 16 |
| Dense-feature decay on long runs | Present, unaddressed | **Fixed by Gram anchoring** |
| High-resolution use | Limited | Post-hoc **high-res adaptation** |
| Model family | A few ViT sizes | ViT-S/B/L/H **+ ConvNeXt**, all distilled from the 7 B teacher |
| Text / zero-shot | — | `dino.txt` aligns a text encoder |

The objective changes are **scale**, **Gram anchoring**, **RoPE +
high-resolution adaptation**, and the **distilled model family**.

---

## 2. Scale

DINOv3 trains a ~7-billion-parameter ViT on ~1.7 billion curated images.
Scale is not just "more": the larger model is the **teacher** from which
every shipped model is distilled (section 5). The point of building a 7 B
model is less to deploy it and more to *distil* it.

Curation matters as much as quantity: the LVD-1689M dataset is built by
retrieval and clustering so the billion-plus images are *diverse and
balanced*, not just numerous.

---

## 3. Gram anchoring — the headline contribution

### 3.1 The problem: dense features decay

Train any DINO-style model for a very long time and a strange thing
happens. **Image-level (CLS) metrics keep improving** — k-NN, linear
probe, classification all go up. But **dense, per-patch features quietly
get worse**: the patch-similarity maps that were once crisp grow noisy,
and dense tasks (segmentation, depth) start to suffer.

The cause: with an enormous number of training steps, the patch
representations slowly drift. Nothing in the DINO/iBOT loss directly
*pins down* the dense feature geometry, so it is free to degrade even
while the global objective is satisfied. `assets/diagrams/08_gram_anchoring.png`
panel A sketches this: the blue (global) curve rises and stays up; the
red (dense, un-anchored) curve rises then **falls**.

### 3.2 The idea: anchor the *relational* structure

DINOv3's fix, **Gram anchoring**:

1. Keep a **frozen snapshot** of the model from *earlier* in training, at
   a point when its dense features were still clean. Call it the
   **Gram teacher**.
2. For a batch of patches, compute the **Gram matrix** — the full matrix
   of patch-to-patch similarities (`features @ featuresᵀ`). This matrix
   captures the *relational* structure: which patches are like which.
3. Add a loss term that pulls the **current model's Gram matrix** towards
   the **Gram teacher's Gram matrix**.

The Gram matrix is exactly what `04_patch_similarity.py` visualises (the
similarity matrix). Gram anchoring says: *the structure of that matrix
must not drift away from the clean early version*.

### 3.3 Why anchor the Gram matrix and not the features?

This is the subtle, clever part. Why not just keep the dense features
themselves close to the early checkpoint?

Because that would **freeze the features** and stop them improving. The
Gram matrix only encodes **relative** structure — how patches relate to
*each other* — and is invariant to a global rotation or rescaling of the
feature space. So anchoring the Gram matrix:

- **pins down** the relational geometry (clean, sharp dense features),
- **leaves free** the absolute feature directions, which can keep
  improving with the global objective.

It regularises *the part that was degrading* and only that part. The
result (panel A, green dashed curve): dense quality keeps pace with
global quality instead of collapsing.

### 3.4 The pay-off

Gram anchoring is the main reason DINOv3 became the first self-supervised
model to **beat weakly-supervised models on dense tasks without
fine-tuning**. Crisp patch features are what dense prediction needs, and
Gram anchoring is what keeps them crisp through a billion-image,
long-schedule training run.

---

## 4. RoPE and high-resolution adaptation

DINOv2 uses **learned positional embeddings**: one trained vector per
grid cell. They are tied to the grid size seen in training; other
resolutions require interpolating that grid.

DINOv3 uses **axial RoPE** (rotary position embeddings): query and key
vectors are *rotated* by an angle set by position, so attention scores
depend on **relative** position (`assets/diagrams/04_positional_encoding.png`).
During training the position scale is **jittered** randomly. Together,
RoPE + jittering make the model robust to resolution changes — which lets
DINOv3 be **adapted, after pretraining, to high resolution**. High-res
inputs mean a finer patch grid and sharper dense predictions
(`08_resolution_sweep.py` shows the grid getting finer with resolution).

---

## 5. A distilled model family

A 7 B-parameter model is too large for most uses. DINOv3 therefore
**distils** the 7 B teacher into a family of smaller models — ViT-S, B, L,
H, and **ConvNeXt** variants. Distillation trains a small student to
reproduce the big teacher's outputs, so the small models inherit much of
the 7 B model's quality at a fraction of the compute. (Note: this is
*model→model* distillation, distinct from the *self*-distillation of the
DINO loss.) DINOv3 also trains a **text encoder aligned** to the frozen
vision features (`dino.txt`), enabling open-vocabulary and zero-shot use.

`09_model_comparison.py` shows the size/speed/quality trade-off across
ViT-S/B/L — the practical reason a model family exists.

---

## 6. What carries over unchanged

Everything in `THEORY.md` sections 2–4 — the ViT backbone, the
student/teacher EMA loop, multi-crop, the DINO loss with centering and
sharpening, the iBOT masked-patch loss, KoLeo regularisation — is shared
by DINOv2 and DINOv3. That shared core is what the figures in this
project demonstrate (on DINOv2 weights); this file is the list of what
DINOv3 adds on top.
