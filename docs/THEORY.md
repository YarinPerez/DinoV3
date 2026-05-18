# How DINOv3 Works — The Full Picture

> A deep-dive companion to the figures in `assets/`. Read it top to bottom:
> each idea builds on the previous one. The condensed version is in
> `../README.md`; the figure catalogue is in `VISUALIZATION_GUIDE.md`.

DINOv3 (Meta AI, August 2025) is a **self-supervised vision foundation
model**. This document explains, in order: *why* such a model exists, the
*Vision Transformer* it is built on, the *self-distillation* training that
needs no labels, the *extra objectives* that sharpen its features, the
*specific innovations* DINOv3 adds over DINOv2, and *how the features are
used*.

---

## 1. Why self-supervised learning?

A traditional image classifier learns from **(image, label)** pairs. That
design has three hard limits:

1. **Labels are expensive.** Someone must look at every image and tag it.
2. **Labels are a bottleneck on scale.** You cannot label a billion images.
3. **Labels throw information away.** A model trained only to answer "is
   this 1 of 1000 classes?" has no reason to represent anything the label
   does not depend on — pose, material, layout, the relationships between
   parts.

**Self-supervised learning (SSL)** removes labels entirely. The supervisory
signal comes from the *structure of images themselves*. The model is asked
to produce **consistent representations of the same image under different
transformations**. To succeed it must discover what is *stable* about an
image — its content — and that turns out to be exactly what downstream
tasks need.

The result is a **foundation model**: train once, then *freeze* it and reuse
its outputs for classification, segmentation, depth estimation, retrieval,
tracking — without ever fine-tuning the backbone. Sections 6 and the
evaluation figures show that frozen DINO features are already excellent.

---

## 2. The Vision Transformer (ViT) backbone

DINO does not invent a network; it trains a **Vision Transformer**.
See `assets/diagrams/01_vit_pipeline.png` for the whole flow.

### 2.1 From pixels to tokens — "patchify"

A transformer operates on a *sequence of vectors* ("tokens"), not a grid of
pixels. So the image is cut into a grid of fixed-size, non-overlapping
**patches** (`assets/diagrams/02_patchify.png`). Each patch — say 16×16×3
pixels — is flattened and pushed through a single **linear layer** (in
practice a strided convolution), producing one **token vector** of
dimension *D* (384, 768 or 1024 depending on model size).

A 224-pixel image with 16-pixel patches becomes a 14×14 grid = **196 patch
tokens**.

### 2.2 The CLS token and register tokens

Two kinds of extra token are prepended to the patch sequence
(`assets/diagrams/03_token_layout.png`):

- **The CLS token** — one learned vector, the same for every image. As it
  passes through the network it *aggregates* information from all patches,
  and its final value is used as the **image-level embedding**.
- **Register tokens** — a few (DINOv2/DINOv3 use 4) extra learned vectors
  with no fixed job. They act as **scratch space**. Researchers discovered
  that a ViT without them dumps high-magnitude "global" information into a
  handful of random background patches, creating bright **artefacts** in
  attention and feature maps. Register tokens give the network a proper
  place to put that information, leaving the patch tokens clean. The effect
  is visible in `assets/attention/register_comparison.png`.

The full sequence is therefore `[ CLS | reg₁..reg₄ | patch₁..patchₙ ]`.

### 2.3 Positional encoding — where each patch is

Self-attention is **permutation-invariant**: by itself it cannot tell a
top-left patch from a bottom-right one. Position must be injected
(`assets/diagrams/04_positional_encoding.png`):

- **DINOv2** adds a *learned positional embedding* — one trained vector per
  grid cell. Simple, but tied to the grid size used in training; other
  resolutions need interpolation.
- **DINOv3** uses **rotary position embeddings (RoPE)**: the query and key
  vectors are *rotated* by an angle proportional to position. A dot product
  between two rotated vectors then depends only on their *relative*
  position. RoPE — with small random "jittering" of the position scale
  during training — lets DINOv3 run at resolutions it never saw in
  training, which is what makes high-resolution dense tasks practical.

### 2.4 The transformer block

The tokens then pass through *L* identical **transformer blocks** (12 for
ViT-B, 24 for ViT-L). Each block has two residual sub-layers:

1. **Multi-head self-attention.** Every token forms a *query*, a *key* and
   a *value*. Token *i* attends to token *j* with weight
   `softmax(qᵢ·kⱼ / √d)`, then collects a weighted sum of values. This is
   the only place tokens *exchange information*. "Multi-head" means several
   such attentions run in parallel on slices of the vector — and, as
   `assets/attention/attention_heads.png` shows, different heads
   specialise on different regions.
2. **An MLP** applied to each token independently.

Both sub-layers are wrapped with **LayerNorm** and a **residual
connection**; DINO also scales each residual with a small learned factor
(**LayerScale**) for stable deep training.

After the last block a final LayerNorm is applied. The outputs are: **one
CLS vector** (image embedding) and **N patch vectors** (a dense feature
map). `assets/features/layerwise_pca.png` shows the patch features sharpen
from texture-like to part-like as depth increases.

---

## 3. DINO self-distillation — learning without labels

We have an architecture. How is it trained with no labels? Through
**self-distillation**: a network teaches a copy of itself.
See `assets/diagrams/05_student_teacher.png`.

### 3.1 Two networks: student and teacher

There are two networks with the **same architecture**:

- the **student**, updated by gradient descent;
- the **teacher**, *never* touched by gradient descent.

The teacher's weights are an **exponential moving average (EMA)** of the
student's. After every student step:

```
θ_teacher  ←  m · θ_teacher  +  (1 − m) · θ_student
```

with the momentum *m* near 1 (≈ 0.996, rising toward 1.0 on a cosine
schedule — `ema.py`). The teacher is thus a slow, smoothed average of the
student's recent history. Averaging makes it a **more stable and slightly
better target** than the student itself — a "model ensemble in weight
space" — which is why the student can productively chase it.

### 3.2 Multi-crop — many views of one image

From each image, DINO makes several augmented **crops**
(`assets/diagrams/06_multicrop.png`, `multicrop.py`):

- **2 global crops** — large (cover most of the image);
- **several local crops** — small (a fraction of the image).

Plus random flips, colour jitter, grayscale and blur. The key rule: **the
teacher sees only the global crops; the student sees all of them.** Asking
the student's *local-crop* output to match the teacher's *global-crop*
output forces a **"part implies whole"** representation — a patch of fur
must map near the whole cat.

### 3.3 The DINO loss

Each network ends in a **projection head** (`dino_head.py`) that maps the
embedding to a score over *K* learned **prototypes**. A softmax turns those
scores into a probability distribution. The student is trained so its
distribution **matches the teacher's**, by cross-entropy
(`assets/diagrams/07_dino_loss.png`, `dino_loss.py`):

```
loss = −  Σ   P_teacher(view A) · log P_student(view B)
       A ≠ B
```

summed over crop pairs from *different* views.

**Avoiding collapse.** Nothing above stops both networks from emitting one
constant vector for every image (loss zero, features useless). Two
mechanisms prevent this:

- **Sharpening** — the teacher's softmax uses a *low temperature* τ_t,
  making its target distribution peaky and confident.
- **Centering** — a running mean of teacher outputs is subtracted before
  its softmax, so no single prototype can win for every image.

Sharpening pushes toward a confident (one-prototype) answer; centering
pushes toward using all prototypes. Their tension yields informative,
non-collapsed distributions. (DINOv2 can instead use **Sinkhorn–Knopp**
normalisation for the same purpose.)

The mini-training demo (`scripts/11`, `12`) runs exactly this loop. Its
curves are in `assets/training/training_curves.png` and the features
literally emerging from noise in `assets/training/feature_emergence.png`.

---

## 4. The extra objectives: iBOT and KoLeo

The DINO loss alone gives a strong *image-level* (CLS) representation.
DINOv2 and DINOv3 add two more terms to also get strong *dense* features:

- **iBOT — masked image modelling.** Some of the student's patch tokens are
  *masked* (hidden); the student must predict the teacher's output for
  those patches from their visible neighbours. This is "fill in the blank"
  at the patch level, and it is what makes per-patch features sharp enough
  for segmentation and depth.
- **KoLeo regularisation.** A term that pushes the embeddings within a
  batch to **spread out** (it penalises a point sitting too close to its
  nearest neighbour). This spreads the feature space evenly and noticeably
  improves nearest-neighbour **retrieval** (`assets/retrieval/`).

The full DINOv3 objective is: DINO loss + iBOT loss + KoLeo + (DINOv3's new)
Gram-anchoring term. *Our mini-demo implements the DINO loss only*, for
clarity; it is enough to make the mechanism visible.

---

## 5. What DINOv3 adds over DINOv2

DINOv2 (2023) already had the ViT + DINO + iBOT + KoLeo recipe. DINOv3
(2025) keeps it and changes four things. The dedicated file
`DINOV3_VS_DINOV2.md` covers this in detail; in brief:

1. **Scale.** A custom **7-billion-parameter** ViT trained on **1.7 billion
   curated images** (the "LVD-1689M" dataset) — far beyond DINOv2.

2. **Gram anchoring** — *the headline contribution.* Over a very long
   training run, image-level metrics keep improving but **dense patch
   features slowly degrade**: similarity maps grow noisy. DINOv3 keeps a
   frozen earlier checkpoint (the **"Gram teacher"**) whose dense features
   were still clean, and adds a loss that aligns the **Gram matrix** —
   the full patch-to-patch similarity matrix — of the current model to the
   Gram teacher's. Because only the *relational* structure is constrained
   (not the raw feature values), the model keeps improving globally while
   its dense features stay sharp. See
   `assets/diagrams/08_gram_anchoring.png`.

3. **RoPE + high-resolution adaptation.** Rotary position embeddings (§2.3)
   with position jittering let DINOv3 be adapted, after pretraining, to run
   at high resolution — essential for detailed dense prediction.

4. **A distilled model family.** The 7B model is a superb *teacher* but too
   big for most uses, so DINOv3 ships smaller models (ViT-S/B/L/H and
   ConvNeXt variants) **distilled** from it — they inherit much of its
   quality at a fraction of the cost. DINOv3 also aligns a text encoder
   (`dino.txt`) for open-vocabulary, zero-shot use.

---

## 6. Using the features

A trained DINO backbone is **frozen** and reused. Standard ways to use it,
all shown in the figures:

- **k-NN classification** — classify a test image by the labels of its
  nearest training neighbours in feature space. *No training at all.*
- **Linear probe** — train only a single linear layer on the frozen
  features. Its accuracy measures how *linearly separable* the
  representation is. (`assets/evaluation/knn_and_probe.png`,
  `confusion_matrix.png`.)
- **2-D projection** — t-SNE / UMAP of CLS embeddings; classes form clusters
  with no labels used in training (`assets/embeddings/`).
- **Dense tasks** — the patch-token grid feeds lightweight heads for
  segmentation, depth and **correspondence** (`assets/correspondence/`).
- **Retrieval** — nearest-neighbour search in feature space
  (`assets/retrieval/`).

The model-size comparison (`assets/evaluation/model_comparison.png`) shows
the trade-off: bigger ViTs cost more parameters and latency but yield
better, more linearly separable features.

---

## 7. What the mini-training demo proves

Sections 1–5 are theory. Scripts 11–12 *run* it: a ~3.4M-parameter TinyViT
trained from scratch with the DINO loop on unlabelled STL-10. Watch three
things in `assets/training/`:

- the **loss falls** — the student successfully tracks the teacher;
- the **k-NN accuracy climbs far above the 10 % chance level** — the
  features became genuinely useful, *measured only afterwards* with labels;
- **feature PCA emerges from noise** — early-epoch patch features are
  random colour mush; late-epoch features paint coherent regions.

That is DINO, end to end: no labels in, useful structure out.

---

### Further reading

- DINO — *Emerging Properties in Self-Supervised Vision Transformers* (2021)
- DINOv2 — *Learning Robust Visual Features without Supervision* (2023)
- DINOv3 — arXiv:2508.10104 (2025)
- *Vision Transformers Need Registers* (2023)
