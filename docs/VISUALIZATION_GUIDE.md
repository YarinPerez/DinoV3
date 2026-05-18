# Visualization Guide — Every Figure Explained

The 24 figures this project generates, grouped by theme. For each:
**what** it shows, **how** it is computed, the **takeaway**. Every figure
lives under `assets/<group>/` as a PNG with a JSON sidecar recording its
parameters. The script that produces each group is named.

> Live figures use **DINOv2** backbones (open weights); `docs/THEORY.md`
> explains the mechanism and `docs/DINOV3_VS_DINOV2.md` the DINOv3 deltas.

---

## Group 1 — Concept diagrams · `assets/diagrams/` · script 01

Hand-drawn (matplotlib) explanatory diagrams — no model is run.

| File | What / How / Takeaway |
|---|---|
| `01_vit_pipeline.png` | **What:** the six stages from image to features. **How:** drawn. **Takeaway:** a ViT is a pipeline operating on a *sequence of tokens*. |
| `02_patchify.png` | **What:** a real photo cut into a patch grid; one patch flattened into a vector. **How:** a sample image with a grid overlay. **Takeaway:** an image *becomes a sequence of token vectors*. |
| `03_token_layout.png` | **What:** the token sequence `[CLS | registers | patches]`. **How:** drawn. **Takeaway:** CLS feeds image-level tasks, patches feed dense tasks, registers absorb artefacts. |
| `04_positional_encoding.png` | **What:** learned positional embeddings (DINOv2) vs rotary RoPE (DINOv3). **How:** drawn. **Takeaway:** RoPE encodes *relative* position and rescales to new resolutions. |
| `05_student_teacher.png` | **What:** the self-distillation loop. **How:** drawn. **Takeaway:** the student learns by gradient descent; the teacher is an EMA copy; gradient flows to the student only. |
| `06_multicrop.png` | **What:** 2 global + 6 local crops of one image. **How:** crop rectangles on a real photo. **Takeaway:** the teacher sees only global crops; matching local→global forces a "part implies whole" representation. |
| `07_dino_loss.png` | **What:** the teacher/student distribution-matching loss. **How:** drawn. **Takeaway:** centering + sharpening prevent collapse to a constant output. |
| `08_gram_anchoring.png` | **What:** dense-feature decay over long training, and the Gram-matrix fix. **How:** schematic curves + diagram. **Takeaway:** anchoring the patch-similarity (Gram) matrix to an early checkpoint keeps dense features sharp. |

## Group 2 — Dense patch features · `assets/features/` · script 02

| File | What / How / Takeaway |
|---|---|
| `pca_feature_maps.png` | **What:** 5 images × [input, PCA-RGB feature map, foreground mask]. **How:** patch tokens → 3-component PCA → RGB; 1st component thresholded → mask. **Takeaway:** patch features group by *meaning*; the 1st PCA component segments object from background with no labels. |
| `layerwise_pca.png` | **What:** PCA feature maps after blocks 3, 6, 9, 12. **How:** `get_intermediate_layers`. **Takeaway:** features sharpen with depth — early blocks ≈ texture, late blocks ≈ object parts. |

## Group 3 — Attention · `assets/attention/` · script 03

| File | What / How / Takeaway |
|---|---|
| `attention_heads.png` | **What:** the CLS token's attention for every head of the last block. **How:** attention captured via a forward hook (`attention_hook.py`). **Takeaway:** different heads specialise on different regions. |
| `attention_overlay.png` | **What:** mean attention overlaid on 4 images. **How:** average over heads, percentile-normalised. **Takeaway:** attention concentrates on the salient object. |
| `register_comparison.png` | **What:** ViT-S attention with vs without register tokens. **How:** `dinov2_vits14` vs `dinov2_vits14_reg`. **Takeaway:** registers absorb the bright high-norm artefacts that otherwise pollute the map. |

## Group 4 — Similarity & correspondence · `assets/similarity/`, `correspondence/` · scripts 04, 05

| File | What / How / Takeaway |
|---|---|
| `click_a_patch.png` | **What:** 3 query patches → a cosine-similarity map each. **How:** dot products of L2-normalised patch features. **Takeaway:** semantically similar patches light up wherever they are. |
| `similarity_matrix.png` | **What:** the full patch-to-patch similarity matrix. **How:** `features @ featuresᵀ`. **Takeaway:** its block structure mirrors the scene; this matrix is what DINOv3's Gram anchoring protects. |
| `dense_correspondence.png` | **What:** matching lines between two photos of the same animal type. **How:** each patch of A → nearest-feature patch of B. **Takeaway:** features are instance-invariant — ear matches ear — with no supervision. |

## Group 5 — Embeddings & evaluation · `assets/embeddings/`, `evaluation/` · scripts 06, 07, 09

| File | What / How / Takeaway |
|---|---|
| `embedding_projection.png` | **What:** t-SNE & UMAP of CLS embeddings, coloured by class. **How:** project frozen CLS features to 2-D. **Takeaway:** classes form clusters although no labels were used in training. |
| `knn_and_probe.png` | **What:** k-NN and linear-probe accuracy for ViT-S/B/L. **How:** frozen features → k-NN / a 1-layer probe. **Takeaway:** bigger backbones give more linearly separable features. |
| `confusion_matrix.png` | **What:** the linear probe's confusion matrix (ViT-B). **How:** `sklearn.confusion_matrix`, row-normalised. **Takeaway:** errors are few and concentrated on visually similar classes. |
| `model_comparison.png` | **What:** parameters, embedding dim, latency, k-NN accuracy for ViT-S/B/L. **How:** measured. **Takeaway:** the cost/quality trade-off that motivates DINOv3's distilled family. |

## Group 6 — Resolution · `assets/resolution/` · script 08

| File | What / How / Takeaway |
|---|---|
| `resolution_feature_maps.png` | **What:** one image at 4 resolutions with its PCA feature map. **How:** re-run the backbone at each resolution. **Takeaway:** higher resolution → a finer patch grid; DINOv3's RoPE makes this rescaling seamless. |

## Group 7 — Retrieval · `assets/retrieval/` · script 10

| File | What / How / Takeaway |
|---|---|
| `retrieval_gallery.png` | **What:** 10 query images and their 5 nearest neighbours. **How:** cosine similarity of CLS features. **Takeaway:** a frozen backbone does content-based retrieval directly; green titles = correct class. |

## Group 8 — Mini-training · `assets/training/` · scripts 11, 12

| File | What / How / Takeaway |
|---|---|
| `training_curves.png` | **What:** DINO loss, teacher k-NN accuracy, EMA momentum vs training. **How:** logged during the from-scratch run. **Takeaway:** the loss falls and k-NN accuracy climbs far above the 10% chance level. |
| `feature_emergence.png` | **What:** patch-feature PCA of one image at epochs 0/8/18/29. **How:** load per-epoch checkpoints, run, PCA. **Takeaway:** structure literally emerges from noise as DINO trains. |

---

**Total: 24 figures.** Each is multi-panel, so the project renders well
over a hundred individual visualizations. Re-generate everything with
`scripts/00`–`13` (see `README.md`); `scripts/13_build_readme_assets.py`
verifies that none is missing.
