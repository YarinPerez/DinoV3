# TASKS — Implementation Checklist

> **Status:** Draft — awaiting user approval
> **Companion to:** `docs/PRD.md`, `docs/PLANNING.md`
> **Date:** 2026-05-18

Tasks are grouped into 8 phases. Each box is ticked **`[x]` only when the task is fully
done and its acceptance check passes**. A short note (e.g. a measured number) may be
appended on completion. Partial work stays unticked.

Legend: `[ ]` pending · `[~]` in progress · `[x]` done.

---

## Phase 0 — Approvals (gating)

- [x] T0.1 — Write & get approval for `docs/PRD.md`.
- [x] T0.2 — Write `docs/PLANNING.md` and this `docs/TASKS.md`; get approval.
- [x] T0.3 — Confirm: no `src/` code written until T0.2 is approved.

## Phase 1 — Environment scaffold

- [x] T1.1 — Create `pyproject.toml` (Python 3.11, all dependencies, project metadata).
- [x] T1.2 — `uv sync`; create the environment; generate `uv.lock`.
- [x] T1.3 — Add `.gitignore` (`.venv/`, `assets/`, `data/`, `__pycache__/`, hub cache).
- [x] T1.4 — `src/config.py` — dataclasses for models, device, sizes, hyperparams, seeds.
- [x] T1.5 — `src/paths.py` — asset/data directory resolution + creation.
- [x] T1.6 — `src/device.py` — device detection, dtype/AMP, `torch.compile()` wrapper.
- [x] T1.7 — `src/plot_style.py` — matplotlib theme + `save_figure()` (PNG + JSON sidecar).
- [x] T1.8 — `scripts/00_setup_check.py` — verify env, CUDA, DINOv2 download, forward pass.
- **Acceptance:** PASSED — `00_setup_check.py` reports RTX 4090, ViT-S embed dim 384,
  32x32 patch grid; torch 2.12.0+cu130, Python 3.11.15.

## Phase 2 — Core extraction modules

- [x] T2.1 — `src/backbone_loader.py` — load DINOv2 S/B/L (+`_reg`) via `torch.hub` + metadata.
- [x] T2.2 — `src/attention_hook.py` — capture last-block attention `(heads, tok, tok)`.
      Verified: monkey-patch reproduces SDPA maths, rows sum to 1.0, model restored.
- [x] T2.3 — `src/features.py` — CLS / patch-grid / intermediate-layer extraction, L2-norm.
- [x] T2.4 — `src/image_io.py` — load/resize/normalize; patch-divisible sizing.
- [x] T2.5 — `src/datasets.py` — STL-10 + Oxford-IIIT Pet wrappers; sample images.
- [x] T2.6 — `src/dinov3_loader.py` — optional HF DINOv3 path; clean skip when gated.
- **Acceptance:** PASSED — ViT-S extracts CLS `(1,384)`, patch grid `(32,32,384)`,
  attention `(1,6,1025,1025)`; Oxford-IIIT Pet sample images load as real photos.

## Phase 3 — Analysis libraries

- [x] T3.1 — `src/decomposition.py` — PCA (RGB map + fg/bg), t-SNE, UMAP.
- [x] T3.2 — `src/similarity.py` — patch cosine similarity, click-a-patch, correspondence.
- [x] T3.3 — `src/knn_eval.py` — k-NN accuracy on frozen CLS features.
- [x] T3.4 — `src/linear_probe.py` — linear probe: accuracy, per-class report, confusion matrix.
- **Acceptance:** PASSED — smoke test on synthetic clusters: k-NN 1.0, probe 0.97,
  PCA/similarity/t-SNE/UMAP all return correctly-shaped output.

## Phase 4 — Diagrams & visualization scripts (DINOv2)

- [x] T4.1 — diagram modules `diagram_utils`, `diagrams_arch`, `diagrams_pos`
      (ViT pipeline, patchify, CLS/registers, RoPE). Split for the 150-line limit.
- [x] T4.2 — `diagrams_train` + `diagrams_loss` (EMA, multi-crop, DINO loss, Gram anchoring).
- [x] T4.3 — `scripts/01_concept_diagrams.py` → 8 diagrams in `assets/diagrams/`.
- [x] T4.4 — `scripts/02_patch_pca_maps.py` — PCA-RGB maps + fg masks + layer-wise PCA.
- [x] T4.5 — `scripts/03_attention_maps.py` — per-head CLS attention; register comparison.
- [x] T4.6 — `scripts/04_patch_similarity.py` — click-a-patch + similarity matrix.
- [x] T4.7 — `scripts/05_dense_correspondence.py` — cross-image patch matching.
- [x] T4.8 — `scripts/06_embedding_projection.py` — t-SNE / UMAP of CLS embeddings.
- [x] T4.9 — `scripts/07_knn_and_probe.py` — k-NN + linear probe + confusion matrix.
- [x] T4.10 — `scripts/08_resolution_sweep.py` — feature maps vs input resolution.
- [x] T4.11 — `scripts/09_model_comparison.py` — ViT-S/B/L params, dim, latency, k-NN.
- [x] T4.12 — `scripts/10_image_retrieval.py` — nearest-neighbour retrieval gallery.
- [x] T4.13 — Optional DINOv3 path = `src/dinov3_loader.py` (load + `probe_dinov3`),
      probed by `00_setup_check`. Per-script comparison panels were NOT wired because
      the gated weights are unavailable in this environment and cannot be tested;
      the docs carry the DINOv3-specific explanation (per PRD section 4.1).
- **Acceptance:** PASSED — scripts `01`–`10` all exit 0; 22 PNG + JSON sidecars
  written. k-NN/probe: ViT-S 0.989/0.983, ViT-B 0.993/0.990, ViT-L 0.993/0.992.

## Phase 5 — Mini DINO training demo

- [x] T5.1 — `src/training/tiny_vit.py` — small ViT (~3.4M params, patch/CLS/blocks).
- [x] T5.2 — `src/training/dino_head.py` — projection head over learned prototypes.
- [x] T5.3 — `src/training/multicrop.py` — 2 global + 6 local crops, tensor-only (v2).
- [x] T5.4 — `src/training/dino_loss.py` — simplified DINO loss (centering + sharpening).
- [x] T5.5 — `src/training/ema.py` — EMA update + cosine momentum schedule.
- [x] T5.6 — `src/training/knn_monitor.py` — periodic k-NN probe during training.
- [x] T5.7 — `src/training/train_dino.py` — training loop, schedules, logging, checkpoints.
- [x] T5.8 — `scripts/11_run_mini_training.py` — run training; save logs + checkpoints.
- [x] T5.9 — `scripts/12_training_visuals.py` — loss / k-NN / momentum curves +
      per-epoch feature-PCA emergence.
- Note: data loading uses `num_workers=0` — subprocess workers (fork and spawn both)
  segfaulted in this environment; single-process loading is reliable. Augmentation is
  tensor-only (no PIL in the hot path) for the same reason.
- **Acceptance (AC4): PASSED** — 30-epoch run (23 min, RTX 4090): loss 7.60 -> 6.42
  (monotonic decrease), teacher k-NN 0.26 -> 0.31 (~3x the 10% chance level),
  feature-PCA emergence panel shows the object separating from the background.

## Phase 6 — Teaching documents

- [x] T6.1 — `docs/THEORY.md` — full deep-dive (SSL → ViT → DINO loss → iBOT → KoLeo
      → DINOv3 innovations → downstream use), each section linking its figure.
- [x] T6.2 — `docs/DINOV3_VS_DINOV2.md` — comparison table + Gram-anchoring deep dive.
- [x] T6.3 — `docs/VISUALIZATION_GUIDE.md` — catalogue of all 24 figures.
- [x] T6.4 — `README.md` — condensed, figure-rich walkthrough; install & run; every
      figure embedded with a caption.

## Phase 7 — Verification

- [x] T7.1 — `scripts/13_build_readme_assets.py` written and run.
- [x] T7.2 — every script `00`–`13` executed via `uv run` during its phase; all exit 0
      and write their figures (figures inspected visually as they were produced).
- [x] T7.3 — `13` reports **0 missing figures**; 24/24 present; all 24 README
      `assets/...png` references resolve.
- [x] T7.4 — Code audit: largest source file is 144 lines — **no file > 150**.
- [x] T7.5 — `torch.compile()` applied in `train_dino.py` via `device.maybe_compile`;
      `config.set_seeds()` seeds Python/NumPy/torch; `uv.lock` present.
- [x] T7.6 — `README.md` read through; training numbers updated to the measured run.
- **Acceptance (AC1–AC7): PASSED** — all PRD acceptance criteria met.

---

## Acceptance Criteria Cross-Reference (from PRD §9)

| AC | Verified by |
|----|-------------|
| AC1 env/GPU/forward pass | T1.8 / T7.2 |
| AC2 scripts 01–12 run | T7.2 |
| AC3 0 missing figures | T7.1, T7.3 |
| AC4 mini-training converges | Phase 5 acceptance |
| AC5 confusion matrix + probe | T3.4, T4.9 |
| AC6 ≤150 lines, compile, seeds | T7.4, T7.5 |
| AC7 coherent README | T6.4, T7.6 |

---

## Approval

**→ Approve `docs/PLANNING.md` + this `docs/TASKS.md` to unlock implementation
(Phase 1 onward).**
