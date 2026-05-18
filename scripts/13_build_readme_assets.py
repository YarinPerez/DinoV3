"""
13_build_readme_assets.py -- Verify every expected figure exists.

The project is "done" only when every figure the docs reference has
actually been generated. This script enumerates the expected figures,
checks each PNG (and its JSON sidecar) is present, cross-checks that every
`assets/...png` reference in README.md resolves, and exits non-zero if
anything is missing.

    uv run python scripts/13_build_readme_assets.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

# The complete catalogue, matching docs/VISUALIZATION_GUIDE.md.
EXPECTED = {
    "diagrams": ["01_vit_pipeline", "02_patchify", "03_token_layout",
                 "04_positional_encoding", "05_student_teacher",
                 "06_multicrop", "07_dino_loss", "08_gram_anchoring"],
    "features": ["pca_feature_maps", "layerwise_pca"],
    "attention": ["attention_heads", "attention_overlay",
                  "register_comparison"],
    "similarity": ["click_a_patch", "similarity_matrix"],
    "correspondence": ["dense_correspondence"],
    "embeddings": ["embedding_projection"],
    "evaluation": ["knn_and_probe", "confusion_matrix", "model_comparison"],
    "resolution": ["resolution_feature_maps"],
    "retrieval": ["retrieval_gallery"],
    "training": ["training_curves", "feature_emergence"],
}


def main() -> None:
    missing, no_sidecar = [], []
    total = 0
    for group, names in EXPECTED.items():
        for name in names:
            total += 1
            png = paths.ASSETS / group / f"{name}.png"
            if not png.exists():
                missing.append(f"{group}/{name}.png")
            elif not png.with_suffix(".png.json").exists():
                no_sidecar.append(f"{group}/{name}.png")

    readme = (ROOT / "README.md").read_text()
    refs = sorted(set(re.findall(r"assets/([\w./-]+\.png)", readme)))
    unresolved = [r for r in refs if not (paths.ASSETS / r).exists()]

    print(f"expected figures : {total}")
    print(f"present          : {total - len(missing)}")
    print(f"missing          : {len(missing)}")
    print(f"README references: {len(refs)}  ({len(unresolved)} unresolved)")
    for item in missing:
        print(f"  MISSING  {item}")
    for item in no_sidecar:
        print(f"  NO JSON  {item}")
    for item in unresolved:
        print(f"  README -> missing  assets/{item}")

    if missing or unresolved:
        raise SystemExit("verification FAILED -- some figures are missing")
    print("\nverification PASSED -- every expected figure is present.")


if __name__ == "__main__":
    main()
