"""
01_concept_diagrams.py -- Render the architecture & training diagrams.

These are hand-drawn (matplotlib) explanatory figures, not model outputs.
They give the reader a mental model before the real-data visualizations:
the ViT pipeline, patchify, the token layout, positional encoding, the
student/teacher loop, multi-crop, the DINO loss and Gram anchoring.

    uv run python scripts/01_concept_diagrams.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import config  # noqa: E402
import paths  # noqa: E402
from datasets import sample_images  # noqa: E402
from diagrams_arch import (draw_patchify, draw_token_layout,  # noqa: E402
                           draw_vit_pipeline)
from diagrams_loss import draw_dino_loss, draw_gram_anchoring  # noqa: E402
from diagrams_pos import draw_positional_encoding  # noqa: E402
from diagrams_train import draw_multicrop, draw_student_teacher  # noqa: E402
from image_io import to_displayable  # noqa: E402
from plot_style import save_figure  # noqa: E402


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("diagrams")

    # One real photo feeds the patchify and multi-crop diagrams.
    _, _, tensor = sample_images(1)[0]
    image_rgb = to_displayable(tensor)

    figures = {
        "01_vit_pipeline": draw_vit_pipeline(),
        "02_patchify": draw_patchify(image_rgb),
        "03_token_layout": draw_token_layout(),
        "04_positional_encoding": draw_positional_encoding(),
        "05_student_teacher": draw_student_teacher(),
        "06_multicrop": draw_multicrop(image_rgb),
        "07_dino_loss": draw_dino_loss(),
        "08_gram_anchoring": draw_gram_anchoring(),
    }
    for name, fig in figures.items():
        path = save_figure(fig, out / f"{name}.png",
                           {"group": "diagrams", "script": "01",
                            "kind": "concept diagram"})
        print(f"  saved {path.name}")
    print(f"\n{len(figures)} concept diagrams written to {out}")


if __name__ == "__main__":
    main()
