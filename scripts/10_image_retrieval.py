"""
10_image_retrieval.py -- Content-based image retrieval with frozen features.

Retrieval is the most direct use of a frozen backbone: embed every image
once, then answer "what looks like this?" with a nearest-neighbour search
in feature space. For each query we show its five closest neighbours by
CLS-feature cosine similarity; a green title means the neighbour shares
the query's class, red means it does not.

    uv run python scripts/10_image_retrieval.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from datasets import STL10_CLASSES, stl10  # noqa: E402
from eval_features import get_eval_features  # noqa: E402
from plot_style import save_figure  # noqa: E402


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("retrieval")
    _, _, test_x, test_y = get_eval_features(config.DEFAULT_BACKBONE)
    images = stl10("test")                       # PIL images, same order

    similarity = test_x @ test_x.t()             # all-pairs cosine sim
    similarity.fill_diagonal_(-1.0)              # never retrieve the query
    k = 5
    queries = [int((test_y == c).nonzero()[0]) for c in range(len(STL10_CLASSES))]

    fig, axes = plt.subplots(len(queries), k + 1,
                             figsize=(2.0 * (k + 1), 2.0 * len(queries)))
    for row, q in enumerate(queries):
        neighbours = similarity[q].topk(k).indices.tolist()
        axes[row, 0].imshow(images[q][0])
        axes[row, 0].set_title(f"query: {STL10_CLASSES[test_y[q]]}",
                               fontsize=9, fontweight="bold")
        for col, idx in enumerate(neighbours):
            axes[row, col + 1].imshow(images[idx][0])
            hit = test_y[idx] == test_y[q]
            axes[row, col + 1].set_title(STL10_CLASSES[test_y[idx]],
                                         fontsize=8,
                                         color="green" if hit else "red")
        for col in range(k + 1):
            axes[row, col].axis("off")
    fig.suptitle("Image retrieval: query (left column) and its 5 nearest "
                 "neighbours by CLS cosine similarity", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out / "retrieval_gallery.png",
                {"group": "retrieval", "script": "10",
                 "model": config.DEFAULT_BACKBONE})
    print(f"retrieval gallery -> {out}")


if __name__ == "__main__":
    main()
