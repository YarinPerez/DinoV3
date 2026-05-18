"""
07_knn_and_probe.py -- Quantify frozen-feature quality.

Two standard label-free-backbone benchmarks, run for ViT-S / B / L:
  * k-NN     -- classify a test image by its nearest training neighbours,
  * linear probe -- train only a single linear layer on frozen features.
Neither touches the backbone, so both measure the representation itself.
A confusion matrix shows *which* classes the probe still mixes up.

    uv run python scripts/07_knn_and_probe.py
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from datasets import STL10_CLASSES  # noqa: E402
from device import get_device  # noqa: E402
from eval_features import get_eval_features  # noqa: E402
from knn_eval import knn_accuracy  # noqa: E402
from linear_probe import evaluate_probe, train_probe  # noqa: E402
from plot_style import save_figure  # noqa: E402
from plot_style import PALETTE  # noqa: E402


def bars_figure(results):
    """Side-by-side bar charts of k-NN and linear-probe accuracy."""
    keys = list(results)
    labels = [results[k]["label"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, field, title in [(axes[0], "knn", "k-NN accuracy"),
                             (axes[1], "probe", "linear-probe accuracy")]:
        values = [results[k][field] for k in keys]
        ax.bar(labels, values, color=PALETTE[:len(keys)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{title} (frozen features, STL-10)")
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    fig.suptitle("Bigger backbones produce more linearly-separable features",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def confusion_figure(matrix, accuracy):
    """Row-normalised confusion matrix of the linear probe."""
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.grid(False)
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ticks = range(len(STL10_CLASSES))
    ax.set_xticks(ticks, STL10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(ticks, STL10_CLASSES)
    for i in ticks:
        for j in ticks:
            ax.text(j, i, f"{normalised[i, j]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if normalised[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"Linear-probe confusion matrix -- accuracy {accuracy:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    return fig


def main() -> None:
    config.set_seeds()
    out = paths.group_dir("evaluation")
    device = get_device()
    n_classes = config.TRAIN.num_classes

    results, confusion = {}, None
    for key in config.COMPARISON_KEYS:
        train_x, train_y, test_x, test_y = get_eval_features(key)
        knn = knn_accuracy(train_x, train_y, test_x, test_y,
                           config.EVAL.knn_k, n_classes)
        probe = train_probe(train_x, train_y, n_classes, device)
        report = evaluate_probe(probe, test_x, test_y, STL10_CLASSES, device)
        results[key] = {"label": config.DINOV2_BACKBONES[key].label,
                        "knn": knn, "probe": report["accuracy"]}
        if key == config.DEFAULT_BACKBONE:
            confusion = report["confusion_matrix"]

    save_figure(bars_figure(results), out / "knn_and_probe.png",
                {"group": "evaluation", "script": "07", "results": results})
    save_figure(confusion_figure(confusion,
                                 results[config.DEFAULT_BACKBONE]["probe"]),
                out / "confusion_matrix.png",
                {"group": "evaluation", "script": "07"})
    (out / "eval_metrics.json").write_text(json.dumps(results, indent=2))
    print("results:", {k: (round(v["knn"], 3), round(v["probe"], 3))
                       for k, v in results.items()})
    print(f"evaluation figures -> {out}")


if __name__ == "__main__":
    main()
