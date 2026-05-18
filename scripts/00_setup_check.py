"""
00_setup_check.py -- Verify the environment before anything else runs.

Checks performed:
  1. Python / PyTorch versions and the active compute device.
  2. A real DINOv2 download from torch.hub + one forward pass.
  3. The token shapes (CLS + patch grid) match what config.py expects.
  4. Whether the optional, gated DINOv3 path is reachable.

Run this first:  uv run python scripts/00_setup_check.py
"""
import sys
from pathlib import Path

# Make the src/ modules importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from device import device_info, get_device  # noqa: E402


def main() -> None:
    print("=" * 64)
    print(" DINOv3 learning resource -- environment check")
    print("=" * 64)
    print(f"Python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")

    device = get_device()
    print(f"device       : {device_info(device)}")

    paths.ensure_dirs()
    print(f"assets dir   : {paths.ASSETS}")

    # --- Download the smallest DINOv2 backbone and run one forward pass. ---
    spec = config.DINOV2_BACKBONES["vits14"]
    print(f"\nDownloading {spec.hub_name} via torch.hub (cached after first run)...")
    model = torch.hub.load(config.DINOV2_HUB_REPO, spec.hub_name, verbose=False)
    model = model.to(device).eval()

    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    with torch.no_grad():
        out = model.forward_features(dummy)
    cls = out["x_norm_clstoken"]
    patch_tokens = out["x_norm_patchtokens"]
    grid = config.IMAGE_SIZE // spec.patch_size

    print(f"  CLS token    : {tuple(cls.shape)}  (embed dim {spec.embed_dim})")
    print(f"  patch tokens : {tuple(patch_tokens.shape)}  "
          f"(expected {grid}x{grid} = {grid * grid})")
    assert cls.shape[-1] == spec.embed_dim, "unexpected CLS dimension"
    assert patch_tokens.shape[1] == grid * grid, "unexpected patch count"

    # --- Probe the optional DINOv3 path; never fail if it is gated. ---
    print("\nOptional DINOv3 (Hugging Face) path:")
    try:
        from dinov3_loader import probe_dinov3
        print("  " + probe_dinov3())
    except Exception as exc:  # ImportError, gated repo, no token, ...
        print(f"  unavailable ({type(exc).__name__}) -- project uses DINOv2 only")

    print("\nAll core checks passed. Environment is ready.\n")


if __name__ == "__main__":
    main()
