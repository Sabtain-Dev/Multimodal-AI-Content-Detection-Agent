
"""
Download and validate the official AEGIS Phase 2 checkpoint.

Destination:
scripts/video_detector/model_4/checkpoints/checkpoint_best.pt
"""

from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download


REPO_ID = "MusapYildiz/aegis-video-detector"
FILENAME = "checkpoint_best.pt"

MODEL4_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODEL4_DIR / "checkpoints"


def download_checkpoint() -> Path:
    """Download the official checkpoint into the Model 4 directory."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Downloading official AEGIS checkpoint...")
    print(f"Repository: {REPO_ID}")
    print(f"File: {FILENAME}")
    print(f"Destination: {CHECKPOINT_DIR}")

    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(CHECKPOINT_DIR),
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint download did not produce a file: {checkpoint_path}"
        )

    print(f"[PASS] Download completed: {checkpoint_path}")
    return checkpoint_path


def validate_file(checkpoint_path: Path) -> None:
    """Check that the downloaded checkpoint is a non-empty file."""

    print("\n[2/4] Validating checkpoint file...")

    file_size = checkpoint_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    print(f"File size: {file_size_mb:.2f} MB")

    if file_size == 0:
        raise RuntimeError("Checkpoint file is empty.")

    if file_size_mb < 1:
        raise RuntimeError(
            "Checkpoint file is unexpectedly small. "
            "The download may be incomplete."
        )

    print("[PASS] Checkpoint file exists and is non-empty.")


def inspect_checkpoint(checkpoint_path: Path) -> None:
    """Load checkpoint safely on CPU and inspect its structure."""

    print("\n[3/4] Loading checkpoint on CPU...")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Safe checkpoint loading failed. "
            "No unrestricted pickle loading was attempted. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the checkpoint to contain a dictionary, "
            f"received: {type(checkpoint).__name__}"
        )

    print("[PASS] Checkpoint loaded successfully.")
    print("\nTop-level checkpoint keys:")

    for key in checkpoint.keys():
        print(f"  - {key}")

    if "model_state" not in checkpoint:
        raise KeyError(
            "Expected 'model_state' was not found in the checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model_state = checkpoint["model_state"]

    if not isinstance(model_state, dict):
        raise TypeError(
            "'model_state' must be a dictionary, "
            f"received: {type(model_state).__name__}"
        )

    tensor_items = {
        key: value
        for key, value in model_state.items()
        if isinstance(value, torch.Tensor)
    }

    if not tensor_items:
        raise RuntimeError(
            "No PyTorch tensors were found in model_state."
        )

    print(f"\nModel-state entries: {len(model_state):,}")
    print(f"Tensor entries: {len(tensor_items):,}")

    print("\nFirst 12 model-state entries:")

    for index, (key, value) in enumerate(tensor_items.items()):
        if index >= 12:
            break

        print(f"  {key}: {tuple(value.shape)}")

    non_finite_keys = [
        key
        for key, value in tensor_items.items()
        if value.is_floating_point()
        and not torch.isfinite(value).all().item()
    ]

    if non_finite_keys:
        raise RuntimeError(
            "Non-finite values detected in checkpoint tensors: "
            f"{non_finite_keys[:10]}"
        )

    print("\n[PASS] Model-state structure and tensor values validated.")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")


def main() -> None:
    print("=" * 70)
    print("AEGIS - CHECKPOINT DOWNLOAD AND VALIDATION")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print("Loading device: CPU")

    try:
        checkpoint_path = download_checkpoint()
        validate_file(checkpoint_path)
        inspect_checkpoint(checkpoint_path)

        print("\n[4/4] Final verification...")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Size: {checkpoint_path.stat().st_size / (1024 * 1024):.2f} MB")

        print("\n" + "=" * 70)
        print("CHECKPOINT DOWNLOAD AND VALIDATION COMPLETED.")
        print("=" * 70)

    except Exception as exc:
        print("\n" + "=" * 70)
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()