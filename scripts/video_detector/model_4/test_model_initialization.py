
"""
AEGIS - Complete model initialization and checkpoint compatibility test.

Uses:
    Phase 2 checkpoint_best.pt
    Original vendored VideoForensicsDetector architecture
    CPU initialization

This script does not modify the original model architecture.
"""

import gc
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL4_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    MODEL4_DIR / "checkpoints" / "checkpoint_best.pt"
)

VENDOR_DIR = MODEL4_DIR / "vendor" / "aegis"

BRANCH_DIR = VENDOR_DIR / "branches"
UTILS_DIR = VENDOR_DIR / "utils"

sys.path.insert(0, str(BRANCH_DIR))
sys.path.insert(0, str(UTILS_DIR))

from detector_model import VideoForensicsDetector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------

def load_checkpoint():
    """Load the checkpoint safely using CPU memory mapping."""

    print("\n[1/6] Checking checkpoint...")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(
        f"File size: "
        f"{CHECKPOINT_PATH.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print("\n[2/6] Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state'.")

    model_state = checkpoint["model_state"]

    if not isinstance(model_state, dict):
        raise TypeError("'model_state' must be a dictionary.")

    print("[PASS] Checkpoint loaded.")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Model-state entries: {len(model_state)}")

    return checkpoint, model_state


# ---------------------------------------------------------
# Model initialization
# ---------------------------------------------------------

def initialize_model():
    """Instantiate the original AEGIS detector."""

    print("\n[3/6] Initializing AEGIS detector...")

    model = VideoForensicsDetector(
        pixel_output_dim=512,
        motion_output_dim=512,
        consistency_output_dim=256,
        freeze_dino=True,
    )

    model = model.to(DEVICE)

    print("[PASS] AEGIS detector initialized.")
    print(f"Device: {DEVICE}")

    return model


# ---------------------------------------------------------
# State dictionary compatibility
# ---------------------------------------------------------

def validate_state_dictionary(model, checkpoint_state):
    """Compare checkpoint keys and tensor shapes with the model."""

    print("\n[4/6] Checking checkpoint compatibility...")

    model_state = model.state_dict()

    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)

    shape_mismatches = []

    for key in sorted(checkpoint_keys & model_keys):
        checkpoint_tensor = checkpoint_state[key]
        model_tensor = model_state[key]

        if not isinstance(checkpoint_tensor, torch.Tensor):
            raise TypeError(
                f"Checkpoint entry is not a tensor: {key}"
            )

        if checkpoint_tensor.shape != model_tensor.shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": tuple(checkpoint_tensor.shape),
                    "model_shape": tuple(model_tensor.shape),
                }
            )

    print(f"Model-state entries: {len(model_keys)}")
    print(f"Checkpoint entries: {len(checkpoint_keys)}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")

    if missing_keys:
        print("\nMissing checkpoint keys:")
        for key in missing_keys[:20]:
            print(f"  - {key}")

    if unexpected_keys:
        print("\nUnexpected checkpoint keys:")
        for key in unexpected_keys[:20]:
            print(f"  - {key}")

    if shape_mismatches:
        print("\nShape mismatches:")
        for item in shape_mismatches[:20]:
            print(f"  Key: {item['key']}")
            print(
                f"    Checkpoint: {item['checkpoint_shape']}"
            )
            print(f"    Model: {item['model_shape']}")

    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Checkpoint and model architecture are incompatible. "
            "Do not load the checkpoint until the mismatch is investigated."
        )

    print("[PASS] All checkpoint keys and tensor shapes match.")

    return len(model_keys)


# ---------------------------------------------------------
# Load trained weights
# ---------------------------------------------------------

def load_model_weights(model, checkpoint_state):
    """Load all trained weights using strict matching."""

    print("\n[5/6] Loading trained model weights...")

    result = model.load_state_dict(
        checkpoint_state,
        strict=True,
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected state dictionary loading result."
        )

    print("[PASS] All trained weights loaded successfully.")
    print(f"Missing keys: {result.missing_keys}")
    print(f"Unexpected keys: {result.unexpected_keys}")


# ---------------------------------------------------------
# Final validation
# ---------------------------------------------------------

def validate_initialized_model(model):
    """Verify model evaluation mode and parameter statistics."""

    print("\n[6/6] Final model validation...")

    model.eval()

    if model.training:
        raise RuntimeError("Model is still in training mode.")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = total_parameters - trainable_parameters

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Frozen parameters: {frozen_parameters:,}")
    print(f"Model training mode: {model.training}")

    print("[PASS] Model is in evaluation mode.")
    print("[PASS] Model initialization validated.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("AEGIS - COMPLETE MODEL INITIALIZATION TEST")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")

    checkpoint = None
    model = None

    try:
        checkpoint, checkpoint_state = load_checkpoint()

        model = initialize_model()

        matched_entries = validate_state_dictionary(
            model,
            checkpoint_state,
        )

        load_model_weights(
            model,
            checkpoint_state,
        )

        validate_initialized_model(model)

        print("\n" + "=" * 70)
        print("AEGIS MODEL INITIALIZATION SUCCESSFUL.")
        print("=" * 70)

        print(f"Matched checkpoint entries: {matched_entries}")
        print(f"Loaded checkpoint epoch: {checkpoint.get('epoch')}")

    except Exception as exc:
        print("\n" + "=" * 70)
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        print("=" * 70)
        raise

    finally:
        del checkpoint
        gc.collect()


if __name__ == "__main__":
    main()