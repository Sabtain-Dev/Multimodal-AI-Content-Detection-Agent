
import torch

from scripts.video_detector.model_4.vendor.aegis.branches.pixel_branch import (
    DINOv2Encoder,
)


def main():
    print("=" * 65)
    print("AEGIS - DINOv2 initialization test")
    print("=" * 65)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        print("\n[1/4] Initializing DINOv2 encoder...")

        encoder = DINOv2Encoder(
            output_dim=256,
            freeze=True,
        )

        encoder.eval()

        print("[PASS] DINOv2 encoder initialized.")

        print("\n[2/4] Checking backbone parameters...")

        backbone_params = list(encoder.backbone.parameters())

        backbone_total = sum(
            parameter.numel()
            for parameter in backbone_params
        )

        backbone_trainable = sum(
            parameter.numel()
            for parameter in backbone_params
            if parameter.requires_grad
        )

        print(f"Backbone parameters: {backbone_total:,}")
        print(f"Trainable backbone parameters: {backbone_trainable:,}")

        if backbone_trainable != 0:
            raise RuntimeError(
                "DINOv2 backbone should be completely frozen."
            )

        print("[PASS] DINOv2 backbone is frozen.")

        print("\n[3/4] Checking projection head parameters...")

        projection_params = list(encoder.proj.parameters())

        projection_total = sum(
            parameter.numel()
            for parameter in projection_params
        )

        projection_trainable = sum(
            parameter.numel()
            for parameter in projection_params
            if parameter.requires_grad
        )

        print(f"Projection parameters: {projection_total:,}")
        print(f"Trainable projection parameters: {projection_trainable:,}")

        if projection_trainable != projection_total:
            raise RuntimeError(
                "Projection head should remain trainable."
            )

        print("[PASS] Projection head is trainable.")

        print("\n[4/4] Testing forward pass...")

        # One video, two frames, RGB, 224x224.
        dummy_input = torch.randn(1, 2, 3, 224, 224)

        with torch.inference_mode():
            output = encoder(dummy_input)

        print(f"Input shape:  {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")

        expected_shape = (1, 2, 256)

        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected output shape: {tuple(output.shape)}"
            )

        if not torch.isfinite(output).all():
            raise RuntimeError(
                "Output contains NaN or infinite values."
            )

        print("[PASS] Forward pass completed.")

        print("\nALL DINOv2 TESTS PASSED.")

    except Exception as exc:
        print(f"\n[FAILED] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()