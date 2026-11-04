from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "timm/vit_base_patch14_dinov2.lvd142m"
FILENAME = "model.safetensors"


def main():
    print("=" * 65)
    print("AEGIS - Download DINOv2 weights")
    print("=" * 65)

    print(f"\nRepository: {REPO_ID}")
    print(f"Checkpoint: {FILENAME}")

    try:
        checkpoint_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
        )

        checkpoint = Path(checkpoint_path)

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Downloaded checkpoint not found: {checkpoint}"
            )

        size_mb = checkpoint.stat().st_size / (1024 * 1024)

        print("\n[PASS] DINOv2 checkpoint is available.")
        print(f"Location: {checkpoint}")
        print(f"Size: {size_mb:.2f} MB")

    except Exception as exc:
        print(f"\n[ERROR] DINOv2 download failed: {exc}")
        raise


if __name__ == "__main__":
    main()