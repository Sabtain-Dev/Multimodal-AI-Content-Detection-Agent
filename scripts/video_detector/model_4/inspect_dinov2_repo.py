from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "timm/vit_base_patch14_dinov2.lvd142m"


def main():
    print("=" * 65)
    print("AEGIS - DINOv2 repository inspection")
    print("=" * 65)

    # Check the local Hugging Face cache.
    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--timm--vit_base_patch14_dinov2.lvd142m"
    )

    print(f"\nLocal cache directory: {cache_dir}")
    print(f"Cache exists: {cache_dir.exists()}")

    # Inspect the public Hugging Face repository.
    print(f"\nChecking Hugging Face repository: {REPO_ID}")

    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=REPO_ID)

        print("\nRepository files:")
        for filename in files:
            print(f"  - {filename}")

        safetensor_files = [
            filename
            for filename in files
            if filename.endswith(".safetensors")
        ]

        if not safetensor_files:
            print("\n[WARNING] No safetensors checkpoint found.")
            return

        print("\nSafetensors files found:")
        for filename in safetensor_files:
            print(f"  - {filename}")

        if "model.safetensors" in safetensor_files:
            print("\n[PASS] Expected file 'model.safetensors' is available.")
        else:
            print(
                "\n[INFO] The repository uses a different safetensors "
                "filename. We will use the actual filename in the next step."
            )

    except Exception as exc:
        print(f"\n[ERROR] Repository inspection failed: {exc}")
        raise


if __name__ == "__main__":
    main()