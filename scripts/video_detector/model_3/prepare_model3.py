from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "mannicui0/Rethinking-the-readout"

MODEL_DIR = Path("models")
HEAD_PATH = MODEL_DIR / "model3_vpvp.pt"

CHECKPOINT_REPO_PATH = (
    "checkpoints/pvp_heads/videomae_k400_vpvp.pt"
)


def main():
    print("=" * 60)
    print("Preparing Model 3: VideoMAE-K400 + V-PVP")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not Path("models/model3_videomae_k400/config.json").exists():
        raise FileNotFoundError(
            "VideoMAE backbone is missing. "
            "The existing backbone download was not found."
        )

    print("\n[1/2] VideoMAE backbone")
    print("Already downloaded: models/model3_videomae_k400")

    print("\n[2/2] Downloading V-PVP checkpoint")

    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=CHECKPOINT_REPO_PATH,
    )

    shutil.copy2(downloaded_file, HEAD_PATH)

    print("\n" + "=" * 60)
    print("Model preparation completed.")
    print(f"Backbone: models/model3_videomae_k400")
    print(f"V-PVP head: {HEAD_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()