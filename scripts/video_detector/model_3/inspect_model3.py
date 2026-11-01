from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_ID = "mannicui0/Rethinking-the-readout"
FILES = [
    "README.md",
    "checkpoints/README.md",
    "src/eval_opensource.py",
    "src/pvp_head.py",
    "src/dataset.py",
    "src/backbones/base.py",
    "src/backbones/videomae_k400.py",
]

OUTPUT_DIR = Path("results/model3_repository")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename in FILES:
    print(f"\n{'=' * 25} {filename} {'=' * 25}")
    try:
        local_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
        )
        content = Path(local_path).read_text(
            encoding="utf-8",
            errors="replace",
        )
        destination = OUTPUT_DIR / filename.replace("/", "__")
        destination.write_text(content, encoding="utf-8")
        print(f"Successfully downloaded and saved to {destination}")
    except Exception as e:
        print(f"Could not download {filename}: {e}")