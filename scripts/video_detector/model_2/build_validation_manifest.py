import csv
from pathlib import Path

AI_DIR = Path("data/validation/ai")
HUMAN_DIR = Path("data/validation/human")
OUTPUT = Path("results/validation_manifest.csv")
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def collect(folder, label):
    if not folder.exists():
        print(f"Folder not found yet: {folder}")
        return []
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    return [
        {
            "file_path": str(path),
            "true_label": label,
            "split": "unseen_validation",
        }
        for path in files
    ]

def main():
    records = collect(AI_DIR, "AI") + collect(HUMAN_DIR, "HUMAN")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["file_path", "true_label", "split"]
        )
        writer.writeheader()
        writer.writerows(records)
        
    ai_count = sum(r["true_label"] == "AI" for r in records)
    human_count = sum(r["true_label"] == "HUMAN" for r in records)
    
    print(f"AI validation videos: {ai_count}")
    print(f"Human validation videos: {human_count}")
    print(f"Total: {len(records)}")
    print(f"Manifest: {OUTPUT}")
    
    if not records:
        print(
            "Add new videos to data/validation/ai/ and "
            "data/validation/human/, then rerun this script."
        )
    elif ai_count == 0 or human_count == 0:
        print("WARNING: Both classes are needed for a balanced evaluation.")

if __name__ == "__main__":
    main()