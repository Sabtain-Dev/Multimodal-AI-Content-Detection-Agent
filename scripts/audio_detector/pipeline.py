import sys
import argparse
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.audio.ensemble import PrimaryDecisionAudioEnsemble


def main():
    parser = argparse.ArgumentParser(description="Run Primary Decision Audio Detector Pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to audio file or directory containing audio files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"input path does not exist: {input_path}")

    supported_extensions = {".mp3", ".m4a", ".wav", ".ogg"}

    if input_path.is_file():
        audio_files = (
            [input_path]
            if input_path.suffix.lower() in supported_extensions
            else []
        )
    else:
        audio_files = [f for f in input_path.rglob("*") if f.suffix.lower() in supported_extensions]

    print(f"Found {len(audio_files)} audio file(s) for inference.\n")
    if not audio_files:
        parser.error(f"no supported audio files found in: {input_path}")

    print("Initializing Audio Detector Models (Model 2 @ 0.90 Primary Decision)...")
    detector = PrimaryDecisionAudioEnsemble()

    results = []
    for fpath in audio_files:
        print(f"Processing: {fpath.name}...")
        res = detector.predict(fpath)
        res["file_name"] = fpath.name
        results.append(res)
        print(f"Result -> Label: {res['label']} | Score: {res['model_score']} | Agreement: {res['models_agree']}")

    print("\n" + "=" * 60)
    print(json.dumps(results if len(results) != 1 else results[0], indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()