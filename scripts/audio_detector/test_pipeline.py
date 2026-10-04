import sys
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.audio.ensemble import PrimaryDecisionAudioEnsemble


def main():
    print("=" * 60)
    print("TESTING REUSABLE AUDIO DETECTOR PIPELINE")
    print("=" * 60)

    # Locate sample evaluation file
    eval_dir = Path("data/audio/evaluation")
    sample_files = list(eval_dir.rglob("*.mp3")) + list(eval_dir.rglob("*.wav")) + list(eval_dir.rglob("*.m4a")) + list(eval_dir.rglob("*.ogg"))

    if not sample_files:
        print(f"Error: No audio samples found in {eval_dir}")
        sys.exit(1)

    test_sample = sample_files[0]
    print(f"Testing sample file: {test_sample}")

    detector = PrimaryDecisionAudioEnsemble()
    output = detector.predict(test_sample)

    print("\nStructured Detector Output Interface:")
    print(json.dumps(output, indent=2))

    # Output verification assertions
    assert output["modality"] == "audio"
    assert output["label"] in {"AI", "HUMAN"}
    assert "model_1" in output
    assert "model_2" in output
    assert output["decision_threshold"] == 0.90

    print("\nPipeline test passed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()