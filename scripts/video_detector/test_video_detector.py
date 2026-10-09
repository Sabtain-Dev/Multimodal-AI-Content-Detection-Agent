import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.video import VideoDetectorModel

VIDEO_PATH = Path("data/Lumeluxe-Chatbot Demo.mp4")


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Test video not found: {VIDEO_PATH}")

    print("=" * 60)
    print("TESTING VIDEO DETECTOR INFERENCE")
    print("=" * 60)
    print(f"Loading model and processing video: {VIDEO_PATH}\n")

    detector = VideoDetectorModel()
    result = detector.predict(str(VIDEO_PATH))

    print("Structured Prediction Output:")
    print(json.dumps(result, indent=2))

    assert result["modality"] == "video"
    assert result["label"] in {"AI", "HUMAN"}
    assert 0.0 <= result["model_score"] <= 1.0
    assert len(result["class_scores"]) == 2

    print("\nSUCCESS: Video inference completed successfully.")


if __name__ == "__main__":
    main()