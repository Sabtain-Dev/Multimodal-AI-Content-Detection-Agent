import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.video.video_processor import VideoProcessor


def main():
    processor = VideoProcessor(num_frames=16)
    test_files = [
        Path("data/ai/9.mp4"),
        Path("data/ai/8.mp4"),
        Path("data/ai/1.mp4"),
    ]

    for f in test_files:
        if not f.exists():
            print(f"Skipping {f} (not found)")
            continue
        try:
            frames = processor.extract_uniform_frames(f)
            print(f"[OK] {f.name}: Extracted {len(frames)} frames with shape {frames[0].shape}")
        except Exception as e:
            print(f"[ERROR] {f.name} failed decord extraction: {e}")

if __name__ == "__main__":
    main()