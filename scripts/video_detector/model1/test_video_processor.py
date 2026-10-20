import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.video.video_processor import VideoProcessor


def main():
    print("=" * 60)
    print("TESTING VIDEO PROCESSOR (FRAME EXTRACTION)")
    print("=" * 60)

    processor = VideoProcessor(num_frames=16)

    # Search for any existing MP4 sample file or prompt user
    sample_videos = list(Path(".").rglob("*.mp4"))

    if not sample_videos:
        print("\nNo .mp4 files found in the repository.")
        print("Please provide a path to an MP4 video file to test frame extraction.")
        return

    test_video = sample_videos[0]
    print(f"\nFound test video: {test_video}")

    frames = processor.extract_uniform_frames(test_video)

    print(f"\nExtracted frames count: {len(frames)}")
    print(f"Individual frame shape (H, W, C): {frames[0].shape}")
    print(f"Data type: {frames[0].dtype}")

    if len(frames) == 16 and frames[0].shape[2] == 3:
        print("\nSUCCESS: Frame extraction produced exact 16 x H x W x 3 sequence!")
    else:
        print("\nWARNING: Frame output shape does not match expected criteria.")


if __name__ == "__main__":
    main()