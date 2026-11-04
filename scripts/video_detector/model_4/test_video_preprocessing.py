
from pathlib import Path

import torch

from scripts.video_detector.model_4.vendor.aegis.utils.video_io import (
    DECORD_AVAILABLE,
    load_video,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

AI_DIR = PROJECT_ROOT / "data" / "ai"
HUMAN_DIR = PROJECT_ROOT / "data" / "human"

N_FRAMES = 16
N_SEMANTIC = 8
FRAME_HEIGHT = 224
FRAME_WIDTH = 224


def get_sample_video(directory: Path) -> Path:
    """Return the first available MP4 video from a dataset directory."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    videos = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )

    if not videos:
        raise FileNotFoundError(
            f"No MP4 videos found in: {directory}"
        )

    return videos[0]


def validate_frame_bundle(bundle, video_path: Path) -> None:
    """Validate the structure and numerical values of a FrameBundle."""

    expected_all = (
        N_FRAMES,
        3,
        FRAME_HEIGHT,
        FRAME_WIDTH,
    )

    expected_semantic = (
        N_SEMANTIC,
        3,
        FRAME_HEIGHT,
        FRAME_WIDTH,
    )

    print(f"\nVideo: {video_path}")
    print(f"Total source frames: {bundle.total_frames}")
    print(f"FPS: {bundle.fps:.2f}")
    print(f"Duration: {bundle.duration:.2f} seconds")
    print(f"Original resolution: {bundle.resolution}")

    print(f"All frames shape: {tuple(bundle.frames_all.shape)}")
    print(
        "Semantic frames shape: "
        f"{tuple(bundle.frames_semantic.shape)}"
    )
    print(
        "Importance scores shape: "
        f"{bundle.importance_scores.shape}"
    )

    if tuple(bundle.frames_all.shape) != expected_all:
        raise AssertionError(
            f"Unexpected frames_all shape: {bundle.frames_all.shape}"
        )

    if tuple(bundle.frames_semantic.shape) != expected_semantic:
        raise AssertionError(
            "Unexpected frames_semantic shape: "
            f"{bundle.frames_semantic.shape}"
        )

    if bundle.importance_scores.shape != (N_FRAMES,):
        raise AssertionError(
            "Unexpected importance_scores shape: "
            f"{bundle.importance_scores.shape}"
        )

    if not torch.isfinite(bundle.frames_all).all():
        raise AssertionError("frames_all contains NaN or infinity.")

    if not torch.isfinite(bundle.frames_semantic).all():
        raise AssertionError(
            "frames_semantic contains NaN or infinity."
        )

    if not torch.isfinite(
        torch.as_tensor(bundle.importance_scores)
    ).all():
        raise AssertionError(
            "Importance scores contain NaN or infinity."
        )

    if bundle.fps <= 0:
        raise AssertionError("Invalid video FPS.")

    if bundle.duration <= 0:
        raise AssertionError("Invalid video duration.")

    if min(bundle.resolution) < 128:
        raise AssertionError(
            f"Unexpectedly low video resolution: {bundle.resolution}"
        )

    print("[PASS] FrameBundle validation completed.")


def test_video(video_path: Path) -> None:
    """Test preprocessing and deterministic frame sampling."""

    print("\n" + "=" * 70)
    print(f"Testing: {video_path.parent.name.upper()} VIDEO")
    print("=" * 70)

    print("\n[1/3] Loading video...")

    bundle = load_video(
        video_path=str(video_path),
        n_frames=N_FRAMES,
        n_semantic=N_SEMANTIC,
        height=FRAME_HEIGHT,
        width=FRAME_WIDTH,
        sampling="window",
        target_dur=4.0,
        random_start=False,
        quality_filter=True,
    )

    print("[PASS] Video loaded successfully.")

    print("\n[2/3] Validating FrameBundle...")

    validate_frame_bundle(bundle, video_path)

    print("\n[3/3] Checking deterministic sampling...")

    second_bundle = load_video(
        video_path=str(video_path),
        n_frames=N_FRAMES,
        n_semantic=N_SEMANTIC,
        height=FRAME_HEIGHT,
        width=FRAME_WIDTH,
        sampling="window",
        target_dur=4.0,
        random_start=False,
        quality_filter=True,
    )

    if not torch.equal(
        bundle.frames_all,
        second_bundle.frames_all,
    ):
        raise AssertionError(
            "Frame sampling is not deterministic."
        )

    if not torch.equal(
        bundle.frames_semantic,
        second_bundle.frames_semantic,
    ):
        raise AssertionError(
            "Semantic frame selection is not deterministic."
        )

    print("[PASS] Repeated preprocessing produced identical frames.")
    print(f"\n[PASS] {video_path.parent.name.upper()} video test passed.")


def main() -> None:
    print("=" * 70)
    print("AEGIS - VIDEO PREPROCESSING TEST")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Decord available: {DECORD_AVAILABLE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    ai_video = get_sample_video(AI_DIR)
    human_video = get_sample_video(HUMAN_DIR)

    print(f"\nSelected AI video: {ai_video}")
    print(f"Selected human video: {human_video}")

    test_video(ai_video)
    test_video(human_video)

    print("\n" + "=" * 70)
    print("ALL VIDEO PREPROCESSING TESTS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    main()