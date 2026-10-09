from pathlib import Path
from typing import List, Union
import numpy as np
import decord
from decord import VideoReader, cpu

# Force decord to use CPU
decord.bridge.set_bridge("native")


class VideoProcessor:
    """Handles video loading and frame extraction for video classification models."""

    def __init__(self, num_frames: int = 16):
        """
        Args:
            num_frames: Number of frames to uniformly sample across the video duration.
        """
        self.num_frames = num_frames

    def extract_uniform_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """Uniformly samples `num_frames` RGB frames across the entire video.

        Args:
            video_path: Path to the video file (.mp4, .avi, .mov, etc.).

        Returns:
            List of RGB numpy arrays, each with shape (H, W, 3).
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Initialize VideoReader
        vr = VideoReader(str(path), ctx=cpu(0))
        total_frames = len(vr)

        if total_frames == 0:
            raise ValueError(f"Video file contains no frames: {video_path}")

        # Calculate uniformly spaced frame indices
        if total_frames <= self.num_frames:
            # If video has fewer frames than required, pad by repeating indices
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        # Extract selected frames as numpy arrays (RGB)
        frames = vr.get_batch(indices).asnumpy()

        # Convert list of NDArrays to python list of HxWx3 RGB arrays
        return [frames[i] for i in range(len(frames))]