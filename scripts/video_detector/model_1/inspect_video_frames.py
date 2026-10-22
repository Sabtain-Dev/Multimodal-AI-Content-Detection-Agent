import sys
from pathlib import Path
import decord
from PIL import Image, ImageDraw

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Target test videos and constants
VIDEO_FILES = [
    PROJECT_ROOT / "data/ai/9.mp4",
    PROJECT_ROOT / "data/ai/8.mp4",
    PROJECT_ROOT / "data/ai/1.mp4",
]

FRAMES_PER_VIDEO = 16
COLUMNS = 4
THUMB_WIDTH = 320
THUMB_HEIGHT = 180
CAPTION_HEIGHT = 30
OUTPUT_DIR = PROJECT_ROOT / "results/video_frame_checks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_video(video_path: Path) -> None:
    print(f"\nInspecting: {video_path.relative_to(PROJECT_ROOT)}")
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    total_frames = len(reader)
    fps = float(reader.get_avg_fps())
    width, height = reader[0].shape[1], reader[0].shape[0]

    indices = [
        round(i * (total_frames - 1) / (FRAMES_PER_VIDEO - 1))
        for i in range(FRAMES_PER_VIDEO)
    ] if total_frames > 1 else [0] * FRAMES_PER_VIDEO

    # Remove duplicate indices when the video has very few frames.
    indices = list(dict.fromkeys(indices))
    frames = reader.get_batch(indices).asnumpy()

    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Resolution: {width}x{height}")
    print(f"Sampled frames: {len(frames)}")

    sheet_width = COLUMNS * THUMB_WIDTH
    rows = (len(frames) + COLUMNS - 1) // COLUMNS
    sheet_height = rows * (THUMB_HEIGHT + CAPTION_HEIGHT)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for position, (frame_index, frame_array) in enumerate(
        zip(indices, frames)
    ):
        frame = Image.fromarray(frame_array).convert("RGB")
        frame.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

        x = (position % COLUMNS) * THUMB_WIDTH
        y = (position // COLUMNS) * (THUMB_HEIGHT + CAPTION_HEIGHT)

        # Center the image in its thumbnail area.
        image_x = x + (THUMB_WIDTH - frame.width) // 2
        image_y = y + (THUMB_HEIGHT - frame.height) // 2

        sheet.paste(frame, (image_x, image_y))
        draw.text(
            (x + 5, y + THUMB_HEIGHT + 5),
            f"Sample {position + 1} | Frame {frame_index}",
            fill="black",
        )

    output_path = OUTPUT_DIR / f"{video_path.stem}_contact_sheet.jpg"
    sheet.save(output_path, quality=90)
    print(f"Saved contact sheet: {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    for video_path in VIDEO_FILES:        
        if not video_path.is_file():
            print(f"[SKIP] Video not found: {video_path}")
            continue
        try:
            inspect_video(video_path)
        except Exception as exc:
            print(f"[ERROR] {video_path.name}: {exc}")


if __name__ == "__main__":
    main()