from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}


def detect_modality(file_path: str) -> str:
    """Determine the supported modality from a file extension.

    Args:
        file_path: Path to the target input file.

    Returns:
        "text", "image", or "audio".

    Raises:
        FileNotFoundError: If the input file does not exist on disk.
        ValueError: If the file extension is unsupported (e.g., .mp4).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return "text"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    else:
        raise ValueError(
            f"Unsupported modality for extension '{ext}'. Currently supported: text, image, audio."
        )