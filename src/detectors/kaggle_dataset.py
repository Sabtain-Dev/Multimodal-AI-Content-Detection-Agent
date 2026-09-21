import os
import shutil
from pathlib import Path

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
TEXT_LABEL_ALIASES = {
    "ai": {"ai", "artificial", "generated"},
    "human": {"human", "real", "person"},
}
IMAGE_LABEL_ALIASES = {
    "ai": {"ai", "artificial", "generated", "synthetic"},
    "human": {"human", "real", "person"},
}


def _project_data_dir(project_root: Path, modality: str) -> Path:
    return project_root / "data" / modality / "evaluation"


def _env_override(modality: str) -> Path | None:
    env_key = {
        "text": "MULTIMODAL_TEXT_DATASET_DIR",
        "image": "MULTIMODAL_IMAGE_DATASET_DIR",
    }.get(modality)
    if not env_key:
        return None
    value = os.getenv(env_key)
    if value:
        return Path(value).expanduser().resolve()
    return None


def _matches_label(path_name: str, label: str, aliases: dict[str, set[str]]) -> bool:
    lowered = path_name.lower()
    for alias in aliases.get(label, {label}):
        if alias in lowered:
            return True
    return False


def _find_matching_files(root: Path, allow_file: callable, label: str, aliases: dict[str, set[str]]) -> list[Path]:
    matches: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if not allow_file(candidate):
            continue
        file_name = candidate.name.lower()
        stem = candidate.stem.lower()
        parent_name = candidate.parent.name.lower()
        if any(
            _matches_label(file_name, label, aliases)
            or _matches_label(stem, label, aliases)
            or _matches_label(parent_name, label, aliases)
            for _ in [0]
        ):
            matches.append(candidate)
    return matches


def _copy_text_csvs_from_download(download_root: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    discovered = False
    for label in ("ai", "human"):
        label_dir = target_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        matches = _find_matching_files(
            download_root,
            lambda candidate: candidate.suffix.lower() == ".csv",
            label,
            TEXT_LABEL_ALIASES,
        )
        if not matches:
            continue
        discovered = True
        for source in sorted(matches, key=lambda p: (len(p.parts), str(p))):
            destination = label_dir / source.name
            if destination.exists() and destination.stat().st_size == source.stat().st_size:
                continue
            shutil.copy2(source, destination)
    if not discovered:
        raise FileNotFoundError(f"No text CSV files found under Kaggle download: {download_root}")


def _copy_image_files_from_download(download_root: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    discovered = False
    for label in ("ai", "human"):
        label_dir = target_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        matches = _find_matching_files(
            download_root,
            lambda candidate: candidate.suffix.lower() in VALID_IMAGE_EXTENSIONS,
            label,
            IMAGE_LABEL_ALIASES,
        )
        if not matches:
            continue
        discovered = True
        for source in sorted(matches, key=lambda p: (len(p.parts), str(p))):
            destination = label_dir / source.name
            if destination.exists() and destination.stat().st_size == source.stat().st_size:
                continue
            shutil.copy2(source, destination)
    if not discovered:
        raise FileNotFoundError(f"No image files found under Kaggle download: {download_root}")


def _download_kaggle_dataset(dataset_slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for Kaggle dataset downloads. Install it with: pip install kagglehub"
        ) from exc

    kaggle_credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    if not os.getenv("KAGGLE_USERNAME") and not os.getenv("KAGGLE_KEY") and not kaggle_credentials_path.exists():
        raise RuntimeError(
            "Kaggle dataset download requires credentials. Add ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY."
        )

    return Path(kagglehub.dataset_download(dataset_slug))


def _has_local_dataset(local_dir: Path, extension: str) -> bool:
    if not local_dir.exists():
        return False
    for candidate in local_dir.rglob(f"*{extension}"):
        if candidate.is_file():
            return True
    return False


def resolve_text_dataset_dir(project_root: Path) -> Path:
    local_dir = _project_data_dir(project_root, "text")
    override = _env_override("text")
    if override:
        return override
    if (local_dir / "ai").exists() or (local_dir / "human").exists() or _has_local_dataset(local_dir, ".csv"):
        return local_dir

    downloaded_root = _download_kaggle_dataset("itssabtain/ai-vs-human-text-detection-dataset")
    _copy_text_csvs_from_download(downloaded_root, local_dir)
    return local_dir


def resolve_image_dataset_dir(project_root: Path) -> Path:
    local_dir = _project_data_dir(project_root, "image")
    override = _env_override("image")
    if override:
        return override
    if any((local_dir / label).exists() for label in ("ai", "human")) or any(
        candidate.is_file() and candidate.suffix.lower() in VALID_IMAGE_EXTENSIONS
        for candidate in local_dir.rglob("*")
    ):
        return local_dir

    downloaded_root = _download_kaggle_dataset("itssabtain/image-detector-ai-vs-human")
    _copy_image_files_from_download(downloaded_root, local_dir)
    return local_dir
