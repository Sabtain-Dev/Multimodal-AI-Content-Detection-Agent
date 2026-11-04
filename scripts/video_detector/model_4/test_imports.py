
"""Test AEGIS imports without initializing the model or downloading weights."""

import sys
from pathlib import Path


MODEL4_DIR = Path(__file__).resolve().parent
AEGIS_DIR = MODEL4_DIR / "vendor" / "aegis"

BRANCHES_DIR = AEGIS_DIR / "branches"
UTILS_DIR = AEGIS_DIR / "utils"


def main():
    # The upstream code uses direct imports such as:
    # from detector_model import VideoForensicsDetector
    #
    # Add the original source directories to Python's module search path.
    sys.path.insert(0, str(UTILS_DIR))
    sys.path.insert(0, str(BRANCHES_DIR))

    print("=" * 65)
    print("AEGIS SOURCE IMPORT TEST")
    print("=" * 65)

    modules = [
        "pixel_branch",
        "motion_branch",
        "consistency_branch",
        "detector_model",
        "video_io",
        "inference",
    ]

    imported_modules = {}

    for module_name in modules:
        try:
            module = __import__(module_name)
            imported_modules[module_name] = module

            print(f"[PASS] {module_name}")
            print(f"       {module.__file__}")

        except Exception as exc:
            print(f"[FAIL] {module_name}")
            print(f"       {type(exc).__name__}: {exc}")
            raise

    # Verify the main model class is accessible.
    from detector_model import VideoForensicsDetector

    print("\nMain model class:")
    print(VideoForensicsDetector)

    # Verify the video loading function is accessible.
    from video_io import load_video

    print("\nVideo loading function:")
    print(load_video)

    print("\n" + "=" * 65)
    print("ALL AEGIS IMPORT TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()