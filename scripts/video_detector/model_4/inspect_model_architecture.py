
"""Inspect the actual vendored AEGIS Phase 2 architecture."""

import inspect
import sys
from pathlib import Path

MODEL4_DIR = Path(__file__).resolve().parent
VENDOR_DIR = MODEL4_DIR / "vendor" / "aegis"

BRANCH_DIR = VENDOR_DIR / "branches"
UTILS_DIR = VENDOR_DIR / "utils"

sys.path.insert(0, str(BRANCH_DIR))
sys.path.insert(0, str(UTILS_DIR))

from detector_model import VideoForensicsDetector
from pixel_branch import PixelBranch
from motion_branch import MotionBranch
from consistency_branch import ConsistencyBranch


def inspect_class(cls, show_forward=False):
    print("\n" + "=" * 70)
    print(f"CLASS: {cls.__name__}")
    print("=" * 70)

    print(f"Source file: {inspect.getfile(cls)}")
    print(f"Constructor: {inspect.signature(cls.__init__)}")

    if show_forward:
        print("\nForward method:")
        print(inspect.getsource(cls.forward))


def main():
    print("=" * 70)
    print("AEGIS - LOCAL ARCHITECTURE INSPECTION")
    print("=" * 70)

    inspect_class(VideoForensicsDetector, show_forward=True)
    inspect_class(PixelBranch)
    inspect_class(MotionBranch)
    inspect_class(ConsistencyBranch)

    print("\n" + "=" * 70)
    print("ARCHITECTURE INSPECTION COMPLETED.")
    print("=" * 70)


if __name__ == "__main__":
    main()