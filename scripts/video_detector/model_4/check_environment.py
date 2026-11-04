
"""Audit the current environment before integrating AEGIS."""

import importlib
import importlib.metadata
import importlib.util
import platform
import sys


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "scikit-learn": "sklearn",
    "safetensors": "safetensors",
    "Pillow": "PIL",
    "timm": "timm",
    "einops": "einops",
    "opencv-python": "cv2",
}

OPENCV_DISTRIBUTIONS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
]


def get_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def check_import(module_name):
    try:
        importlib.import_module(module_name)
        return "IMPORT OK"
    except Exception as exc:
        return f"IMPORT FAILED: {type(exc).__name__}: {exc}"


def main():
    print("=" * 65)
    print("AEGIS MODEL 4 - ENVIRONMENT AUDIT")
    print("=" * 65)

    print(f"\nPython version : {sys.version}")
    print(f"Python path    : {sys.executable}")
    print(f"Platform       : {platform.platform()}")

    print("\n--- Package Versions and Import Tests ---")

    for distribution, module in PACKAGES.items():
        version = get_version(distribution)
        import_status = check_import(module)

        print(f"\n{distribution}")
        print(f"  Version : {version}")
        print(f"  Import  : {import_status}")

    print("\n--- OpenCV Distribution Check ---")

    for distribution in OPENCV_DISTRIBUTIONS:
        print(f"{distribution}: {get_version(distribution)}")

    print("\n--- PyTorch Runtime Check ---")

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"Device          : CPU")

        if torch.cuda.is_available():
            print(f"GPU             : {torch.cuda.get_device_name(0)}")

    except Exception as exc:
        print(f"PyTorch runtime check failed: {exc}")

    print("\n--- Torchvision Compatibility Check ---")

    try:
        import torch
        import torchvision

        print(f"Torchvision version : {torchvision.__version__}")

        # Basic tensor operation using torchvision.
        from torchvision import transforms

        transform = transforms.Resize((224, 224))
        print(f"Resize transform   : {transform}")
        print("Torchvision test   : PASSED")

    except Exception as exc:
        print(
            "Torchvision test   : FAILED - "
            f"{type(exc).__name__}: {exc}"
        )

    print("\n" + "=" * 65)
    print("ENVIRONMENT AUDIT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()