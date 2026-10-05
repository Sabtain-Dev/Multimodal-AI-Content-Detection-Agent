import json
import sys
import wave
import struct
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import MultimodalAgent


def create_dummy_wav(file_path: Path, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Generate a valid, silent 16kHz WAV file for audio detector testing."""
    num_samples = int(sample_rate * duration_sec)
    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        # Write silent samples
        frames = struct.pack(f"<{num_samples}h", *[0] * num_samples)
        wav_file.writeframes(frames)


def main():
    agent = MultimodalAgent()
    print("=== Multimodal Agent Initialized Successfully ===")

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 1. Test Text Detection
    sample_txt = results_dir / "sample_text.txt"
    sample_txt.write_text(
        "Artificial intelligence systems generate text using deep learning architectures.",
        encoding="utf-8",
    )
    txt_res = agent.detect(str(sample_txt))
    print("\n--- Text Detection Result ---")
    print(json.dumps(txt_res, indent=2))

    # 2. Test Image Detection
    sample_img = results_dir / "sample_image.png"
    sample_img.write_bytes(b"dummy image bytes")
    img_res = agent.detect(str(sample_img), image_m1_score=0.85, image_m2_score=0.92)
    print("\n--- Image Detection Result ---")
    print(json.dumps(img_res, indent=2))

    # 3. Test Audio Detection
    sample_audio = results_dir / "sample_audio.wav"
    create_dummy_wav(sample_audio)
    audio_res = agent.detect(str(sample_audio))
    print("\n--- Audio Detection Result ---")
    print(json.dumps(audio_res, indent=2))

    print("\n=== All Multimodal Integration Tests Completed Successfully ===")


if __name__ == "__main__":
    main()