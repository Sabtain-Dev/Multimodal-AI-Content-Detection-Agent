import sys
from pathlib import Path
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_1_ID = "Sayantan090/audio-fake-detector"
MODEL_2_ID = "Mahmoud59/wav2vec2-fake-audio-detector"

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = SCRIPT_DIR / "samples"

# Use the sample files shipped in this repo's audio detector dev samples folder.
HUMAN_AUDIO_PATH = SAMPLES_DIR / "Human.ogg"
FAKE_AUDIO_PATH = SAMPLES_DIR / "AI.mp3"

TARGET_SAMPLING_RATE = 16000


def load_and_resample(audio_path: str | Path):
    audio_file = Path(audio_path)
    if not audio_file.is_absolute():
        audio_file = (SCRIPT_DIR / audio_file).resolve()

    if not audio_file.exists():
        print(f"Error: Test file not found: {audio_file}")
        sys.exit(1)

    speech, sr = librosa.load(str(audio_file), sr=TARGET_SAMPLING_RATE)
    return speech


def test_audio_inference(model_id: str, audio_path: str | Path, label_tag: str):
    audio_file = Path(audio_path)
    print(f"\n--- Model: {model_id} | Sample: {label_tag} | File: {audio_file.name} ---")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    speech = load_and_resample(audio_path)
    inputs = feature_extractor(speech, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    id2label = model.config.id2label

    for idx, prob in enumerate(probs):
        label_str = id2label.get(idx, f"LABEL_{idx}")
        print(f"  Index {idx} ({label_str}): {prob:.4f}")

    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label.get(predicted_idx, f"LABEL_{predicted_idx}")
    print(f"  --> Predicted Output: {predicted_label}")


def main():
    print("=" * 60)
    print("CONTROLLED INFERENCE TEST: BOTH MODELS ON BOTH AUDIO FILES")
    print("=" * 60)

    samples = [
        ("Known HUMAN", HUMAN_AUDIO_PATH),
        ("Known FAKE/AI", FAKE_AUDIO_PATH),
    ]

    for label_tag, audio_path in samples:
        print(f"\n[TESTING {label_tag} AUDIO: {Path(audio_path).name}]")
        test_audio_inference(MODEL_1_ID, audio_path, label_tag)
        test_audio_inference(MODEL_2_ID, audio_path, label_tag)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()