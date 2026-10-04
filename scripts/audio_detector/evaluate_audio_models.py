import sys
import time
import io
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.audio.label_normalization import normalize_audio_label

MODEL_1_ID = "Sayantan090/audio-fake-detector"
MODEL_2_ID = "Mahmoud59/wav2vec2-fake-audio-detector"

DATASET_DIR = REPO_ROOT / "data" / "audio" / "evaluation"
RESULTS_DIR = REPO_ROOT / "results"
SAMPLE_RATE = 16000

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"}


def iter_supported_audio_files(base_dir: Path) -> list[tuple[Path, str]]:
    """Recursively collect supported audio files from the evaluation dataset."""
    files: list[tuple[Path, str]] = []
    if not base_dir.exists():
        return files

    for audio_path in sorted(base_dir.rglob("*")):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if "ai" in audio_path.parts:
            files.append((audio_path, "AI"))
        elif "human" in audio_path.parts:
            files.append((audio_path, "HUMAN"))

    return files


def load_audio_robust(audio_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Robust audio loader using PyDub with FFmpeg fallback to support
    M4A, MP3, OGG, WAV, FLAC, AAC, and edge-case files with unusual headers.
    """
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(audio_path))
        seg = seg.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)

        sample_width = seg.sample_width
        if sample_width == 1:
            samples = (samples - 128.0) / 128.0
        elif sample_width == 2:
            samples /= 32768.0
        elif sample_width == 3:
            samples /= 8388608.0
        elif sample_width == 4:
            samples /= 2147483648.0
        return samples.astype(np.float32)
    except Exception:
        pass

    try:
        audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception:
        pass

    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Unable to decode audio file: {audio_path} ({exc})") from exc


def calculate_metrics(df: pd.DataFrame) -> dict:
    actuals = df["actual_label"]
    preds = df["predicted_label"]

    tp = len(df[(actuals == "AI") & (preds == "AI")])
    tn = len(df[(actuals == "HUMAN") & (preds == "HUMAN")])
    fp = len(df[(actuals == "HUMAN") & (preds == "AI")])
    fn = len(df[(actuals == "AI") & (preds == "HUMAN")])

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_inference_time = df["inference_time_sec"].mean() if "inference_time_sec" in df else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "avg_inference_time_sec": avg_inference_time
    }


def evaluate_single_model(model_id: str, sample_files: list) -> pd.DataFrame:
    print(f"\nEvaluating Model: {model_id}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    results = []

    for path, actual_label in sample_files:
        try:
            start_time = time.time()
            speech = load_audio_robust(path, target_sr=SAMPLE_RATE)

            inputs = feature_extractor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt")

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()

            predicted_idx = int(torch.argmax(logits, dim=-1).item())
            native_label = model.config.id2label.get(predicted_idx, f"LABEL_{predicted_idx}")
            predicted_label = normalize_audio_label(model_id, native_label)
            
            predicted_score = probs[predicted_idx] if isinstance(probs, list) else probs
            inference_time = time.time() - start_time

            results.append({
                "file": path.name,
                "file_path": str(path),
                "actual_label": actual_label,
                "native_label": native_label,
                "predicted_label": predicted_label,
                "score": round(float(predicted_score), 4),
                "correct": (actual_label == predicted_label),
                "inference_time_sec": round(inference_time, 4)
            })
        except Exception as e:
            print(f"FAILED to process file {path.name}: {e}")

    return pd.DataFrame(results)


def run_evaluation():
    print("=" * 60)
    print("STARTING DUAL AUDIO DETECTOR EVALUATION (ROBUST LOADERS)")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory {DATASET_DIR} does not exist.")
        sys.exit(1)

    ai_files = iter_supported_audio_files(DATASET_DIR / "ai")
    human_files = iter_supported_audio_files(DATASET_DIR / "human")

    sample_files = sorted(ai_files + human_files, key=lambda x: x[0].name)

    if not sample_files:
        print("Error: No audio files found in data/audio/evaluation/(ai|human)")
        sys.exit(1)

    print(f"Found {len(sample_files)} total audio files ({len(ai_files)} AI, {len(human_files)} HUMAN).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate Model 1
    df_m1 = evaluate_single_model(MODEL_1_ID, sample_files)
    csv_m1_path = RESULTS_DIR / "audio_detector_model_1_results.csv"
    df_m1.to_csv(csv_m1_path, index=False)
    m1_metrics = calculate_metrics(df_m1)

    # Evaluate Model 2
    df_m2 = evaluate_single_model(MODEL_2_ID, sample_files)
    csv_m2_path = RESULTS_DIR / "audio_detector_model_2_results.csv"
    df_m2.to_csv(csv_m2_path, index=False)
    m2_metrics = calculate_metrics(df_m2)

    # Comparison summary
    comparison_data = [
        {"Model": MODEL_1_ID, **m1_metrics},
        {"Model": MODEL_2_ID, **m2_metrics}
    ]
    df_comp = pd.DataFrame(comparison_data)
    csv_comp_path = RESULTS_DIR / "audio_detector_model_comparison.csv"
    df_comp.to_csv(csv_comp_path, index=False)

    print("\n" + "=" * 60)
    print("AUDIO DETECTOR BENCHMARK RESULTS")
    print("=" * 60)
    
    for m in comparison_data:
        print(f"Model: {m['Model']}")
        print(f"  Samples Evaluated : {m['total']}")
        print(f"  Accuracy          : {m['accuracy'] * 100:.2f}%")
        print(f"  Precision         : {m['precision'] * 100:.2f}%")
        print(f"  Recall            : {m['recall'] * 100:.2f}%")
        print(f"  Specificity       : {m['specificity'] * 100:.2f}%")
        print(f"  F1-Score          : {m['f1'] * 100:.2f}%")
        print(f"  TP / TN / FP / FN : {m['tp']} / {m['tn']} / {m['fp']} / {m['fn']}")
        print(f"  Avg Time / Sample : {m['avg_inference_time_sec']:.4f}s")
        print("-" * 60)

    print(f"\nSaved detailed evaluation outputs to:")
    print(f" - {csv_m1_path}")
    print(f" - {csv_m2_path}")
    print(f" - {csv_comp_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_evaluation()