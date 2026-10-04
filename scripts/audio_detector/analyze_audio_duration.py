import sys
from pathlib import Path

import librosa
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
DATASET_DIR = REPO_ROOT / "data" / "audio" / "evaluation"
M1_CSV = RESULTS_DIR / "audio_detector_model_1_results.csv"
M2_CSV = RESULTS_DIR / "audio_detector_model_2_results.csv"

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg"}


def main():
    print("=" * 60)
    print("FIXED AUDIO DURATION VS INFERENCE TIME ANALYSIS")
    print("=" * 60)

    if not M1_CSV.exists() or not M2_CSV.exists():
        print(f"Error: Evaluation results not found in {RESULTS_DIR}")
        sys.exit(1)

    df_m1 = pd.read_csv(M1_CSV)
    df_m2 = pd.read_csv(M2_CSV)

    durations = []
    for actual_class in ["ai", "human"]:
        class_dir = DATASET_DIR / actual_class
        if not class_dir.exists():
            continue
        for file_path in class_dir.glob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    duration_sec = librosa.get_duration(path=file_path)
                    durations.append(
                        {
                            "file_path": str(file_path.resolve()),
                            "file": file_path.name,
                            "duration_sec": round(float(duration_sec), 2),
                        }
                    )
                except Exception as exc:
                    print(f"Could not compute duration for {file_path.name}: {exc}")

    df_durations = pd.DataFrame(durations)

    if df_durations.empty:
        print("Error: No audio files located for duration analysis.")
        sys.exit(1)

    df_m1 = df_m1[["file", "actual_label", "inference_time_sec"]].drop_duplicates(subset=["file"]).copy()
    df_m2 = df_m2[["file", "inference_time_sec"]].drop_duplicates(subset=["file"]).copy()
    df_durations = df_durations[["file", "duration_sec"]].drop_duplicates(subset=["file"]).copy()

    merged = pd.merge(df_m1, df_durations, on="file", how="inner")
    merged = pd.merge(
        merged,
        df_m2,
        on="file",
        how="inner",
        suffixes=("_m1", "_m2"),
    )

    merged_sorted = merged.sort_values(by="inference_time_sec_m1", ascending=False)

    print("\n--- TOP 10 LONGEST INFERENCE TIME SAMPLES ---")
    disp_cols = ["file", "actual_label", "duration_sec", "inference_time_sec_m1", "inference_time_sec_m2"]
    print(merged_sorted[disp_cols].head(10).to_string(index=False))

    corr_m1 = merged["duration_sec"].corr(merged["inference_time_sec_m1"])
    corr_m2 = merged["duration_sec"].corr(merged["inference_time_sec_m2"])

    print("\n--- SUMMARY STATISTICS ---")
    print(f"Total Unique Analyzed Files            : {len(merged)}")
    print(f"Average Audio Duration                 : {merged['duration_sec'].mean():.2f}s")
    print(f"Fixed Correlation (Duration vs M1 Latency) : {corr_m1:.4f}")
    print(f"Fixed Correlation (Duration vs M2 Latency) : {corr_m2:.4f}")

    output_path = RESULTS_DIR / "audio_detector_duration_analysis_fixed.csv"
    merged_sorted.to_csv(output_path, index=False)
    print(f"\nSaved fixed duration analysis to {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()