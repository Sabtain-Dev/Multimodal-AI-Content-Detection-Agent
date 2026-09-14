import gc
import os
import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "text" / "evaluation"
OUTPUT_CSV = PROJECT_ROOT / "results" / "three_models_comparison.csv"

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def evaluate_single_model(model_name: str, threshold: float, samples: list[dict]) -> list[dict]:
    print(f"\n--- Loading and Evaluating: {model_name} (Threshold: {threshold}) ---")
    detector = TextAIDetector(model_name=model_name, threshold=threshold)
    
    results = []
    for item in samples:
        text = item["text"]
        res = detector.detect(text)
        results.append({
            "ai_score": res.ai_score,
            "prediction": res.prediction  # 'likely_ai_generated' or 'likely_human'
        })
    
    # Unload model and release memory
    del detector
    clean_memory()
    return results

def main():
    dataset_frames = []
    for label, folder_name in (("human", "human"), ("ai", "ai")):
        dataset_path = DATASET_DIR / folder_name / f"{folder_name}.csv"
        if not dataset_path.exists():
            print(f"Error: Dataset not found at {dataset_path}")
            sys.exit(1)

        frame = pd.read_csv(dataset_path)
        frame["label"] = label
        dataset_frames.append(frame)

    df_dataset = pd.concat(dataset_frames, ignore_index=True)
    samples = df_dataset.to_dict(orient="records")
    print(f"Loaded {len(samples)} samples from {DATASET_DIR}")

    # 1. Evaluate TMR (RoBERTa)
    tmr_results = evaluate_single_model(
        model_name="Oxidane/tmr-ai-text-detector",
        threshold=0.70,
        samples=samples
    )

    # 2. Evaluate Multilingual (XLM-RoBERTa ONNX)
    multi_results = evaluate_single_model(
        model_name="mujian2026/multilingual-ai-text-detector",
        threshold=0.50,
        samples=samples
    )

    # 3. Evaluate Gradient (DeBERTa-v3-large)
    gradient_results = evaluate_single_model(
        model_name="ShantanuT01/gradient-ai-text-detector",
        threshold=0.50,
        samples=samples
    )

    # Combine results
    combined_rows = []
    for idx, row in df_dataset.iterrows():
        t_res = tmr_results[idx]
        m_res = multi_results[idx]
        g_res = gradient_results[idx]

        label = row.get("label", row.get("actual_label", "unknown"))

        # Determine 2-Model Ensemble Output (TMR + Multilingual)
        if t_res["prediction"] == m_res["prediction"]:
            v2_pred = t_res["prediction"]
        else:
            v2_pred = "uncertain"

        # Determine 3-Model Majority Vote (TMR + Multilingual + Gradient)
        preds = [t_res["prediction"], m_res["prediction"], g_res["prediction"]]
        ai_votes = preds.count("likely_ai_generated")
        human_votes = preds.count("likely_human")

        if ai_votes >= 2:
            v3_majority = "likely_ai_generated"
        else:
            v3_majority = "likely_human"

        combined_rows.append({
            "row_id": idx + 1,
            "actual_label": label,
            "text_snippet": row["text"][:80].replace("\n", " "),
            "tmr_score": t_res["ai_score"],
            "tmr_pred": t_res["prediction"],
            "multi_score": m_res["ai_score"],
            "multi_pred": m_res["prediction"],
            "gradient_score": g_res["ai_score"],
            "gradient_pred": g_res["prediction"],
            "v2_ensemble_pred": v2_pred,
            "v3_majority_pred": v3_majority,
            "three_way_agreement": (t_res["prediction"] == m_res["prediction"] == g_res["prediction"])
        })

    res_df = pd.DataFrame(combined_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved 3-model detailed comparison to {OUTPUT_CSV}")

    # Summary Analysis
    print("\n================ THREE-MODEL COMPARISON ANALYSIS ================")
    print(f"Total Samples Evaluated: {len(res_df)}")
    print(f"Full 3-Way Agreement Count: {res_df['three_way_agreement'].sum()} / {len(res_df)}")
    
    # Resolution of the 34 V2 Disagreements
    v2_uncertain = res_df[res_df["v2_ensemble_pred"] == "uncertain"]
    print(f"\n--- Analysis of 2-Model Disagreements ({len(v2_uncertain)} samples) ---")
    
    resolved_as_ai = (v2_uncertain["gradient_pred"] == "likely_ai_generated").sum()
    resolved_as_human = (v2_uncertain["gradient_pred"] == "likely_human").sum()
    print(f"Gradient voted AI    (Broke tie to AI)   : {resolved_as_ai}")
    print(f"Gradient voted HUMAN (Broke tie to HUMAN): {resolved_as_human}")

    # Metrics helper
    def calc_metrics(preds, targets):
        tp = sum(1 for p, t in zip(preds, targets) if p == "likely_ai_generated" and t in ["ai", "ai_generated", "likely_ai_generated"])
        fp = sum(1 for p, t in zip(preds, targets) if p == "likely_ai_generated" and t in ["human", "likely_human"])
        tn = sum(1 for p, t in zip(preds, targets) if p == "likely_human" and t in ["human", "likely_human"])
        fn = sum(1 for p, t in zip(preds, targets) if p == "likely_human" and t in ["ai", "ai_generated", "likely_ai_generated"])
        
        acc = (tp + tn) / len(targets) if targets else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return tp, fp, tn, fn, acc, prec, rec, f1

    labels = res_df["actual_label"].str.lower().tolist()
    
    g_tp, g_fp, g_tn, g_fn, g_acc, g_prec, g_rec, g_f1 = calc_metrics(res_df["gradient_pred"].tolist(), labels)
    v3_tp, v3_fp, v3_tn, v3_fn, v3_acc, v3_prec, v3_rec, v3_f1 = calc_metrics(res_df["v3_majority_pred"].tolist(), labels)

    print("\n--- Model Performance Metrics ---")
    print(f"Gradient Alone   -> Acc: {g_acc:.4f} | Prec: {g_prec:.4f} | Rec: {g_rec:.4f} | F1: {g_f1:.4f} (TP:{g_tp}, FP:{g_fp}, TN:{g_tn}, FN:{g_fn})")
    print(f"3-Model Majority -> Acc: {v3_acc:.4f} | Prec: {v3_prec:.4f} | Rec: {v3_rec:.4f} | F1: {v3_f1:.4f} (TP:{v3_tp}, FP:{v3_fp}, TN:{v3_tn}, FN:{v3_fn})")
    print("=================================================================")

if __name__ == "__main__":
    main()