import sys
from pathlib import Path
import pandas as pd
import torch
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.detectors.text.preprocessing import validate_and_clean_text, chunk_text_by_tokens

MODELS = {
    "TMR": "Oxidane/tmr-ai-text-detector",
    "Multilingual": "mujian2026/multilingual-ai-text-detector"
}
MODEL_SUBFOLDER = "fp32"

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

def get_ai_score_for_chunk(model, tokenizer, chunk_text, ai_label_index, device, session=None):
    if session is not None:
        inputs = tokenizer(chunk_text, return_tensors="np", truncation=True, max_length=512)
        logits = torch.from_numpy(session.run(["logits"], inputs)[0])
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        token_count = inputs["input_ids"].shape[1]
        return float(probs[ai_label_index].item()), token_count

    inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    return float(probs[ai_label_index].cpu().item()), inputs["input_ids"].shape[1]

def evaluate_model_on_dataset(model_key, model_name, base_dir, device):
    print(f"\n---> Evaluating {model_key} ({model_name})...")
    session = None
    if model_key == "Multilingual":
        tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=MODEL_SUBFOLDER)
        config = AutoConfig.from_pretrained(model_name, subfolder=MODEL_SUBFOLDER)
        model_path = hf_hub_download(
            model_name, filename=f"{MODEL_SUBFOLDER}/onnx/model.onnx"
        )
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        config = model.config

    # Determine AI label index safely from config
    id2label = config.id2label
    ai_index = 1
    for idx, label in id2label.items():
        if "ai" in str(label).lower() or "generated" in str(label).lower():
            ai_index = int(idx)
            break

    records = []
    
    def process_folder(folder_path, actual_label):
        for csv_file in sorted(folder_path.glob("*.csv")):
            df = pd.read_csv(csv_file)
            if "text" not in df.columns:
                continue
            for idx, row in df.iterrows():
                text = str(row["text"]).strip()
                if not text:
                    continue
                
                cleaned = validate_and_clean_text(text)
                chunks = chunk_text_by_tokens(cleaned, tokenizer, chunk_size=480, overlap=32)
                
                chunk_scores = []
                total_tokens = 0
                for c in chunks:
                    score, t_count = get_ai_score_for_chunk(
                        model, tokenizer, c, ai_index, device, session
                    )
                    chunk_scores.append(score)
                    total_tokens += t_count

                avg_ai_score = sum(chunk_scores) / len(chunk_scores)
                records.append({
                    "model": model_key,
                    "file_name": csv_file.name,
                    "row_index": idx + 1,
                    "actual_label": actual_label,
                    "ai_score": round(avg_ai_score, 4),
                    "token_count": total_tokens,
                    "text_snippet": cleaned[:70] + "..."
                })

    process_folder(base_dir / "human", "HUMAN")
    process_folder(base_dir / "ai", "AI")

    # Cleanup memory before running next model
    del model
    del tokenizer
    del session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(records)

def run_comparison():
    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root / "data" / "text" / "evaluation"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_raw_dfs = []
    for m_key, m_name in MODELS.items():
        raw_df = evaluate_model_on_dataset(m_key, m_name, base_dir, device)
        raw_df.to_csv(results_dir / f"{m_key.lower()}_raw_results.csv", index=False)
        all_raw_dfs.append(raw_df)

    combined_raw = pd.concat(all_raw_dfs, ignore_index=True)

    # Threshold Sweeps Comparison
    sweep_records = []
    for m_key in MODELS.keys():
        m_df = combined_raw[combined_raw["model"] == m_key]
        for t in THRESHOLDS:
            tp, fp, tn, fn = 0, 0, 0, 0
            for _, row in m_df.iterrows():
                pred = "AI" if row["ai_score"] >= t else "HUMAN"
                act = row["actual_label"]
                if act == "AI" and pred == "AI": tp += 1
                elif act == "HUMAN" and pred == "AI": fp += 1
                elif act == "HUMAN" and pred == "HUMAN": tn += 1
                elif act == "AI" and pred == "HUMAN": fn += 1

            total = tp + fp + tn + fn
            acc = (tp + tn) / total if total > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            sweep_records.append({
                "Model": m_key,
                "Threshold": t,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "FP": fp,
                "FN": fn
            })

    comparison_df = pd.DataFrame(sweep_records)
    comparison_df.to_csv(results_dir / "models_threshold_comparison.csv", index=False)

    print("\n================ MODEL COMPARISON SUMMARY ================")
    print(comparison_df.to_string(index=False))
    print("==========================================================")

if __name__ == "__main__":
    run_comparison()