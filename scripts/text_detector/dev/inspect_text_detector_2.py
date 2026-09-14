import sys
import importlib
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

ort = importlib.import_module("onnxruntime")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.detectors.text.preprocessing import validate_and_clean_text, chunk_text_by_tokens

MODEL_NAME = "mujian2026/multilingual-ai-text-detector"
MODEL_SUBFOLDER = "fp32"
AI_THRESHOLD = 0.50

def inspect_model_b():
    print("====================================================")
    print(f"       INSPECTING MODEL B: {MODEL_NAME}")
    print("====================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder=MODEL_SUBFOLDER)
    config = AutoConfig.from_pretrained(MODEL_NAME, subfolder=MODEL_SUBFOLDER)
    model_path = hf_hub_download(
        MODEL_NAME, filename=f"{MODEL_SUBFOLDER}/onnx/model.onnx"
    )
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("[Configuration Details]")
    print(f"id2label              : {config.id2label}")
    print(f"label2id              : {config.label2id}")
    print(f"tokenizer.model_max_length : {tokenizer.model_max_length}\n")

    sample_human = "In this paper, we explore the application of neural networks in medical imaging."
    sample_ai = "Certainly! Here is a comprehensive overview of machine learning algorithms in modern healthcare."

    print("------------------ Single Inference Tests ------------------")
    for sample_type, sample_text in [("HUMAN", sample_human), ("AI", sample_ai)]:
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
        onnx_inputs = {name: value.numpy() for name, value in inputs.items()}
        logits = torch.from_numpy(session.run(["logits"], onnx_inputs)[0])
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        print(f"\nSample Type : {sample_type}")
        print(f"Text        : \"{sample_text}\"")
        print(f"Raw Logits  : {logits.numpy().tolist()}")
        print(f"Probabilities:")
        for idx, prob in enumerate(probs):
            label_str = config.id2label.get(idx, f"LABEL_{idx}")
            print(f"  Index {idx} ({label_str}) : {prob.item():.4f}")

    evaluate_dataset(tokenizer, config, session)
    print("\n====================================================")


def evaluate_dataset(tokenizer, config, session):
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / "text" / "evaluation"
    ai_index = next(
        (int(index) for index, label in config.id2label.items()
         if "ai" in str(label).lower() or "generated" in str(label).lower()),
        1,
    )
    records = []

    for actual_label, folder_name in [("HUMAN", "human"), ("AI", "ai")]:
        for csv_path in sorted((dataset_dir / folder_name).glob("*.csv")):
            dataframe = pd.read_csv(csv_path)
            if "text" not in dataframe.columns:
                continue

            for row_index, row in dataframe.iterrows():
                text = str(row["text"]).strip()
                if not text:
                    continue

                cleaned = validate_and_clean_text(text)
                chunks = chunk_text_by_tokens(cleaned, tokenizer, chunk_size=480, overlap=32)
                chunk_scores = []
                token_count = 0

                for chunk in chunks:
                    inputs = tokenizer(
                        chunk, return_tensors="np", truncation=True, max_length=512
                    )
                    logits = torch.from_numpy(session.run(["logits"], inputs)[0])
                    probabilities = torch.softmax(logits, dim=-1).squeeze(0)
                    chunk_scores.append(float(probabilities[ai_index].item()))
                    token_count += inputs["input_ids"].shape[1]

                ai_score = sum(chunk_scores) / len(chunk_scores)
                prediction = "AI" if ai_score >= AI_THRESHOLD else "HUMAN"
                records.append({
                    "actual": actual_label,
                    "prediction": prediction,
                    "ai_score": ai_score,
                    "tokens": token_count,
                    "file": csv_path.name,
                    "row": row_index + 1,
                })

    print("\n------------------ Dataset Evaluation ------------------")
    print(f"Threshold: {AI_THRESHOLD:.2f}")
    for record in records:
        print(
            f"{record['file']} row {record['row']:>2}: "
            f"actual={record['actual']:<5} predicted={record['prediction']:<5} "
            f"ai_score={record['ai_score']:.4f} tokens={record['tokens']}"
        )

    true_positive = sum(r["actual"] == "AI" and r["prediction"] == "AI" for r in records)
    false_positive = sum(r["actual"] == "HUMAN" and r["prediction"] == "AI" for r in records)
    true_negative = sum(r["actual"] == "HUMAN" and r["prediction"] == "HUMAN" for r in records)
    false_negative = sum(r["actual"] == "AI" and r["prediction"] == "HUMAN" for r in records)
    total = len(records)
    accuracy = (true_positive + true_negative) / total if total else 0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(
        f"Summary: records={total} accuracy={accuracy:.4f} precision={precision:.4f} "
        f"recall={recall:.4f} f1={f1:.4f} FP={false_positive} FN={false_negative}"
    )

if __name__ == "__main__":
    inspect_model_b()