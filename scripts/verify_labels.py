import sys
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

MODEL_NAME = "Oxidane/tmr-ai-text-detector"

def verify_pipeline():
    print(f"--- Step 1: Loading Model Config for {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    print("\n[Config Label Mappings]")
    print(f"id2label : {model.config.id2label}")
    print(f"label2id : {model.config.label2id}")

    sample_human = "In this paper, we explore the application of neural networks in medical imaging. We evaluate the performance of convolutional neural networks (CNNs) in detecting anomalies in MRI scans, comparing their accuracy to traditional image processing techniques."
    sample_ai = "Certainly! Here is a comprehensive overview of machine learning algorithms in modern healthcare. So, let's delve into the various types of algorithms, their applications, and the impact they have on patient care and medical research."

    print("\n--- Step 2: Testing Raw Output Inference ---")
    
    for label, text in [("Human", sample_human), ("AI", sample_ai)]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        print(f"\nSample Type : {label}")
        print(f"Text        : \"{text}\"")
        print(f"Raw Logits  : {logits.numpy().tolist()}")
        print(f"Probs       : Index 0 = {probs[0].item():.4f}, Index 1 = {probs[1].item():.4f}")

if __name__ == "__main__":
    verify_pipeline()