import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformers import AutoConfig, AutoTokenizer

from src.detectors.text.detector import TextAIDetector


MODEL_NAME = "ShantanuT01/gradient-ai-text-detector"
AI_THRESHOLD = 0.50


def inspect_model():
    print("====================================================")
    print(f"       INSPECTING MODEL C: {MODEL_NAME}")
    print("====================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    print("[Configuration Details]")
    print(f"id2label                    : {config.id2label}")
    print(f"label2id                    : {config.label2id}")
    print(f"tokenizer.model_max_length  : {tokenizer.model_max_length}\n")

    detector = TextAIDetector(model_name=MODEL_NAME, threshold=AI_THRESHOLD)
    samples = {
        "HUMAN": "In this paper, we explore the application of neural networks in medical imaging.",
        "AI": "Certainly! Here is a comprehensive overview of machine learning algorithms in modern healthcare.",
    }

    print("------------------ Single Inference Tests ------------------")
    for sample_type, sample_text in samples.items():
        result = detector.detect(sample_text)
        print(f"\nSample Type : {sample_type}")
        print(f"Text        : \"{sample_text}\"")
        print(f"AI Score    : {result.ai_score:.4f}")
        print(f"Human Score : {result.human_score:.4f}")
        print(f"Prediction  : {result.prediction}")
        print(f"Chunks      : {result.chunks_analyzed}")

    print("\n====================================================")


if __name__ == "__main__":
    inspect_model()