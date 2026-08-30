import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector

def run_test():
    print("Initializing Text AIDetector...")
    detector = TextAIDetector()
    
    sample_texts = [
        "This is a human-written sample intended to test the base functionality of our text detector.",
        "Artificial intelligence systems generate text by predicting the most probable next token given a context window."
    ]

    for idx, sample in enumerate(sample_texts, 1):
        print(f"\n--- Running Test {idx} ---")
        result = detector.detect(sample)
        print(f"Prediction : {result.prediction}")
        print(f"AI Score   : {result.ai_score}")
        print(f"Human Score: {result.human_score}")
        print(f"Metadata   : {result.metadata}")

if __name__ == "__main__":
    run_test()