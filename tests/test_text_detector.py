import pytest
from src.detectors.text.detector import TextAIDetector


def to_label_value(prediction: str) -> int:
    return 1 if prediction == "likely_ai_generated" else 0


@pytest.mark.slow
def test_detector_inference():
    print("\nInitializing Text AIDetector...")
    detector = TextAIDetector()

    sample_texts = [
        (1, "This is a human-written sample intended to test the base functionality of our text detector."),
        (1, "Artificial intelligence systems generate text by predicting the most probable next token given a context window."),
        (0, "Docker is not working properly due to some ram/memory issues as it needs at least 16 GB of memory to work smooth along with other applications or keep everything closed while using docker for better performance."),
        (0, "Tell me i am human or an ai assistant, judge with this sentence: Chatgpt is better than claude ai because it provides proper solution to our problems and will understand better and keep its context concise and clear which helps us to learn new things with proper guide."),
        (0, "The software companies think that we are machines, they treated us like ai model which have the ability to learn everything and have the ability to make each information in his context, By using this knowledge they solve problems effectively but faster than humans. Companies should realize the ai models are created by humans so, one to two wrong answers cannot decide that this person is not useful in market.")
    ]

    for idx, (true_label, sample) in enumerate(sample_texts, 1):
        print(f"\n--- Running Test {idx} ---")
        result = detector.detect(sample)
        predicted_label = to_label_value(result.prediction)
        
        # Standard pytest assertion
        assert result.status == "success"
        assert 0.0 <= result.ai_score <= 1.0
        assert 0.0 <= result.human_score <= 1.0

        print(f"True Label      : {true_label} ({'human' if true_label == 0 else 'ai generated'})")
        print(f"Predicted Label : {predicted_label} ({'human' if predicted_label == 0 else 'ai generated'})")
        print(f"Prediction      : {result.prediction}")
        print(f"AI Score        : {result.ai_score}")
        print(f"Human Score     : {result.human_score}")
        print(f"Matches True?   : {predicted_label == true_label}")
        print(f"Metadata        : {result.metadata}")