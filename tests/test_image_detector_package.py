from src.detectors.image import ImageDetector, evaluate_image_ensemble


def test_image_detector_public_api_imports():
    detector = ImageDetector()
    result = detector.predict("test.jpg", 0.8, 0.2)

    assert result["modality"] == "image"
    assert result["label"] == "HUMAN"
    assert result["model_2"]["label"] == "HUMAN"


def test_ensemble_uses_valid_optional_defaults():
    result = evaluate_image_ensemble(0.80, 0.20, None, None)
    assert result["label"] == "HUMAN"
    assert result["decision_mode"] == "model_2_primary_fallback"
