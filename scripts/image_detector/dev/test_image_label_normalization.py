import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.detectors.image.label_normalization import (
    build_normalized_prediction,
    normalize_image_label,
)


def main() -> None:
    assert normalize_image_label("ai") == "AI"
    assert normalize_image_label("human") == "HUMAN"

    assert normalize_image_label("artificial") == "AI"
    assert normalize_image_label("real") == "HUMAN"

    result = build_normalized_prediction(
        model_name="Ateeqq/ai-vs-human-image-detector",
        native_label="ai",
        ai_score=0.92,
    )

    assert result["prediction"] == "AI"
    assert result["ai_score"] == 0.92
    assert result["human_score"] == 0.08

    result = build_normalized_prediction(
        model_name="haywoodsloan/ai-image-detector-deploy",
        native_label="real",
        ai_score=0.13,
    )

    assert result["prediction"] == "HUMAN"
    assert result["ai_score"] == 0.13
    assert result["human_score"] == 0.87

    print("All image label-normalization tests passed.")


if __name__ == "__main__":
    main()