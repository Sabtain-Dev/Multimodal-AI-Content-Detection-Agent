import sys
from pathlib import Path

# Add script directory to import pipeline functions
sys.path.append(str(Path(__file__).parent))
from pipeline import process_image_detection


def run_pipeline_unit_tests():
    print("=" * 60)
    print("RUNNING IMAGE PIPELINE UNIT TESTS")
    print("=" * 60)

    # Test Case 1: Both models agree on AI
    t1 = process_image_detection("sample_ai.jpg", m1_score=0.92, m2_score=0.95)
    assert t1["user_result"]["label"] == "AI"
    assert t1["user_result"]["confidence_indicator"] == "High"
    assert t1["user_result"]["label"] != "UNCERTAIN"
    assert t1["internal_telemetry"]["models_agree"] is True
    print("[PASS] Test Case 1: Consensus AI")

    # Test Case 2: Both models agree on HUMAN
    t2 = process_image_detection("sample_human.jpg", m1_score=0.10, m2_score=0.05)
    assert t2["user_result"]["label"] == "HUMAN"
    assert t2["user_result"]["confidence_indicator"] == "High"
    assert t2["user_result"]["label"] != "UNCERTAIN"
    assert t2["internal_telemetry"]["models_agree"] is True
    print("[PASS] Test Case 2: Consensus HUMAN")

    # Test Case 3: Models disagree (M1 = AI, M2 = HUMAN) -> Prefer M2
    t3 = process_image_detection("disagreement_01.jpg", m1_score=0.82, m2_score=0.15)
    assert t3["user_result"]["label"] == "HUMAN"
    assert t3["user_result"]["confidence_indicator"] == "Moderate"  # Capped due to disagreement
    assert t3["user_result"]["label"] != "UNCERTAIN"
    assert t3["internal_telemetry"]["models_agree"] is False
    assert t3["internal_telemetry"]["decision_mode"] == "model_2_primary_fallback"
    print("[PASS] Test Case 3: Disagreement Fallback to Model 2 (HUMAN)")

    # Test Case 4: Models disagree (M1 = HUMAN, M2 = AI) -> Prefer M2
    t4 = process_image_detection("disagreement_02.jpg", m1_score=0.20, m2_score=0.88)
    assert t4["user_result"]["label"] == "AI"
    assert t4["user_result"]["confidence_indicator"] == "Moderate"  # Capped due to disagreement
    assert t4["user_result"]["label"] != "UNCERTAIN"
    assert t4["internal_telemetry"]["models_agree"] is False
    assert t4["internal_telemetry"]["decision_mode"] == "model_2_primary_fallback"
    print("[PASS] Test Case 4: Disagreement Fallback to Model 2 (AI)")

    print("-" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline_unit_tests()