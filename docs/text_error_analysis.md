# Text Detector Error Analysis

## Model

`mujian2026/multilingual-ai-text-detector`

## Evaluation Dataset

| Class | Samples |
| --- | ---: |
| Human | 12 |
| AI | 10 |
| **Total** | **22** |

## Overall Results

Evaluation used an AI-score threshold of `0.50`. Scores at or above the threshold were classified as AI-generated.

| Metric | Result |
| --- | ---: |
| Accuracy | 95.45% |
| Precision | 100.00% |
| Recall | 90.00% |
| F1 score | 94.74% |
| False positives | 0 |
| False negatives | 1 |

The model produced no false positives and missed one AI sample.

## False Negative Analysis

| Field | Value |
| --- | --- |
| Dataset file | `ai.csv` |
| Row | 8 |
| Actual label | AI |
| Predicted label | HUMAN |
| AI score | 0.0171 |
| Token count | 40 |
| Text type | Technical / instruction-based (preliminary) |

### Observation

The model classified this AI-generated sample as human with high confidence. Because the evaluation set contains only one comparable false negative, the cause cannot be established from this result alone.

### Possible Factors

- Technical or instruction-based writing style
- Short sample length
- Code-related terminology

These factors are hypotheses rather than confirmed causes. More samples with similar characteristics are required for a reliable error analysis.

## Model Comparison

The same 22-record dataset was also used to compare the multilingual model with `Oxidane/tmr-ai-text-detector`.

| Model | Best threshold | Accuracy | Precision | Recall | F1 score | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TMR | 0.70-0.90 | 68.18% | 58.82% | 100.00% | 74.07% | 7 | 0 |
| Multilingual | 0.50-0.95 | 95.45% | 100.00% | 90.00% | 94.74% | 0 | 1 |

The multilingual model achieved the strongest overall balance on this dataset. The result is indicative only because the evaluation set is small and may not represent other domains, languages, or text-generation systems.

## Reproduction

Run these commands from the repository root with the project virtual environment activated:

```bash
source .venv/bin/activate
python3 scripts/inspect_text_detector_2.py
python3 scripts/compare_text_models.py
```

The scripts print the evaluation summary and write comparison CSV files to the `results/` directory. Hugging Face authentication is optional, but setting `HF_TOKEN` avoids unauthenticated-request warnings and provides higher Hub rate limits.